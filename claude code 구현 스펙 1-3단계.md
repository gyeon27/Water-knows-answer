# 구현 요청: 차원 가변 유체 라우팅 시스템 — 신경망 없는 부분 (Phase 1)

## 목표

Taichi(Python) 기반으로 입자 유체 시뮬레이션에 **위상 기반 적응형 솔버 라우팅**을 구현한다. 이 단계는 신경망(GNN)을 전혀 포함하지 않으며, 이후 SPLASH 영역에 신경망을 붙이기 전 라우팅 로직 자체의 안정성을 검증하는 것이 목적이다.

**중요**: SPLASH 상태로 분류된 입자는 이번 단계에서 임시로 단순 물리(중력 낙하 + 입자간 충돌 반발력) 정도로만 처리한다. 이 부분은 나중에 GNN으로 교체될 자리이므로 정교하게 만들 필요 없음 — 단, 인터페이스(입출력 데이터 형태)는 나중에 GNN으로 바로 교체 가능하도록 함수로 분리해둘 것.

## 기술 스택

- Python 3.10+, Taichi (`pip install taichi`)
- 시각화: Taichi GGUI (실시간 3D 뷰어)
- 베이스 시뮬레이터: SPH 또는 Taichi 공식 예제(`taichi_elements`)의 MPM/SPH 유체 예제를 참고해 기본 입자 시스템 먼저 구축

## 파일 구조

```
fluid_routing/
├── main.py                 # 진입점, 시뮬레이션 루프, GGUI 시각화
├── particles.py             # 입자 상태 (위치, 속도, 질량, 상태 라벨)
├── grid.py                  # 성긴 보조 격자 바인딩
├── features.py               # 4개 특징량 계산 (정렬도, 밀도, 발산, 정체도)
├── routing.py                # 이중 임계값 분류 + 히스테리시스 + 전파 마진
├── solvers/
│   ├── stream_solver.py     # 1D 파라메트릭 적분
│   ├── pool_solver.py       # 2D shallow water
│   └── splash_solver.py     # 임시 단순 물리 (추후 GNN으로 교체)
├── blending.py                # 전환 경계 블렌딩
├── config.py                  # 모든 임계값/하이퍼파라미터 한 곳에 모음
└── logging_utils.py           # 프레임별 상태 로깅 (셀별 라벨, 특징량 값)
```

`config.py`에 하이퍼파라미터를 전부 모아두는 이유: 나중에 씬별 튜닝과 ablation 실험을 쉽게 하기 위함.

---

## 1. 입자 시스템 (particles.py)

```python
# 필드 정의 (Taichi field)
position: ti.Vector.field(3, dtype=ti.f32, shape=N)
velocity: ti.Vector.field(3, dtype=ti.f32, shape=N)
mass: ti.field(dtype=ti.f32, shape=N)
state: ti.field(dtype=ti.i32, shape=N)  # 0=STREAM, 1=SPLASH, 2=POOL
cell_index: ti.Vector.field(3, dtype=ti.i32, shape=N)  # 소속 격자 셀
```

N은 초기 5,000~20,000개 수준으로 시작 (프로토타입이므로 대규모 불필요). 초기 씬은 "물기둥이 낙하해서 바닥에 고이는" 단순한 시나리오로 시작할 것 (STREAM→SPLASH→POOL 전이를 한 씬 안에서 다 볼 수 있음).

---

## 2. 성긴 격자 바인딩 (grid.py)

- 격자 셀 크기 `c = 0.5` (config.py에서 조정 가능)
- 매 프레임 입자를 셀에 바인딩 (공간 해싱, Taichi의 `ti.algorithms` 또는 직접 구현)
- 셀별 입자 리스트를 담을 자료구조 필요 (Taichi의 동적 리스트 또는 카운팅 정렬 방식)

```python
@ti.kernel
def bind_particles_to_grid():
    for i in range(N):
        cell = ti.floor(position[i] / cell_size).cast(ti.i32)
        cell_index[i] = cell
        # 카운팅 정렬 방식으로 cell -> particle 리스트 구성
```

---

## 3. 특징량 계산 (features.py)

셀 단위로 4개 지표 계산. 아래 수식 그대로 구현.

### 3.1 수직 정렬도
```
A_stream(k) = mean_i∈P_k( |v_i · g_hat| / (|v_i| + eps) )
```
- `g_hat = (0, -1, 0)`, `eps = 1e-4`

### 3.2 이웃 밀집도
```
N_density(k) = |P_k| / cell_volume
```

### 3.3 속도장 발산 + 밀도 변화율
```
D_splash(k) = |mean_divergence(k)| + lambda * |(rho_k_t - rho_k_t-1) / dt|
```
- `mean_divergence(k)`: 6-이웃 셀과의 평균 속도 차분으로 근사
  ```
  div ≈ sum_{j in 6-neighbors(k)} (v_bar_j - v_bar_k)·n_kj / cell_size
  ```
- `lambda = 0.5` (config에서 조정)
- 이전 프레임 밀도값(`rho_k_t-1`)은 프레임마다 버퍼에 저장해뒀다가 다음 프레임에서 참조

### 3.4 정체도 (시간창 필요)
```
S_pool(k) = (moving_avg(|v|, N_win) < v_th) AND (max-min of height over N_win < h_th)
```
- `N_win = 8` 프레임
- `v_th = 0.05`, `h_th = 0.01` (config)
- 셀별로 최근 8프레임의 평균 속력과 높이(대표 y좌표)를 링버퍼에 저장

**구현 참고**: 시간창이 필요한 지표는 셀 ID가 프레임 간 안정적으로 매핑되어야 함 — 격자가 월드 고정(world-fixed)이므로 셀 좌표 자체가 ID 역할을 하면 됨 (파티클처럼 셀도 사라지거나 생기지 않으므로 간단).

---

## 4. 라우팅 로직 (routing.py)

### 4.1 이중 임계값 상태표

| 상태 | 진입 조건 | 이탈 조건 |
|---|---|---|
| SPLASH | `D_splash > 0.8` | `D_splash < 0.5` |
| STREAM | `A_stream > 0.85 AND N_density < N_density_threshold_high` | `A_stream < 0.6` |
| POOL | `S_pool == True`가 4프레임 연속 | `S_pool == False`가 1프레임이라도 발생 |

임계값 사이 구간에서는 셀의 이전 상태 유지 (상태 미변경).

### 4.2 우선순위
동시에 여러 조건 만족 시: `SPLASH > STREAM > POOL`

### 4.3 전파 마진 (Dilation)
SPLASH로 분류된 셀의 1-ring 인접 셀(6방향 또는 26방향, 26 권장)도 SPLASH 후보로 포함 — 모폴로지 팽창 연산.

### 4.4 갱신 주기
- 전체 재분류: 4프레임마다 1회
- 경계 셀(주변에 다른 상태 셀이 있는 셀)만: 매 프레임 재평가

```python
if frame_count % 4 == 0:
    reclassify_all_cells()
else:
    reclassify_boundary_cells_only()
```

셀 상태가 바뀐 프레임 번호(`transition_frame[cell]`)를 반드시 기록해둘 것 — 5단계 블렌딩에서 필요.

---

## 5. 상태별 솔버

### 5.1 STREAM 솔버 (solvers/stream_solver.py)
```
x_{t+1} = x_t + v_t * dt + 0.5 * g * dt^2
```
+ 인접 STREAM 노드 간 거리 구속(간단한 스프링 제약)으로 물줄기 형태 유지.

### 5.2 POOL 솔버 (solvers/pool_solver.py)
얕은 물 방정식(shallow water)을 2D 하이트필드에 유한차분으로 근사:
```
∂h/∂t + ∇·(h*u) = 0
∂u/∂t + (u·∇)u = -g*∇h
```
POOL 상태 입자들의 높이/속도를 하이트필드에 투영 → 방정식 업데이트 → 입자 위치 재투영.

### 5.3 SPLASH 임시 솔버 (solvers/splash_solver.py)
이번 단계에서는 정교할 필요 없음. 단순 중력 + 입자 간 반발력(soft collision) 정도로 구현:
```python
def splash_step_placeholder(particles):
    # 나중에 GNN 추론으로 교체될 자리
    # 지금은: 중력 적용 + 이웃 입자와의 단순 반발력
    ...
```
**함수 시그니처를 명확히 분리**: 입력(이전 위치/속도, 이웃 입자 목록) → 출력(다음 위치/속도) 형태로, 추후 이 함수 내부만 GNN 추론 호출로 교체하면 되도록 설계.

---

## 6. 경계 블렌딩 (blending.py)

상태 전환된 셀은 전환 시점(`transition_frame`)부터 `K=6` 프레임 동안 이전 솔버와 신규 솔버 결과를 블렌딩:
```
w(t) = min(1, (t - transition_frame) / K)
x_blend = (1 - w) * x_prev_solver + w * x_new_solver
```
전환 직후 6프레임 동안만 양쪽 솔버를 동시 실행.

---

## 7. 시각화 및 검증 (main.py)

Taichi GGUI로 실시간 3D 뷰어 구현, 필수 표시 요소:
- 입자를 상태별로 다른 색으로 표시 (STREAM=파랑, SPLASH=빨강, POOL=초록)
- 화면 한쪽에 프레임별 상태 분포 비율(%) 텍스트 오버레이
- 상태 전환이 빈번하게 발생하는 셀(깜빡임 의심 지점)을 노란색 하이라이트로 별도 표시 — 히스테리시스가 제대로 작동하는지 육안 검증용

## 8. 로깅 (logging_utils.py)

프레임마다 CSV 또는 JSON으로 기록:
- 프레임 번호
- 상태별 입자 수/비율 (STREAM/SPLASH/POOL)
- 셀별 상태 전환 발생 횟수 (누적) — 특정 셀이 비정상적으로 자주 전환되면 임계값 튜닝 필요 신호
- 평균 프레임타임 (ms)

---

## 9. 검증 기준 (Definition of Done)

이 단계가 완료됐다고 판단하는 기준:
1. 물기둥 낙하 → 바닥 충돌 → 고임 시나리오에서 STREAM→SPLASH→POOL 전이가 시각적으로 자연스럽게 관찰됨
2. 로그 상 특정 셀이 초당 여러 번 상태를 왔다갔다 하는 깜빡임 현상이 없음 (또는 있다면 원인이 되는 임계값 특정 가능)
3. 전환 경계에서 입자가 순간이동하거나 튀는 현상 없음 (블렌딩이 시각적으로 매끄러움)
4. 전체 파이프라인(격자 바인딩~렌더링)이 최소 30fps 이상으로 동작 (5,000~20,000 입자 기준, 프로토타입이므로 성능은 이 단계에서 크게 신경 쓰지 않아도 됨)
5. `config.py`의 임계값들을 바꿔가며 A/B 비교가 쉬운 구조로 되어 있음

## 10. 하지 않을 것 (Out of Scope for this phase)

- SPLASH 영역 신경망 학습/추론 (다음 단계)
- 스크린 공간 렌더링 (메타볼, 프레넬, 굴절 등) — Phase 2
- 엔진(Unity/Unreal) 통합 — Phase 3
- 성능 최적화 — 지금은 정확성/안정성 검증이 우선
