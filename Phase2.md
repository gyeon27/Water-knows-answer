# Phase 2 — 지형·SWE·2D↔3D 전환·PI-GNN

2026 제주과학고 과제연구, 고가연·엄시후 팀
작성일: 2026-08-09

---

## Phase 2 최종 상태 업데이트 (2026-08-15)

이 문서의 아래 본문은 설계가 변한 과정을 포함한 개발 기록이다. Phase 2 종료
시점의 확정 상태는 다음과 같다.

- 결정론적 height field와 Unreal용 16-bit height map 생성 완료
- finite-volume SWE, 지속 수원, 절벽 2D→3D flux 구현 완료
- `single_cliff` 12초 질량수지 오차: 유입 대비 약 0.49%
- Taichi WCSPH 기반 12개 trajectory PI-GNN 데이터셋 구성
- **NVIDIA GeForce RTX 3080에서** CUDA AMP residual PI-GNN 10,000-step 학습 수행
- 1→4→8→16→32-step rollout curriculum과 checkpoint 재개 구현
- WCSPH teacher/PI-GNN 1-step·자율 rollout 3D 비교 GUI 구현
- 최고 Phase 2 validation 속도 RMSE 기록: 약 0.767 m/s

주요 실행 명령:

```powershell
& .\.venv-gpu\Scripts\python.exe phase2\generate_pi_dataset.py
& .\.venv-gpu\Scripts\python.exe phase2\train_pi_gnn.py --steps 10000
& .\.venv-gpu\Scripts\python.exe phase2\rollout_pi_gnn.py --steps 32
& .\.venv-gpu\Scripts\python.exe phase2\compare_pi_gnn_3d.py
```

Phase 2의 자체 WCSPH 결과만으로 일반화를 주장하지 않는다. 이 한계를 검증하기
위해 공개 Water-3D 전체 데이터를 사용하는 실험을 [Phase3.md](Phase3.md)로
분리했다. Unreal 연동은 [Phase4.md](Phase4.md)로 이동했다.

---

## 1. 연구 전체 목표

**차원 가변(dimension-adaptive) 유체 시뮬레이션**을 통해, 강·폭포처럼 넓은 영역은 저렴한 저차원 솔버로 계산하고, 물리적으로 복잡한 영역(낙하·충돌·비산)만 신경망(GNN)으로 보정하는 실시간 유체 파이프라인을 만든다. 최종적으로는 Unreal Engine에서 동작 가능한 수준의 성능과 사실성을 동시에 확보하는 것이 목표다.

### 1.1 핵심 아이디어 — 차원 가변 물리 파이프라인

Phase 1에서는 모든 물을 입자로 표현하고 STREAM/SPLASH/POOL 상태를 분류했다.
Phase 2부터는 이 분류를 계산 영역 선택에 사용하되, 안정 영역은 입자가 아니라
연속적인 수심·유속장으로 계산한다.

| 상태 | 물리적 의미 | 솔버 |
|---|---|---|
| STREAM | 강·수로처럼 표면을 따라 흐르는 물 | 2D SWE 수심·유속장 |
| POOL | 정체되거나 천천히 배수되는 고인 물 | 같은 SWE 수심·유속장 내 저유속 영역 |
| FALL/SPLASH | 절벽 낙하·충돌·비산 | 3D 기본 물리 + **PI-GCN 잔차 보정** |

따라서 POOL은 고정된 별도 입자 집합이 아니다. SWE 셀의 수심과 유속에 따라
STREAM과 POOL이 자연스럽게 바뀐다. 절벽 경계에서는 질량·운동량 flux를 3D
입자로 바꾸고, 안정된 하류 물은 다시 SWE 장으로 환원한다.

### 1.2 GNN을 쓰는 방식에 대한 결정

GNN은 폭포/비산 영역 전체를 대체하지 않고, **기존 물리 솔버가 계산한 값에 대한 잔차 가속도(residual acceleration)만 예측**한다.

```text
2D 천수방정식 → 폭포 시트/기본 중력·충돌 솔버
                         ↓
              불안정·충돌 영역만 GNN 보정
                         ↓
                   3D 입자 rollout
                         ↓
               안정 영역을 2D 하류로 환원
```

이 방식을 택한 이유:

1. 중력과 단순 경계 충돌은 이미 알려진 물리이므로 신경망이 다시 학습할 필요가 없다.
2. GNN이 실패해도 기본 솔버 결과가 남아 있어 폭발적인 rollout(발산)을 억제할 수 있다.
3. 최종 Unreal 구현에서 GNN이 적용되는 영역과 계산량을 제한할 수 있어 실시간성을 확보하기 쉽다.
4. Phase 1에서 만든 `phase1/solvers/splash_solver.py`를 기준(base) 솔버로 그대로 재사용할 수 있다.

DeepMind GNS(Graph Network Simulator)와 마찬가지로 입자를 그래프 노드, 반경 이내 이웃을 엣지로 표현하되, 본 연구는 여기에 **지형 법선·경계 거리·폭포 유량**을 추가 특징으로 넣고, 전체 가속도가 아닌 **기본 물리 대비 잔차**만 학습한다는 점이 다르다.

### 1.3 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| Phase 1 | 신경망 없는 위상 기반 적응형 솔버 라우팅 검증 (STREAM/SPLASH/POOL 분류, 히스테리시스, 경계 블렌딩, SPLASH는 임시 단순 물리) | **완료** (2026-08-09, `Complete Phase 1 adaptive fluid routing`) |
| Phase 2 | 지형 조건부 강/폭포 파이프라인 + SPLASH 영역 GNN 학습 설계·구현 | **완료** |
| Phase 3 | Water-3D 공개 데이터 일반화·6조건 성능 검증 | **실행 중** |
| Phase 4 | Unreal Engine 통합과 Niagara 렌더링 | 계획 수립 |

---

## 2. Phase 2 계획

Phase 2 폴더는 두 축으로 구성된다: **(A) 지형 조건부 데이터 파이프라인**과 **(B) SPLASH 영역을 보정하는 residual GNN 학습 설계**.

### 2.1 지형 생성 (완료)

```powershell
python terrain_generation\generate_heightmaps.py
```

257×257 해상도의 결정론적(deterministic) 지형 4종을 `terrains/`에 생성한다. 각 지형은 다음을 포함:

- `height_meters.npy`: 실제 시뮬레이션용 높이 필드 (미터 단위, exact)
- `height_u16.png`: Unreal Landscape 임포트용 16비트 정규화 높이맵
- `surface_normal.npy`, `slope.npy`: 지형 특징 (법선, 경사도)
- `cliff_mask.png`, `channel_mask.png`, `source_mask.png`: 절벽/유로/수원 라우팅 마스크
- `metadata.json`: 월드 스케일, 디코딩 범위, 수원 위치, 유량
- `preview.png`: 진단용 시각화 이미지

### 2.2 Height-field 천수방정식 솔버와 차원 전환 (구현 완료, 검증 중)

`height_meters.npy`, `source_mask.png`, 수원 유량 메타데이터를 입력으로 받아
보존형 finite-volume SWE로 수심과 수평 운동량을 계산한다. 절벽으로 나가는
양의 flux는 2D 제어체적에서 제거하고 고정 질량 3D 입자로 변환한다.

현재 `single_cliff`, 12초 실행 결과:

| 항목 | 결과 |
|---|---:|
| 수원 유입 | 14.4000 m³ |
| 절벽 3D 전환 | 5.8836 m³ |
| 최종 2D 수량 | 10.3577 m³ |
| 질량수지 오차 | -0.0712 m³ (유입 대비 0.49%) |

2% 기준은 통과했지만 정상상태 수렴과 3D→2D 하류 환원은 추가 검증이 필요하다.

### 2.3 정답(teacher) 데이터 생성 전략

- **파이프라인 연결 검증**: 현재 Taichi Phase 1 솔버
- **최종 학습 정답**: 고해상도 3D SPH 또는 MPM
- **DeepMind `WaterDropSample`**: 데이터 로더·학습 코드 검증에만 사용
- **DeepMind `WaterDrop`/`WaterRamps`**: 사전학습 비교 실험용, 최종 정답으로는 사용하지 않음

> Phase 1 솔버로 학습하면 모델이 Phase 1의 근사 오차까지 모방하게 되므로, Phase 1 솔버는 최종 모델의 teacher로 사용하지 않는다.

**시나리오 무작위화 범위**

| 파라미터 | 범위 |
|---|---|
| 절벽 높이 | 3–12 m |
| 하천 폭 | 1.5–6 m |
| 유량 | 0.3–3.0 m³/s |
| 벽 경사 | 55–90° |
| 표면 거칠기 | 0–0.25 m |
| 돌출부 개수 | 0–5 (dataset v2) |
| 입자 간격 | 0.05–0.12 m |

**목표 데이터량**: 디버그 8 trajectories × 120 frames → 소규모 학습 120 trajectories × 300 frames → 최종 비교 600 train / 80 validation / 80 test trajectories (trajectory 단위로 split 분리, 프레임이 섞이지 않도록 함).

### 2.4 그래프·입력 특징 설계

- **노드**: SPLASH ROI 입자 + 1-ring 문맥 입자 (연결 반경 = 평균 입자 간격의 2.5배, 최대 이웃 48개, 매 inference step마다 그래프 재생성)
- **노드 특징(총 33)**: 최근 6프레임 속도(18) + 지형 대비 높이차(1) + 지형 법선(3) + 국소 경사도(1) + 절벽/충돌 여부(2) + 2D 시트 두께(1) + 국소 유량(1) + 상태 one-hot(3) + 중력 방향(3)
- **엣지 특징(총 8)**: 상대 위치/연결 반경(3) + 상대 거리(1) + 상대 속도(3) + 지형 높이차(1)
- 절대 위치는 사용하지 않는다 (다른 크기·위치의 지형에 일반화하기 위함).

### 2.5 예측 대상과 모델

```text
target_delta_v = v_gt(t+1) - v_base(t+1)
v_pred(t+1) = v_base(t+1) + GNN(graph_t)
x_pred(t+1) = x(t) + dt * v_pred(t+1)
```

기준 모델: Encode–Process–Decode GNS (은닉 128, message-passing 6 블록, SiLU, LayerNorm), 이후 3/6/10 블록 ablation.

### 2.6 손실 함수

```text
L = 1.0 L_delta_v + 0.25 L_rollout + 0.10 L_penetration + 0.05 L_momentum + 0.05 L_density
```

질량은 입자 생성·삭제 없는 구조로 우선 보존, 2D↔3D 전환 유입 입자 수는 별도 flux accounting으로 검증한다.

### 2.7 학습 절차와 현재 검증 결과

1. 자체 디버그 8×120 trajectory로 데이터 로딩/그래프 생성 확인 (**완료**)
2. 고정 그래프 1-step overfit (**완료**: 0.05946 → 1.84×10⁻⁶)
3. 고해상도 SPH/MPM teacher 생성 (**다음 작업**)
4. 120개 이상 trajectory 1-step 학습
5. 위치에 random-walk noise 추가 (누적 오차 강건성)
6. 8 → 16 → 32 step curriculum rollout 학습
7. validation rollout 정체 시 조기 종료 → OOD 지형에서 test

옵티마이저: AdamW, lr 1e-4, weight decay 1e-6, grad clip 1.0, 동적 배치(노드 8k–16k).

### 2.8 데이터 split (지형 구조 기준)

| Split | 지형 |
|---|---|
| train | single_cliff, sloped_cliff (다양한 매개변수) |
| validation | 같은 구조, 보지 않은 높이·유량·폭 |
| test-ID | rocky_cliff의 보지 않은 seed |
| test-OOD | split_channel, 다단 절벽, mesh 돌출부 |

### 2.9 비교 실험 (4개 모델)

1. 기본 중력+충돌 솔버
2. 현재 Phase 1 SPLASH 솔버
3. 전체 가속도를 예측하는 GNS (baseline)
4. **제안 방식**: 국소 residual GNS

평가 지표: 1/10/50/100-step 위치 RMSE, 지형 침투율, 질량·운동량 오차, SPLASH ROI 입자 비율, 프레임당 inference 시간, teacher 대비 속도 향상, 영상 블라인드 평가.

### 2.10 구현 순서와 통과 기준

| 단계 | 내용 | 상태 |
|---|---|---|
| P2-1a | height map 생성 | **완료** |
| P2-1b | height-field 천수방정식 | **기준 솔버 구현 완료, 장기 안정성 검증 중** |
| P2-1c | 절벽 시작/종료 및 2D→3D flux 추출 | **flux 및 고정질량 입자 변환 구현 완료** |
| P2-2a | teacher trajectory 저장기 | **디버그 teacher 구현 완료, SPH/MPM 교체 예정** |
| P2-2b | graph dataset loader | **구현 완료** |
| P2-2c | residual GNS 1-step 학습 | **디버그 overfit 검증 완료, 고품질 teacher 대기** |
| P2-2d | 1→4→8→16→32-step rollout 학습 및 비교 GUI | **완료** |
| P2-3 | Unreal inference/렌더링 연결 | Phase 4로 이동 |

**GNN 구현 착수 조건** (현재 충족):

- 천수방정식에서 수원 → 하천 → 절벽까지 유량이 전달될 것
- 절벽을 통과한 누적 flux 오차가 2% 이하일 것
- 동일 seed의 지형 생성 결과가 byte-level로 재현될 것
- 디버그 teacher trajectory 8개가 생성될 것

### 2.11 현재 결론과 남은 한계

- SWE→절벽 flux→3D 입자→그래프→잔차 GNS의 전체 데이터 경로가 연결됐다.
- 디버그 overfit 성공은 학습 가능성을 확인한 것이며 사실성 향상을 입증한 결과는 아니다.
- 현재 ballistic debug teacher를 최종 학습에 사용하면 기준 솔버 오차를 모방하므로
  고해상도 SPH/MPM teacher로 교체해야 한다.
- 최종 연구 주장은 teacher 대비 정확도뿐 아니라 전체 3D 방식 대비 속도 향상,
  질량보존, 지형 침투율, OOD 지형 일반화로 검증한다.

---

## 3. 참고 문헌

- DeepMind, *Learning to Simulate Complex Physics with Graph Networks* (ICML 2020)
- 공식 `google-deepmind/deepmind-research/learning_to_simulate` 구현과 데이터 형식

---

## 4. 발표 자료 구성 원칙

### 4.1 연구 방법 슬라이드의 핵심 문장

> 넓고 안정적인 물은 2D SWE로 계산하고, 절벽에서 추출한 유량만 3D 입자로
> 전환하며, PI-GCN은 충돌·분열 영역의 잔차 속도만 보정한다.

현재 구현과 맞지 않는 표현:

- “밀도장과 마스크를 신경망이 출력한다” → 밀도장·마스크는 물리 파이프라인 출력
- “신경망이 고정밀 시뮬레이션 전체를 근사한다” → 국소 SPLASH 잔차만 근사
- “하나의 신경망을 구성한다” → 같은 가중치의 GNS를 매 프레임 동적 국소 그래프에 적용

권장 시각 구성:

```text
[Height field + 수원]
          ↓
[2D SWE: 수심 h, 유속 u] ── 절벽 flux ──> [3D 기본 물리]
                                             ↓
                                  [SPLASH ROI 그래프]
                                             ↓  PI-GCN: Δv
                                  [보정된 3D rollout]
                                             ↓
                                      [하류 SWE 환원]
```

한 슬라이드에는 위 흐름도와 다음 세 개의 짧은 보조 문장만 둔다.

- **물리 기반:** 중력·SWE·비관통은 기본 솔버가 계산
- **신경망 역할:** 충돌·분열에서 발생하는 오차만 Δv로 보정
- **경량화:** 전체 영역이 아니라 SPLASH ROI에만 GNS 적용

### 4.2 연구 결과 슬라이드에 넣을 현재 결과

현재 결과는 “사실적인 폭포 완성”이 아니라 “전체 파이프라인 연결과 보존성 검증”으로 표현한다.

| 결과 | 제시할 수치/그림 | 의미 |
|---|---|---|
| 지형 조건부 입력 | 257×257 지형 4종 비교 이미지 | 서로 다른 지형에서 같은 파이프라인 사용 |
| SWE 질량수지 | 12초 유입 14.4 m³, 오차 0.49% | 차원 전환 중 질량 보존 가능성 |
| 2D→3D 전환 | 절벽 flux와 3D 입자 위치 시각화 | 폭포 구간만 선택적으로 3D화 |
| 데이터 파이프라인 | 8 trajectories × 120 frames | 학습 입력 자동 생성 |
| 그래프 구성 | 노드 33차원, 엣지 8차원 예시 | 지형·유량·운동 정보를 GNS에 전달 |
| 1-step overfit | loss 0.05946 → 1.84×10⁻⁶ | 로더·모델·역전파 연결 확인 |
| 소프트웨어 검증 | 단위 테스트 6개 통과 | 수치·스키마·모델 순전파 검증 |

추천 결과 슬라이드 순서:

1. **2D 계산이 절벽에서만 3D로 바뀐다** — 수심장/flux/입자 3단 비교
2. **차원 전환 후에도 질량 오차는 0.49%였다** — 누적 유입·3D 전환·잔량 그래프
3. **지형이 달라도 같은 데이터 구조를 사용한다** — 4개 지형 스몰멀티플
4. **잔차 GNS 학습 파이프라인이 연결됐다** — loss 로그축 그래프와 그래프 샘플
5. **아직 사실성 향상은 검증 전이다** — SPH/MPM teacher와 rollout 평가를 다음 과제로 명시

### 4.3 최종 연구 결과에서 추가해야 할 지표

- 전체 3D SPH/MPM 대비 프레임 시간과 속도 향상 배수
- 1/10/50/100-step 위치 RMSE
- 지형 침투 입자 비율과 최대 침투 깊이
- 2D↔3D 누적 질량·운동량 오차
- SPLASH ROI가 전체 물 영역에서 차지하는 비율
- single/sloped 학습 후 rocky/split-channel OOD 성능
- 기본 물리, 전체 GNS, 제안 residual PI-GCN의 ablation
- 동일 카메라·동일 유량 영상에 대한 블라인드 사실성 평가
