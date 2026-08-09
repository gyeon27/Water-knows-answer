# 지형 조건부 폭포 GNN 학습 계획

## 1. 결정 사항과 현재 범위

GNN은 폭포 전체를 대체하지 않고 **기존 물리 솔버의 잔차 속도(residual Δv)**만 예측한다.
SWE의 수심·밀도장과 2D↔3D 전환 마스크는 신경망 출력이 아니라 물리 파이프라인이 계산한다.

```text
2D 천수방정식 → 폭포 시트/기본 중력·충돌 솔버
                         ↓
              불안정·충돌 영역만 GNN 보정
                         ↓
                   3D 입자 rollout
                         ↓
               안정 영역을 2D 하류로 환원
```

이 설계를 택한 이유는 다음과 같다.

1. 중력과 단순 경계 충돌은 이미 알고 있는 물리이므로 신경망이 다시 학습할 필요가 없다.
2. GNN이 실패해도 기본 솔버 결과가 남아 있어 폭발적인 rollout을 억제할 수 있다.
3. 최종 Unreal 구현에서 GNN 적용 영역과 계산량을 제한할 수 있다.
4. `fluid_routing/solvers/splash_solver.py`를 기준 솔버로 재사용할 수 있다.

DeepMind GNS와 마찬가지로 입자를 그래프 노드로, 반경 이내 이웃을 엣지로 표현한다. 다만 본 연구는 지형 법선·경계 거리·폭포 유량을 추가하고 전체 가속도가 아닌 기본 물리 대비 잔차를 학습한다.

## 2. height map 사용 방식

`terrain_generation/generate_heightmaps.py`가 생성한 데이터를 다음과 같이 사용한다.

| 파일 | 2D 솔버 | 3D/GNN |
|---|---:|---:|
| `height_meters.npy` | 바닥 높이와 수면 높이 계산 | 입자 아래 지형 높이 표본화 |
| `surface_normal.npy` | 경사 방향과 유속 계산 | 노드의 경계 법선 특징 |
| `slope.npy` | 절벽 후보 검출 | 충돌 난이도 특징 |
| `cliff_mask.png` | 2D→3D 전환 후보 | SPLASH ROI 중심 |
| `channel_mask.png` | 유체 계산 영역 제한 | 폭포 폭 조건 |
| `source_mask.png` | 유량 주입 | 전역 유량 조건 |
| `metadata.json` | 격자 간격과 경계조건 | 정규화 및 시나리오 조건 |

height map으로 표현할 수 없는 돌출 암반은 Phase 2의 두 번째 데이터 버전에서 triangle mesh와 SDF로 추가한다. 첫 데이터 버전은 height-field 지형만 사용한다.

## 3. 정답 데이터 생성

### 3.1 Teacher

- 파이프라인 연결 시험: SWE flux + ballistic 3D debug teacher (**8×120 생성 완료**)
- 최종 학습 정답: 고해상도 3D SPH 또는 MPM
- DeepMind `WaterDropSample`: 데이터 로더와 학습 코드 검증에만 사용
- DeepMind `WaterDrop`/`WaterRamps`: 사전학습 비교 실험에 사용 가능하지만 최종 정답을 대신하지 않음

현재 debug teacher로 학습하면 기준 물리의 근사 오차까지 모방하므로 최종 모델의
teacher로 사용하지 않는다. 데이터 로더·그래프·역전파 검증에만 사용한다.

### 3.2 시나리오 표본

각 trajectory에서 다음 조건을 무작위화한다.

```text
절벽 높이       3–12 m
하천 폭         1.5–6 m
유량            0.3–3.0 m³/s
벽 경사         55–90°
표면 거칠기     0–0.25 m
돌출부 개수     0–5 (dataset v2)
입자 간격       0.05–0.12 m
```

최초 목표량:

- 디버그: 8 trajectories × 120 frames
- 소규모 학습: 120 trajectories × 300 frames
- 최종 비교: 600 train / 80 validation / 80 test trajectories

trajectory 단위로 분리하며 하나의 trajectory 프레임이 여러 split에 섞이지 않게 한다.

### 3.3 저장 스키마

trajectory 하나를 압축 NPZ 또는 Zarr group으로 저장한다.

```text
positions       float32 [T, N, 3]      # world metres
velocities      float32 [T, N, 3]
particle_id     int32   [N]
particle_type   uint8   [N]
active_mask     bool    [T, N]
splash_roi      bool    [T, N]
terrain_id      string
flow_rate       float32 [T, 1]
dt              float32 scalar
```

지형 배열은 trajectory마다 복제하지 않고 `terrain_id`로 `phase2/terrains/`를 참조한다.

## 4. 그래프와 입력 특징

### 4.1 그래프

- 노드: SPLASH ROI 입자 + 반경 1-ring의 문맥 입자
- 엣지: 반경 검색 그래프
- 초기 연결 반경: 평균 입자 간격의 2.5배
- 최대 이웃: 48개
- 그래프 재생성: 매 inference step
- 지형: 별도 고정 노드 대신 각 입자 위치에서 height/normal/slope를 표본화

초기 모델은 고정 지형 노드를 두지 않는다. height-field 표본 특징이 더 가볍고 Unreal 이식이 쉽기 때문이다. 돌출 mesh/SDF 버전에서만 경계 노드를 비교 실험한다.

### 4.2 노드 특징

```text
최근 6프레임 속도                  18
현재 높이 - 지형 높이              1
지형 법선                           3
국소 경사도                         1
절벽 마스크/충돌 여부               2
2D 시트 두께                         1
국소 유량                            1
STREAM/SPLASH/POOL one-hot           3
중력 방향                            3
```

절대 위치는 넣지 않는다. 경계까지의 거리와 상대 위치를 사용해 다른 크기와 위치의 지형에 일반화한다.

### 4.3 엣지 특징

```text
상대 위치 / 연결 반경               3
상대 거리                            1
상대 속도                            3
송신·수신 입자의 지형 높이 차       1
```

## 5. 예측값과 적분

기본 솔버가 계산한 다음 속도를 `v_base(t+1)`, teacher 속도를 `v_gt(t+1)`라 할 때 학습 target은 다음과 같다.

```text
target_delta_v = v_gt(t+1) - v_base(t+1)
v_pred(t+1) = v_base(t+1) + GNN(graph_t)
x_pred(t+1) = x(t) + dt * v_pred(t+1)
```

출력은 데이터셋의 train split 평균과 표준편차로 정규화한다. 중력은 기본 솔버가 처리하므로 target에서 자연스럽게 제거된다.

## 6. 모델 구조

최종 기준 모델은 Encode–Process–Decode GNS로 고정한다. 현재 디버그 모델은
연결 검증을 위해 hidden 64, message-passing 3블록을 사용한다.

```text
node encoder     MLP → 128
edge encoder     MLP → 128
processor        6 message-passing blocks
hidden width     128
activation       SiLU
normalization    LayerNorm
decoder          MLP 128→128→3
output           residual Δv (x, y, z)
```

초기에는 attention, transformer, recurrent state를 넣지 않는다. 기준 모델의 장기 rollout과 속도를 먼저 측정한 뒤 message-passing block 수 3/6/10만 ablation한다.

## 7. 손실 함수

```text
L = 1.0 L_delta_v
  + 0.25 L_rollout
  + 0.10 L_penetration
  + 0.05 L_momentum
  + 0.05 L_density
```

- `L_delta_v`: 한 스텝 residual 속도 MSE
- `L_rollout`: 8→16→32 step curriculum rollout 위치 오차
- `L_penetration`: 입자가 height field 아래로 들어간 거리 페널티
- `L_momentum`: SPLASH ROI의 전체 운동량 변화 오차
- `L_density`: 반경 내 이웃 수 또는 SPH 밀도 편차

질량은 입자를 생성·삭제하지 않는 구조로 우선 보존한다. 이후 2D↔3D 전환에서 생성되는 입자 수는 별도의 flux accounting으로 검증한다.

## 8. 학습 절차

1. 자체 디버그 데이터 로딩·그래프 생성 확인 (**완료**)
2. 고정 그래프 1-step overfit (**완료**, loss 0.05946 → 1.84×10⁻⁶)
3. 고해상도 SPH/MPM teacher 생성 (**다음 작업**)
4. 120개 이상 자체 trajectory로 1-step 학습
5. 입력 위치에 random-walk noise를 추가해 누적 오차에 강하게 학습
6. 8-step rollout loss 활성화
7. 16-step, 이후 32-step으로 curriculum 확장
8. validation rollout이 개선되지 않으면 조기 종료
9. 보지 않은 `split_channel`과 돌출 지형에서 test
10. TorchScript/ONNX 등 배포 형식은 모델이 확정된 뒤 선택

초기 optimizer 설정:

```text
AdamW
learning rate       1e-4
weight decay        1e-6
gradient clipping   1.0
batch               그래프 총 노드 8k–16k가 되도록 동적 배치
```

## 9. 데이터 split

단순 random split 대신 지형 구조를 기준으로 분리한다.

```text
train       single_cliff, sloped_cliff의 다양한 매개변수
validation  같은 구조이지만 보지 않은 높이·유량·폭
test-ID     rocky_cliff의 보지 않은 seed
test-OOD    split_channel, 다단 절벽, mesh 돌출부
```

이 구성이 새로운 오픈월드 지형에 대한 일반화 주장을 검증한다.

## 10. 비교 실험

최소한 다음 네 모델을 같은 trajectory에서 비교한다.

1. 기본 중력+충돌 솔버
2. 현재 Phase 1 SPLASH 솔버
3. 전체 가속도를 예측하는 GNS
4. 제안 방식: 국소 residual GNS

평가 지표:

- 1/10/50/100-step 위치 RMSE
- 지형 침투율
- 질량·운동량 오차
- SPLASH ROI 입자 비율
- 한 프레임 inference 시간
- teacher 대비 속도 향상
- 영상 블라인드 평가에서의 사실성

## 11. 구현 순서와 통과 기준

```text
P2-1a  height map 생성                         완료
P2-1b  height-field 천수방정식                 구현 완료, 장기 검증 중
P2-1c  절벽 시작/종료 및 2D→3D flux 추출       완료
P2-2a  teacher trajectory 저장기               디버그 완료, SPH/MPM 대기
P2-2b  graph dataset loader                    완료
P2-2c  residual GNS 1-step 학습                디버그 overfit 완료
P2-2d  rollout 학습 및 평가
P2-3   Unreal inference/렌더링 연결
```

GNN 구현 착수 조건(현재 충족):

- 천수방정식에서 수원→하천→절벽까지 유량이 전달될 것
- 절벽을 통과한 누적 flux 오차가 2% 이하일 것
- 동일 seed의 지형 생성 결과가 byte-level로 재현될 것
- 디버그 teacher trajectory 8개가 생성될 것

## 참고

- DeepMind, *Learning to Simulate Complex Physics with Graph Networks* (ICML 2020)
- 공식 `google-deepmind/deepmind-research/learning_to_simulate` 구현과 데이터 형식
