# Phase 4 계획 — Unreal Engine 실시간 통합 및 렌더링

작성일: 2026-08-15
선행 조건: Phase 3의 사전 정의 실험·보고서 생성 완료

## 1. 목표와 범위

Phase 4의 목표는 Phase 3에서 검증한 차원 가변 유체 파이프라인을 Unreal
Engine 안에서 실행하고, 계산 결과를 Niagara로 시각화하여 실제 게임 프레임에서
동작하는지를 확인하는 것이다.

```text
Unreal Landscape / Height Field
              ↓
       2D SWE 수심·유속장
              ↓ 절벽 flux
       3D 기본 물리 입자
              ↓ SPLASH ROI
       PI-GNN 잔차 추론
              ↓
   Niagara 폭포·비말·거품 렌더링
```

Phase 4에서는 모델을 다시 학습하거나 Phase 3의 판정 기준을 바꾸지 않는다.
Unreal 연동, 좌표·단위 변환, 추론, 렌더링 및 엔진 내 성능 측정만 수행한다.

## 2. 시작 전 고정할 사항

1. Unreal Engine의 정확한 버전을 고정하고 결과 문서에 기록한다.
2. NNE는 현재 Beta 기능이므로 사용할 runtime과 CPU/GPU 실행 경로를 함께 기록한다.
3. Phase 3에서 선택된 validation-best 체크포인트의 SHA-256을 고정한다.
4. 비교 장면, 카메라, 해상도, 품질 설정, 목표 FPS를 실행 전에 고정한다.
5. Phase 3에서 제안 방식의 우위가 입증되지 않으면 Unreal 데모는 만들되
   성능 우위를 연구 결론으로 주장하지 않는다.

## 3. Phase 3 → Phase 4 데이터 계약

### 3.1 모델 입력과 출력

| 텐서 | dtype | 논리적 크기 | 의미 |
|---|---|---:|---|
| `node_features` | float32 | N×27 | 속도 이력, 경계 거리, 상태, 중력 방향 |
| `particle_type` | int64 | N | 유체/고정 경계 입자 종류 |
| `edge_features` | float32 | E×4 | 정규화 상대 위치와 거리 |
| `edge_index` | int64 | 2×E | sender/receiver 인덱스 |
| `delta_acceleration` | float32 | N×3 | 기본 물리 대비 잔차 가속도 |

동적 크기 N/E가 NNE runtime에서 지원되지 않으면 최대 노드·간선 크기로
padding하고 `node_mask`, `edge_mask`를 추가한다. 기본 상한은 Phase 3과 같은
N=8,000, E=120,000이며, 메모리 측정 후에만 변경한다.

### 3.2 좌표와 단위

학습 데이터의 Y-up 미터 좌표를 Unreal의 Z-up 센티미터 좌표로 변환한다.

```text
UE_Position_cm = (x, z, y) × 100
UE_Velocity_cm_s = (vx, vz, vy) × 100
```

축 부호는 첫 기준 장면에서 검증하고 `coordinate_contract.json`에 기록한다.
모델 입력은 학습 당시의 좌표·정규화 단위를 유지하며, Niagara로 넘기기 직전에만
Unreal 좌표로 변환한다.

### 3.3 Niagara 전달 속성

| 속성 | 용도 |
|---|---|
| Particle ID | 프레임 간 동일 입자 추적 |
| Position / Velocity | 이동과 모션 벡터 |
| STREAM/SPLASH/POOL | 렌더러와 재질 선택 |
| Density proxy | 입자 크기·불투명도 조절 |
| Impact energy | 비말·거품·안개 생성량 |
| Lifetime / Active mask | flux 기반 생성·하류 환원 |

## 4. 구현 순서

### P4-0. 모델 배포 가능성 검증

1. Phase 3 best checkpoint를 evaluation 모드로 고정한다.
2. 같은 입력으로 PyTorch 출력 기준값을 저장한다.
3. ONNX로 export하고 ONNX Runtime 출력과 비교한다.
4. Unreal NNE asset으로 import한 뒤 같은 고정 입력을 실행한다.
5. 최대 절대 오차 1e-4 이하인지 확인한다.

GNN의 이웃 합산(`edge_index`, scatter/index-add)이 선택한 NNE runtime에서
지원되지 않으면 모델을 임의로 단순화하지 않는다. 다음 순서로 대안을 적용한다.

1. padding된 고정 크기 그래프로 다시 export
2. 그래프 구성·메시지 합산을 Unreal RDG compute shader에서 수행하고 MLP만 NNE로 실행
3. 그래도 불가능하면 ONNX Runtime 기반 별도 Unreal plugin을 실험 경로로 사용

### P4-1. 오프라인 Replay 데모

Phase 3의 대표 NPZ를 Unreal에서 읽어 Niagara에 표시한다. 이 단계에는 실시간
물리나 신경망 추론을 넣지 않는다. 좌표, 단위, 입자 ID, 카메라, 재질 및 데이터
전달 경로만 검증한다.

완료 기준:

- Python 3D 비교 GUI와 Unreal에서 같은 프레임의 입자 위치가 일치
- 입자 ID가 프레임 사이에서 섞이지 않음
- quiet/complex/violent 대표 장면을 모두 재생 가능

### P4-2. Landscape와 2D SWE 연결

1. `height_u16.png`와 높이 범위 metadata로 Landscape를 생성한다.
2. 수원 위치·유량을 Data Asset으로 만든다.
3. 2D SWE를 우선 CPU C++로 이식하고 결과를 Grid2D 또는 texture로 전달한다.
4. 절벽 flux의 질량·운동량을 3D 입자 생성량으로 변환한다.
5. CPU가 병목일 때만 RDG compute shader로 이동한다.

Python 기준 결과와 10초 동안 비교하여 수심 RMSE, 누적 유량 오차, 질량수지
오차를 기록한다. 질량수지 오차 목표는 2% 이하이다.

### P4-3. 온라인 PI-GNN 추론

매 프레임 다음 순서로 실행한다.

```text
활성 3D 입자 갱신
→ 공간 해시/Neighbor Grid로 반경 그래프 생성
→ 현재 프레임 정보만으로 SPLASH ROI 판정
→ ROI 그래프 텐서 구성
→ NNE PI-GNN 추론
→ 기본 가속도 + 잔차 적용
→ 지형 투영·inward normal velocity 제거
→ 하류 안정 입자를 SWE flux로 환원
```

그래프와 추론은 전체 입자가 아니라 SPLASH ROI에만 적용한다. Phase 3과 같은
정규화 통계를 모델 asset 옆 JSON/Data Asset에 저장하고 런타임에서 변경하지 않는다.

### P4-4. Niagara 렌더링

계산 표현과 렌더링 표현을 분리한다.

| 영역 | 권장 표현 |
|---|---|
| 강·고인 물 | Landscape 위 수면 mesh 또는 height-field material |
| 폭포 본체 | ribbon/얇은 sheet mesh |
| 큰 비말 | Niagara GPU particle |
| 작은 물방울 | 저비용 sprite |
| 거품 | 충돌 에너지 기반 decal/particle |
| 물안개 | 강한 충돌 지점에서만 제한 생성 |

모든 물을 입자 surface reconstruction으로 렌더링하지 않는다. 연구의 실시간성은
SWE 수면과 폭포 sheet를 저렴하게 유지하고 SPLASH만 입자로 보이는 구조에서 나온다.

### P4-5. 엔진 내 성능·정확도 검증

Phase 3과 동일하게 2k/5k/10k/20k/50k 입자와 SPLASH 비율
5/25/50/100%를 측정한다. Unreal Insights와 GPU profiler를 사용해 다음 시간을
분리한다.

- SWE
- 그래프 구성
- 라우팅
- NNE 추론
- 블렌딩·충돌
- Niagara simulation
- rendering
- 전체 game/GPU frame

워밍업 10프레임 후 300프레임의 평균, 표준편차, p95, 최대값을 기록하고
30/60/120/144 FPS 예산 달성 여부를 보고한다.

## 5. Unreal 프로젝트 구조

```text
unreal/WaterKnowsAnswer/
├── Config/
├── Content/
│   ├── Data/                 # 지형·정규화 Data Asset
│   ├── Models/               # NNE model asset
│   ├── Niagara/              # sheet/splash/foam/mist systems
│   ├── Materials/
│   └── Maps/                 # 고정 benchmark scenes
├── Plugins/WaterRuntime/
│   ├── Source/WaterRuntime/  # SWE, routing, graph, NNE bridge
│   └── WaterRuntime.uplugin
└── Scripts/                  # ONNX export 및 asset 검증
```

`Content/`의 대용량 생성 asset과 DerivedDataCache는 Git에서 제외한다. 소스,
설정, 작은 model metadata, 자동화 스크립트와 결과 요약은 Git에 포함한다.

## 6. 테스트

- PyTorch/ONNX/NNE 고정 입력 출력 parity
- 미터↔센티미터 및 Y-up↔Z-up 왕복 변환
- Python/Unreal height-field 표본 높이·법선 일치
- 동일 위치에서 생성한 반경 그래프의 이웃 수와 edge hash 일치
- 미래 프레임이나 test 통계가 라우터에 들어가지 않음
- 2D→3D→2D 누적 질량수지 오차 2% 이하
- 지형 침투율 1% 미만
- 입자 ID와 flux 생성·소멸 수 보존
- 300프레임 벤치마크 raw row 수 일치
- 오프라인 replay와 온라인 inference 표시를 UI에서 구분

## 7. 완료 기준

1. 임의의 지원 height map에서 수원→강→절벽→폭포가 생성된다.
2. Phase 3 대표 장면 세 그룹을 Unreal에서 재현한다.
3. 제안 방식과 GNN-only의 엔진 내 프레임타임을 동일 장면에서 비교한다.
4. Python과 Unreal의 고정 rollout 오차가 사전 기준 안에 든다.
5. 전체 렌더링 포함 30/60/120/144 FPS 달성 범위를 표로 제시한다.
6. 결과가 목표에 미달해도 설정을 사후 변경하지 않고 적용 가능 범위와 실패 사례를 보고한다.

## 8. Phase 3 실행 중 미리 할 수 있는 작업

- Unreal 버전과 target platform 고정
- 빈 C++ 프로젝트 및 `WaterRuntime` plugin scaffold 생성
- Phase 3 대표 NPZ를 읽는 offline replay importer 작성
- 좌표 변환 단위 테스트 작성
- `coordinate_contract.json` 및 model manifest schema 확정
- Niagara STREAM/SPLASH/POOL prototype 제작
- PyTorch→ONNX export smoke test 작성

실제 best checkpoint를 사용하는 ONNX/NNE parity와 최종 성능 측정은 Phase 3
학습 완료 후 수행한다.

### 생성된 ONNX 전달 패키지

- `phase4/onnx/ours_full_graph.onnx`: 전체 그래프 입력, SPLASH 출력만 채택
- `phase4/onnx/ours_roi_splash.onnx`: 라우팅 후 SPLASH ROI 그래프만 입력

두 버전 모두 STREAM과 POOL은 Unreal 측 천해방정식으로 계산한다. ROI 버전은
SPLASH가 없으면 NNE 호출을 생략하고, `roi_to_global` 배열로 출력 가속도를 전체
입자 배열에 다시 배치한다. 두 파일은 동일한 학습 가중치를 사용하므로 ONNX
그래프와 SHA-256은 같으며, 성능 차이는 JSON manifest에 정의된 입력 그래프 구성
방식에서 발생한다.

ONNX checker, 동적 `N/E` 실행 및 PyTorch parity를 통과했다. CPU ONNX Runtime
기준 최대 절대오차는 `4.1e-10`이다. Unreal에서는 선택한 NNE runtime이
`ScatterElements(reduction=add)`와 동적 차원을 지원하는지 import 후 확인한다.

## 9. 기술 선택 근거

Unreal NNE는 runtime별 모델 import와 CPU/GPU 추론 인터페이스를 제공하지만 Beta
기능이므로 버전과 runtime을 고정해야 한다. GPU 경로는 render-thread 호출 및 RDG
buffer 입출력을 요구할 수 있다. Niagara는 외부 데이터용 Data Interface와 사용자
정의 속성을 제공하므로, 시뮬레이션 결과를 렌더링 표현과 분리해 전달하는 구조가
적합하다.

공식 문서:

- https://dev.epicgames.com/documentation/unreal-engine/neural-network-engine-overview-in-unreal-engine
- https://dev.epicgames.com/documentation/unreal-engine/overview-of-niagara-effects-for-unreal-engine
- https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/Niagara
