# Water Knows Answer

2026 제주과학고 과제연구 — 고가연·엄시후

넓고 안정적인 물은 2D 천해방정식(SWE)으로 계산하고, 절벽에서 3D 입자로
전환된 물 중 충돌·분열이 복잡한 영역만 PI-GNN으로 보정하는 차원 가변 폭포
시뮬레이션 연구이다. 최종 목표는 임의 지형에 강과 폭포를 생성하고 Unreal
Engine에서 실시간으로 실행하는 것이다.

## 현재 상태

- Phase 1: STREAM/SPLASH/POOL 라우팅과 기본 입자 물리 — 완료
- Phase 2: 지형, SWE, 2D↔3D flux, WCSPH, PI-GNN — 완료
- Phase 3: Water-3D 공개 데이터 학습, 7조건 검증과 ROI 최적화 — 완료
- Phase 4: ONNX 2종 export·parity 검증 — 완료, Unreal 통합 예정

## 계산 구조

```text
Height Map
  → 2D SWE (STREAM·POOL)
  → 절벽 유량을 3D 입자로 변환
  → STREAM / SPLASH / POOL 라우팅
  → SPLASH ROI만 PI-GNN 추론
  → SWE–GNN 경계 블렌딩
  → Unreal Niagara 렌더링
```

PI-GNN은 모든 물을 대신 계산하지 않는다. 넓고 안정적인 흐름은 저렴한 SWE가
담당하고, 충돌·분열·비말처럼 3차원 상호작용이 필요한 영역만 그래프 신경망이
보정한다. 최적화 버전은 라우팅 후 SPLASH ROI에 대해서만 그래프를 생성한다.

## 저장소 구조

```text
Water-knows-answer/
├── README.md                  # 빠른 시작과 저장소 안내
├── 전체.md                    # 전체 연구 내용과 변경 이력
├── Phase1.md                  # 라우팅 프로토타입 설명
├── Phase2.md                  # 지형·SWE·WCSPH·PI-GNN 설명
├── Phase3.md                  # 공개 데이터 학습·평가·성능 결과
├── Phase4.md                  # Unreal 통합 규격과 완료 기준
├── phase1/                    # STREAM/SPLASH/POOL 라우팅 데모
│   ├── main.py
│   ├── routing.py
│   └── solvers/               # stream, splash, pool 기초 솔버
├── phase2/
│   ├── terrains/              # 다섯 종류 height map과 라우팅 마스크
│   ├── shallow_water/         # 2D 천해방정식과 2D→3D emitter
│   ├── teacher/               # WCSPH teacher와 trajectory writer
│   ├── gnn/                   # Phase 2 PI-GNN 모델·손실·배칭
│   ├── train_pi_gnn.py        # PI-GNN 학습
│   ├── compare_pi_gnn_3d.py   # teacher/GNN 3D 비교 GUI
│   └── archive/               # 초기 debug·구형 뷰어
├── phase3/
│   ├── data.py                # Water-3D 인덱싱과 통합 그래프 구성
│   ├── models.py              # 공통 GNS/PI-GNN 구조
│   ├── training.py            # CUDA AMP 학습과 재개
│   ├── evaluation.py          # 1/8/16/32/100-step rollout
│   ├── benchmark.py           # 프레임 단계별 성능 측정
│   ├── reporting.py           # CSV·그래프·보고서 생성
│   ├── swe_baseline.py        # 실제 SWE-only 기준선
│   ├── roi_validation.py      # SPLASH ROI 정확도 검증
│   ├── continuous_terrain_runtime.py # 무한 수원 게임형 런타임
│   ├── external_teacher/      # Palouse DEM·외부 DFSPH 검증 파이프라인
│   ├── archive/               # 선택 실험·구형 유한 rollout
│   ├── tests/                 # 회귀·물리·그래프 단위 테스트
│   └── results_summary/       # Git에 보존하는 결과표와 그래프
└── phase4/
    ├── export_onnx.py         # best checkpoint ONNX 변환·검증
    └── onnx/                  # Unreal NNE 전달 모델과 manifest
```

## 문서

- [전체 연구 설명과 변경 이력](전체.md)
- [Phase 1 — 상태 라우팅](Phase1.md)
- [Phase 2 — 지형·SWE·PI-GNN](Phase2.md)
- [Phase 3 — 공개 데이터 성능 검증](Phase3.md)
- [Phase 4 — Unreal 통합](Phase4.md)

## 주요 실행 진입점

```powershell
# Phase 1 라우팅 데모
& C:\venvs\fluid_routing\Scripts\python.exe phase1\main.py

# Phase 2 3D teacher/PI-GNN 비교
& .\.venv-gpu\Scripts\python.exe phase2\compare_pi_gnn_3d.py

# Phase 3 전체 파이프라인: 준비→학습→평가→벤치마크→보고서
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 all --data-root auto

# Phase 3 단위 테스트
& .\.venv-gpu\Scripts\python.exe -m unittest discover -s phase3\tests -v

# 최적화 Ours의 32-step teacher/ROI 3D GUI (학습 curriculum 범위)
& .\.venv-gpu\Scripts\python.exe -m phase3.view_optimized_ours `
    --data-root E:\WaterKnowsAnswer_Phase3 --group violent --steps 32

# quiet/complex/violent 중 장면을 고를 수 있으며 최초 실행은 Water-3D와
# ours/best.pt로 검증 조건 G와 동일한 rollout 캐시를 만든 뒤 GUI를 연다.
# 100-step은 학습 범위 밖의 장기 안정성 stress test이므로 --steps 100으로 별도 실행한다.

# 게임 런타임형 무한 수원: Height Map → SWE → 3D SPLASH ROI PI-GNN → SWE 복귀/입자 재사용
& .\.venv-gpu\Scripts\python.exe -m phase3.continuous_terrain_runtime `
    --data-root E:\WaterKnowsAnswer_Phase3 `
    --terrain phase2\terrains\natural_waterfall

# 위 연속 지형 런타임과 동일한 정규화/단위의 Unreal NNE용 ROI ONNX
& .\.venv-gpu\Scripts\python.exe -m phase4.export_terrain_runtime_onnx `
    --data-root E:\WaterKnowsAnswer_Phase3 --output phase4\onnx

# Unreal용 전체 그래프/ROI ONNX 재생성 및 parity 검사
& .\.venv-gpu\Scripts\python.exe -m phase4.export_onnx `
    --data-root E:\WaterKnowsAnswer_Phase3 --output phase4\onnx
```

대용량 Water-3D 원본, 체크포인트와 rollout은 Samsung T7의
`WaterKnowsAnswer_Phase3`에 저장하며, 코드와 요약 결과는 이 저장소에 둔다.

## Phase 3 비교 조건

|조건|계산 방식|목적|
|---|---|---|
|SWE-only|모든 물을 2D SWE로 계산|가장 저렴한 기준선|
|GNN-only|모든 입자에 residual GNN|전체 GNN 정확도·비용|
|Reversed|STREAM/POOL에 GNN|반대 라우팅 대조군|
|Ours|SPLASH에만 PI-GNN|제안 방식|
|Baseline GNS|모든 입자의 전체 가속도 예측|기존 GNS 비교|
|Simple 3D|중력·저항·반발만 사용|신경망 없는 3D 기준선|
|Optimized Ours|SPLASH ROI만 그래프 생성|최종 런타임 방식|

동일 코드로 다시 측정한 5,000입자·SPLASH 5% 조건에서 기존 Ours는
평균 183.67 FPS, Optimized Ours는 394.49 FPS를 기록했다. Optimized Ours는
SPLASH 5–50% 구간에서 학습 기반 조건 중 가장 빠르고 p95 60 FPS를 통과했다.
100% SPLASH에서는 통과하지 못하므로 제안 범위를 국소 ROI 조건으로 한정한다.
전체 수치와 95% 신뢰구간은
[`phase3/results_summary`](phase3/results_summary)에서 확인한다.

## Unreal 전달 파일

- [`ours_full_graph.onnx`](phase4/onnx/ours_full_graph.onnx): 전체 그래프 입력
- [`ours_roi_splash.onnx`](phase4/onnx/ours_roi_splash.onnx): SPLASH ROI 입력
- [`README_Unreal.md`](phase4/onnx/README_Unreal.md): 입력 tensor와 실행 순서
- 각 ONNX 옆 JSON: 입력 규격, 좌표 변환, SWE/GNN 라우팅 계약

두 ONNX는 같은 학습 가중치를 사용한다. STREAM과 POOL은 Unreal 측 SWE로
계산하며, ONNX는 SPLASH 가속도만 예측한다. ROI 버전의 성능 향상은 신경망을
바꾼 것이 아니라 불필요한 전체 그래프 생성을 제거해서 얻는다.

## 재현성과 주의점

- Water-3D split은 1000/100/100 trajectory이며 seed는 `20260809`이다.
- 원본 데이터와 checkpoint는 용량 때문에 Git에 포함하지 않는다.
- 저장소의 결과표는 전체 test split 집계이며 대표 장면을 임의 선택하지 않는다.
- Water-3D 좌표 RMSE와 자체 WCSPH의 SI 단위 RMSE를 직접 비교하지 않는다.
- Unreal 좌표는 Z-up 센티미터지만 ONNX 특징량은 학습 당시 Y-up 정규화를 따른다.
- NNE runtime이 동적 N/E와 `ScatterElements(reduction=add)`를 지원하는지 확인해야 한다.
