# Phase 3 — 3D 공개 데이터 일반화·성능 검증·Novelty 실증

작성일: 2026-08-15
상태: 전체 실행 파이프라인 구현 완료, Water-3D 데이터 준비 및 실험 실행 중

## 1. 목적

Phase 2는 자체 WCSPH teacher에서 PI-GNN 학습 가능성을 확인했다. Phase 3는
자체 데이터 편향, 직접적인 실행 속도 측정 부족, 비교 조건 부족이라는 문제를
해결한다. 결과가 가설과 다르더라도 설정을 사후 조정하지 않고 그대로 보고한다.

| Track | 질문 | 산출물 |
|---|---|---|
| A | 실제 게임 프레임에서 얼마나 빠른가? | 입자 수·SPLASH 비율별 단계 타이밍 |
| B | 자체 데이터에만 맞은 모델인가? | Water-3D zero-shot/retrain/GNS 비교 |
| C | 제안한 솔버 배치가 타당한가? | 동일 입력의 6조건 ablation |
| D | 단순 3D 물리 대신 GNN이 필요한가? | 정확도·속도·실패 사례 기반 결론 |

## 2. 확정 실험 조건과 변경점

초기 계획의 2D WaterRamps 우선 실험은 연구 목표와 차원이 달라 필수 실험에서
제외했다. 최종 계획은 DeepMind GNS의 **Water-3D 전체 1000/100/100
trajectories**를 사용한다. WaterDropSample은 어댑터 회귀 테스트에만 사용한다.

| 항목 | 초기안 | 최종 확정안 |
|---|---|---|
| 공개 데이터 | WaterRamps 우선 | Water-3D 전체 split |
| 비교 조건 | 5조건 | Simple-3D를 추가한 6조건 |
| 학습 GPU | Phase 2는 실제 RTX 3080 | Phase 3는 RTX 3060 12GB |
| 학습량 | 최초 주요 모델별 20,000 step 계획 | 1차 실험은 모델별 5,000 step |
| 모델 규모 | Phase 2 6블록 중심 | hidden 128, message passing 10블록 |
| 결과 판단 | 제안 방식 우위 기대 | 우위가 없으면 “입증되지 않음” 자동 보고 |
| 렌더링 | 일부 계획에 혼재 | Phase 4로 완전히 분리 |

고정 seed는 `20260809`이다. 데이터 순서, 시간창, 코드 commit, GPU와 CUDA
정보를 T7의 `run_manifest.json`과 `experiment_config.json`에 기록한다.

> 하드웨어 구분: Phase 2의 10,000-step PI-GNN 학습 결과는 실제 RTX 3080에서
> 얻었다. 현재 Water-3D 다운로드·학습·벤치마크를 포함한 Phase 3 실험은
> RTX 3060 12GB에서 수행한다. 두 GPU의 프레임타임을 같은 하드웨어 결과처럼
> 직접 비교하지 않는다.

## 3. 데이터 저장과 준비

대용량 자료는 볼륨 라벨에 `T7`이 포함되고 여유 공간 300GB 이상인 드라이브의
`WaterKnowsAnswer_Phase3`에 저장한다.

```text
WaterKnowsAnswer_Phase3/
├── raw/          # 원본 TFRecord와 metadata
├── indices/      # trajectory offset index와 검증 manifest
├── checkpoints/  # 모델·optimizer·scheduler·AMP scaler
├── rollouts/     # 평가 JSON과 대표 trajectory NPZ
├── benchmark/    # 프레임별 raw CSV
└── reports/      # 표·그래프·결론 문서
```

원본 파일은 SHA-256으로 전체 무결성을 기록하고, TFRecord의 각 레코드는
CRC32C를 검사한다. 위치를 중복 NPZ로 변환하지 않고 `[payload offset, length]`
인덱스를 생성한다. 한 trajectory만 메모리에 유지하는 LRU 방식으로 RAM 사용량을
제한한다. 중단된 다운로드와 학습은 같은 명령으로 재개된다.

Train 1,000개 trajectory에서 trajectory별 고정된 5개 시간창을 사용하여 정확히
5,000 학습 표본을 구성한다. Validation/Test 각각 100개도 전부 평가한다. 최초
20,000-step 계획은 WCSPH validation이 2,500~4,500 step 사이 약 0.25%만 개선된
초기 plateau와 전체 실험 시간을 고려해 변경했다. 모든 비교 모델에 동일한 5,000
step 예산을 적용하며, 마지막 구간에서도 validation이 명확히 개선될 때만 모든
비교 모델을 같은 기준으로 연장한다.

## 4. 공통 3D 그래프

자체 WCSPH와 Water-3D를 `UnifiedParticleGraphDataset`의 같은 표현으로 변환한다.

### 노드 입력

| 특징 | 차원 |
|---|---:|
| 최근 5개 3D 속도 | 15 |
| 하한·상한 경계까지 정규화 거리 | 6 |
| STREAM/SPLASH/POOL one-hot | 3 |
| 중력 방향 | 3 |
| 합계 | 27 |

particle type은 별도 ID와 16차원 학습 임베딩으로 처리한다. Water-3D의
`particle_type=3`은 고정 경계 입자로 유지한다. WCSPH height field는 연결 반경
간격의 경계 입자로 표본화한다.

간선은 반경 이내 이웃으로 매 프레임 재구성한다. 입력은 연결 반경으로 나눈
상대 위치 3차원과 정규화 거리 1차원이다. 그래프를 batch로 합칠 때 노드 offset을
적용하며 서로 다른 trajectory 사이에는 간선을 만들지 않는다.

공개 데이터에는 상태 정답이 없으므로 현재 프레임의 속도, 이전 가속도, 국소
밀도와 경계 거리만으로 상태를 결정한다. 미래 프레임과 test split 통계는 사용하지
않는다.

## 5. 학습 모델

모든 모델은 hidden 128, message-passing 10블록, LayerNorm 구조를 사용한다.

1. WCSPH 공통 피처 PI-GNN — Water-3D zero-shot 평가
2. Water-3D GNN-only residual
3. Water-3D reversed-routing residual
4. Water-3D 제안 방식 PI-GNN
5. Water-3D 원형 GNS acceleration baseline

각 모델은 5,000 step, AdamW, CUDA AMP, gradient clipping 1.0,
validation-best checkpoint로 학습한다. CUDA가 없으면 CPU로 자동 전환하지 않고
중단한다.

제안 residual 모델의 손실은 다음과 같다.

```text
L = 1.00 L_supervised
  + 0.10 L_penetration
  + 0.05 L_momentum
  + 0.05 L_density
  + 0.05 L_energy
```

밀도·운동량·에너지 집계는 AMP 안에서도 float32로 계산한다. Baseline GNS는
비교의 공정성을 위해 전체 가속도 지도 손실만 사용한다.

## 6. 6조건 Ablation

모든 조건은 같은 초기 상태, 프레임과 teacher를 사용한다.

| ID | 조건 | 동작 |
|---|---|---|
| A | SWE-only | 모든 유체를 2D height field로 투영해 SWE 적용 |
| B | GNN-only | 모든 유체 입자에 residual PI-GNN 적용 |
| C | Reversed | STREAM/POOL에 GNN, SPLASH에 SWE 적용 |
| D | Ours | STREAM/POOL은 해석 솔버·SWE, SPLASH만 PI-GNN |
| E | Baseline GNS | 라우팅 없이 전체 가속도를 GNS로 예측 |
| F | Simple 3D | D와 같은 라우팅, SPLASH는 중력·저항·반발만 적용 |

Water-3D test 전체를 SPLASH 비율 순위로 quiet/complex/violent 세 그룹으로
나눈다. 대표 화면은 각 그룹의 중앙값 trajectory를 자동 선택한다. held-out
자연형 WCSPH 폭포는 연구 목표 장면으로 별도 평가한다.

## 7. 평가지표

- 1/8/16/32/100-step 위치·속도 RMSE
- 경계 침투율
- 상대 밀도 오차
- 질량 정규화 운동량 오차
- 기계적 에너지 초과량
- 활성 유체 입자 수 보존 오차
- 그래프·라우팅·SWE·GNN·블렌딩·전체 프레임타임

Water-3D 지표는 데이터셋 좌표/step 단위이다. 자체 WCSPH의 SI 단위 RMSE와
직접 숫자를 비교하지 않는다.

대표 trajectory는 teacher/predicted 위치·속도, particle ID, type, 위치 오차를
NPZ로 저장한다. 정성 화면을 사람이 골라 좋은 사례만 제시하지 않는다.

## 8. 성능 벤치마크

워밍업 10프레임 뒤 300프레임을 측정한다.

- 입자 수: 2k / 5k / 10k / 20k / 50k
- SPLASH 비율: 5% / 25% / 50% / 100%
- 목표 FPS: 30 / 60 / 120 / 144
- 통계: 평균 / 표준편차 / p95 / 최대 / 달성 FPS

PyTorch GPU는 CUDA Event, Taichi는 `ti.sync()`, CPU SWE는 양쪽 GPU 동기화
후 `perf_counter_ns()`를 사용한다. 20k/50k에서도 시각화 LOD를 적용하지 않는다.

## 9. 실행법

프로젝트 루트에서 GPU 가상환경을 사용한다.

```powershell
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 doctor --data-root auto
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 prepare --data-root auto
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 train --data-root auto
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 evaluate --data-root auto
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 benchmark --data-root auto
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 report --data-root auto
```

전체를 순차 실행하려면 다음 명령 하나를 사용한다.

```powershell
& .\.venv-gpu\Scripts\python.exe -m phase3.run_phase3 all --data-root auto
```

## 10. 자동 산출물

- `benchmark_results.csv`
- 6조건×장면 성능표
- 입자 수–프레임타임 그래프
- SPLASH 비율–프레임타임 그래프
- rollout horizon–오차 그래프
- zero-shot/retrain/GNS 비교표
- 조건별 대표 실패 NPZ/프레임
- `phase3_results.md`
- `phase3_discussion_gnn_justification.md`

결론 생성기는 실제 JSON/CSV 수치를 인용한다. D가 B보다 빠르거나 A보다
정확하다는 조건이 성립하지 않으면 novelty를 “현재 실험에서는 입증되지 않음”으로
기록하고 적용 가능한 입자 수와 SPLASH 비율 범위만 제시한다.

## 11. 테스트와 완료 기준

- WaterDropSample 6개 trajectory 어댑터 회귀 테스트
- Water-3D 차원·particle type·경계 입자 및 split 중복 검사
- 현재 프레임 특징의 미래 정보 누출 검사
- WCSPH/Water-3D 공통 특징 차원·정규화 검사
- 단독 그래프와 disconnected batch 출력 일치
- CUDA AMP forward/backward와 NaN/Inf 검사
- 물리 손실별 정상/위반 단위 테스트
- checkpoint 재개 시 step·scheduler·scaler·데이터 위치 유지
- 6조건의 동일 입력·teacher 검사
- raw benchmark row 수 검사
- 보고서 수치와 원본 CSV/JSON 일치 검사
- manifest에 전체 1000/100/100 trajectory 사용 기록

Phase 3 완료는 결과의 우위가 아니라 위 사전 정의 실험과 정직한 보고서가 모두
생성된 시점으로 판단한다. Unreal 통합과 최종 렌더링은 [Phase4.md](Phase4.md)의
범위이다.

## 12. 사후 런타임 최적화 실험

본 실험은 사전 정의한 A–F 결과를 바꾸는 재학습이 아니라, 선택적 GNN의 실제
실행 순서를 구현한 별도 사후 최적화 실험이다. 기존 벤치마크는 모든 입자의
radius graph를 만든 뒤 SPLASH 이외의 노드를 제거해 선택적 계산의 이점을 일부
상쇄했다. 최적화 버전은 다음 순서를 사용한다.

1. 전체 입자에 저비용 라우팅을 먼저 적용한다.
2. SPLASH ROI만 추출한다.
3. ROI에 대해서만 radius graph와 PI-GNN 추론을 수행한다.
4. 2,000개 미만의 작은 그래프는 `cKDTree` 단일 스레드를 사용해 worker 생성
   오버헤드를 제거한다.

5,000입자·SPLASH 5%·300프레임에서 Ours는 9.62 ms(104.0 FPS)에서
4.84 ms(206.8 FPS)로 개선됐다. 실제 test에서 SPLASH가 존재하는 54개
trajectory를 비교한 결과, 전체 그래프의 정규화 가속도 RMSE는 2.5509,
ROI 그래프는 2.5852로 1.35% 증가했다. 따라서 약 2배의 실행 속도 개선과 작은
정확도 손실 사이의 trade-off로 보고하며, 사전 정의한 정확도 표와 혼합하지 않는다.

관련 원본은 `benchmark_roi_5k_5pct_v2.csv`, `benchmark_roi_7k_5pct.csv`,
`roi_accuracy_validation.json`에 저장한다.

## 참고 문헌

- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks*, ICML 2020
- DeepMind `learning_to_simulate` 공식 Water-3D 데이터와 GNS 구조
