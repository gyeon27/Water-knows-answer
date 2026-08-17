# Phase 3 archive

최종 모델·평가·게임 런타임에 직접 사용되지 않는 실험 코드를 분리한 폴더다.
메인 진입점은 `phase3/run_phase3.py`, `phase3/evaluation.py`,
`phase3/benchmark.py`, `phase3/continuous_terrain_runtime.py`이다.

## 하위 폴더

- `external_finetuning/`: 외부 DFSPH teacher에 별도로 재학습한 선택 실험
- `legacy_terrain/`: 무한 수원 런타임 이전의 유한 terrain rollout 생성기
- `pipeline_helpers/`: 다운로드·학습 중 사용한 일회성 연속 실행 도우미

아카이브 코드는 결과 재현을 위해 실행 가능한 상태로 유지하지만 README의
빠른 시작과 최종 Unreal 경로에서는 사용하지 않는다.
