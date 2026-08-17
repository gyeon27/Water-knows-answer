# Phase 2 archive

이 폴더는 최종 Phase 2 실행 경로에서 사용하지 않는 초기 검증 코드를 보관한다.
삭제하지 않은 이유는 연구 개발 과정과 소규모 overfit 실험을 재현하기 위해서다.

- `prototypes/generate_debug_teachers.py`: 초기 debug teacher 생성기
- `prototypes/train_debug_gnn.py`: 소규모 overfit 확인용 학습기
- `prototypes/view_pi_gnn_splash.py`: 구형 단일 패널 SPLASH 뷰어

현재 기준 실행 코드는 상위 `phase2/`의 WCSPH·PI-GNN 파이프라인을 사용한다.
