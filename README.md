# Water-knows-answer

2026 제주과학고 과제연구 — 고가연·엄시후

넓고 안정적인 물은 2D 천해방정식(SWE)으로 계산하고, 절벽에서 추출한
유량만 3D 입자로 전환한 뒤 충돌·분열 영역을 PI-GCN 잔차 모델로 보정하는
차원 가변 폭포 시뮬레이션 연구입니다. 최종 목표는 임의의 지형으로부터
강·폭포를 자동 생성하고 Unreal Engine에서 실시간으로 실행하는 것입니다.

## Project structure

- `fluid_routing/`: Phase 1 입자 상태 라우팅 및 3D 충돌 기준 솔버
- `phase2/terrain_generation/`: 결정론적 height-field 지형 생성
- `phase2/shallow_water/`: 보존형 SWE와 절벽 2D→3D flux
- `phase2/teacher/`: 디버그 trajectory 저장기
- `phase2/gnn/`: 그래프 데이터 로더와 residual GNS

현재 상태: SWE 질량수지 오차 0.49%, 디버그 trajectory 8×120 생성,
33차원 노드/8차원 엣지 그래프 로더 구현, 1-step 고정 그래프 overfit 완료.
현재 데이터는 연결 검증용이며 최종 모델 학습에는 고해상도 SPH/MPM teacher가 필요합니다.
