# Unreal NNE용 PI-GNN 모델

두 ONNX는 같은 `ours/best.pt` 가중치를 사용하며 입력 그래프 범위가 다르다.

- `ours_full_graph.onnx`: 전체 그래프를 입력하고 SPLASH 출력만 사용한다.
- `ours_roi_splash.onnx`: 먼저 라우팅한 뒤 SPLASH ROI만 입력한다.

두 방식 모두 STREAM과 POOL은 ONNX로 계산하지 않고 Unreal 측 2D 천해방정식
솔버로 전진시킨다. SPLASH가 없으면 최적화형 NNE 호출을 생략한다. ROI형에서는
`roi_to_global` 인덱스를 Unreal 측에 보관하고 ONNX 출력을 전체 입자 배열에
scatter한 뒤 SWE 경계와 블렌딩한다.

ONNX 입력은 `node_features[N,27]`, `particle_type[N]`,
`edge_features[E,4]`, `edge_index[2,E]`이며 출력은
`acceleration[N,3]`이다. 세부 규격은 각 JSON manifest와 parity report를 따른다.
