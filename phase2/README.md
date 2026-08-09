# Phase 2 — SWE·3D 입자·PI-GCN 폭포 파이프라인

안정적인 상류·하류 물은 2D SWE로 계산하고, 절벽 flux만 3D 입자로 전환한다.
신경망은 밀도장이나 전체 운동을 생성하지 않고 충돌·분열 영역에서 기본 물리의
잔차 속도 `delta_v`만 예측한다.

## 1. 지형 생성

```powershell
python terrain_generation\generate_heightmaps.py
```

The default command writes four deterministic 257×257 terrains under
`terrains/`. Exact simulation data is stored in metres as NumPy arrays. The
16-bit PNG is intended for Unreal Landscape import and `preview.png` is only a
diagnostic image.

Each terrain contains:

- `height_meters.npy`: exact height field in metres;
- `height_u16.png`: normalized 16-bit height map;
- `surface_normal.npy`, `slope.npy`: terrain features;
- `cliff_mask.png`, `channel_mask.png`, `source_mask.png`: routing masks;
- `metadata.json`: world scale, decoding range, source position and flow rate;
- `preview.png`: blue channel, cyan source, orange cliff diagnostic overlay.

## 2. SWE와 절벽 flux 실행

```powershell
python run_swe.py --terrain single_cliff --seconds 12
```

`shallow_water/solver.py`는 finite-volume 방식으로 수심과 수평 운동량을
갱신한다. 절벽 flux는 2D 질량에서 차감되고 `particle_emitter.py`가 고정 질량
3D 입자로 변환한다. 입자 하나보다 작은 잔여 질량은 다음 프레임으로 이월한다.

Outputs contain depth/momentum frames, `waterfall_flux.npy`, and a mass-balance
summary. The flux columns are substep time, duration, position, discharge, and
3D velocity. Multiplying each face discharge by its duration gives the exact
volume transferred to 3D. This
is the deterministic physical baseline; the later PI-GCN predicts only a
residual correction in collision/splitting regions.

현재 12초 `single_cliff` 질량수지 오차는 유입량 대비 0.49%다.

## 3. 디버그 teacher trajectory

```powershell
python generate_debug_teachers.py --count 8 --frames 120
```

`datasets/debug/`에는 8개×120프레임 데이터가 생성된다. 이는 전체 연결 검증용
ballistic teacher이며 `not_final_training_teacher=true`로 표시된다. 최종 PI-GCN
학습 정답은 고해상도 SPH/MPM으로 다시 생성해야 한다.

## 4. 그래프와 디버그 GCN

```powershell
python train_debug_gnn.py --steps 200
```

로더는 SPLASH ROI와 반경 1-ring 문맥 입자를 선택해 33차원 노드 특징과 8차원
간선 특징을 만든다. 작은 GNS는 중력·공기저항 baseline에 더할 `delta_v`를
예측한다. 고정 그래프 overfit 결과는 0.05946에서 1.84×10⁻⁶으로 감소했다.

기본 명령은 연결 검증을 위해 그래프 하나를 overfit한다. `--cycle-graphs`를
추가하면 trajectory의 모든 유효 그래프를 순환한다.

## 5. 현재 통과 상태와 다음 작업

| 단계 | 상태 |
|---|---|
| Height field 4종 | 완료 |
| SWE 및 2D→3D flux | 구현 완료, 장기 안정성 검증 중 |
| 디버그 teacher 8×120 | 완료 |
| 그래프 로더 | 완료 |
| residual GNS 1-step overfit | 완료 |
| Taichi WCSPH 후보 teacher | 구현 및 단일 지형 검증 완료 |
| WCSPH 보정·다중 지형 데이터 | 다음 작업 |
| Residual GNS WCSPH baseline | 2 trajectories, 1,000-step 학습 완료 |
| rollout 8→16→32 | 예정 |
| Unreal 통합 | 예정 |

## 6. WCSPH teacher

```powershell
python generate_wcsph_teacher.py --terrain single_cliff --frames 120 --particle-mass 0.25
```

생성 결과를 3D로 재생하려면 다음 명령을 실행한다.

```powershell
python view_trajectory_3d.py
```

기본 화면은 굽은 절벽선·하부 침식 웅덩이·능선·바위 돌출부가 포함된
`natural_waterfall` 지형과 0.1 kg 고밀도 WCSPH trajectory를 연다.

마우스 왼쪽 드래그는 회전, 휠은 확대/축소, Space는 재생/일시정지,
좌우 방향키는 한 프레임 이동이다. 파란색은 STREAM, 빨간색은 충돌 SPLASH,
초록색은 지면 가까이의 POOL/표면류를 뜻한다.

재생 중에는 지형 메시를 캐시하고 최대 900개 대표 입자와 320개 속도 잔상만
그린다. 상태 표시줄의 `노드`는 전체 물리 입자 수, `표시`는 LOD 적용 후 실제로
그리는 입자 수다. 원본 trajectory와 GNN 학습 데이터는 줄이지 않는다.

기본 0.25 kg 설정은 기존 2 kg 설정보다 그래프 연결 밀도가 높아 GNN 학습에
적합하다. 생성 파일은 현재 `candidate_requires_calibration` 상태이며, 최종 정답
데이터로 확정하기 전에 정수압·자유낙하·충돌 검증과 압력 계수 보정을 수행한다.

이 백엔드는 SWE 절벽 flux로 입자를 공급받고 WCSPH 압력·점성, 중력, height-field
충돌을 계산한다. 디버그 ballistic teacher와 같은 NPZ 스키마을 사용하므로 그래프
로더를 변경하지 않고 학습 데이터만 교체할 수 있다.

## 7. WCSPH Residual GNS 학습

```powershell
python train_wcsph_gnn.py --steps 1000
```

각 trajectory 앞부분을 train, 뒤쪽을 validation으로 나누고 history 누수를 막기 위한
6프레임 간격을 둔다. 노드·간선·target은 train split 통계로 정규화하며 가장 낮은
validation RMSE checkpoint를 `checkpoints/wcsph_gns_baseline.pt`에 저장한다.

## 8. CUDA PI-GNN 및 비교 GUI

```powershell
powershell -ExecutionPolicy Bypass -File phase2\setup_gpu.ps1
& .\.venv-gpu\Scripts\python.exe phase2\generate_pi_dataset.py
& .\.venv-gpu\Scripts\python.exe phase2\train_pi_gnn.py --steps 10000
& .\.venv-gpu\Scripts\python.exe phase2\rollout_pi_gnn.py --steps 32
& .\.venv-gpu\Scripts\python.exe phase2\compare_pi_gnn_3d.py
```

학습기는 CUDA가 없으면 CPU로 내려가지 않고 중단한다. 노드 8,000개 또는 간선
120,000개 단위로 disconnected graph batch를 만들고 CUDA AMP를 사용한다. GUI는
좌측 WCSPH teacher와 우측 PI-GNN을 동기화하며 자율 rollout/1-step, 오버레이와
위치 오차 heatmap을 제공한다.
