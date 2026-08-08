# 모든 임계값/하이퍼파라미터를 여기 한 곳에 모아둔다.
# 씬별 튜닝과 ablation 실험 시 이 파일만 바꾸면 되도록 하기 위함.

# ---------------------------------------------------------------------------
# 입자 시스템
# ---------------------------------------------------------------------------
# Fast development default. Use --n-particles 8000 for final-quality runs.
N_PARTICLES = 4000

# ---------------------------------------------------------------------------
# 도메인 / 초기 씬 (물기둥 낙하 -> 바닥에 고임)
# ---------------------------------------------------------------------------
DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (4.0, 5.0, 4.0)

COLUMN_MIN = (1.5, 3.0, 1.5)
COLUMN_MAX = (2.5, 4.5, 2.5)

# 초기 배치를 순수 무작위가 아니라 "지터를 준 격자"로 만든다 (particles.py 참고).
# 1.0에 가까울수록 격자칸 절반 범위까지 흔들려서 자연스러운 무작위성을 주고,
# 0에 가까울수록 완전한 정격자가 된다. 너무 크면(1.0 이상) 이웃 격자칸과 겹쳐서
# 다시 근접-쌍 문제가 생길 수 있다.
LATTICE_JITTER_FRACTION = 0.3

# ---------------------------------------------------------------------------
# 성긴 보조 격자
# ---------------------------------------------------------------------------
CELL_SIZE = 0.5
GRID_RES = (
    max(1, round((DOMAIN_MAX[0] - DOMAIN_MIN[0]) / CELL_SIZE)),
    max(1, round((DOMAIN_MAX[1] - DOMAIN_MIN[1]) / CELL_SIZE)),
    max(1, round((DOMAIN_MAX[2] - DOMAIN_MIN[2]) / CELL_SIZE)),
)
CELL_VOLUME = CELL_SIZE ** 3

# ---------------------------------------------------------------------------
# 물리
# ---------------------------------------------------------------------------
GRAVITY = -9.8  # y축 방향 가속도
DT = 1.0 / 60.0
G_HAT = (0.0, -1.0, 0.0)
EPS = 1e-4

# ---------------------------------------------------------------------------
# 상태 라벨
# ---------------------------------------------------------------------------
STATE_STREAM = 0
STATE_SPLASH = 1
STATE_POOL = 2

# ---------------------------------------------------------------------------
# 특징량 임계값 (4.1 이중 임계값 상태표)
# ---------------------------------------------------------------------------
# SPLASH: D_splash
# 주의: 이 유한차분 기반 D_splash는 셀 크기/입자수에 따라 절대 스케일이 크게 달라진다.
# 아래 기본값은 config.py 상단의 기본 씬(N_PARTICLES, GRID_RES 등)에 맞춰 실측 보정한 값이며,
# 씬을 바꾸면 이 값도 함께 재보정해야 한다 (features.py의 D_splash 계산 참고).
SPLASH_ENTER_THRESHOLD = 5.0
SPLASH_EXIT_THRESHOLD = 2.5
IMPACT_SPLASH_MIN_DOWNWARD_SPEED = 0.8
DIVERGENCE_DENSITY_LAMBDA = 0.5  # D_splash 수식의 lambda

# 밀도 변화율 정규화 기준값.
# 주의: (N_t - N_t-1) / N_t-1 처럼 "이전 프레임 밀도"로 나누면, 셀이 이전 프레임에
# 비어있다가(밀도 0) 이번 프레임에 막 채워진 경우 분모가 0에 가까워 변화율이 수백~수천 배로
# 폭발한다 (물이 새 영역으로 흘러들기만 해도 항상 SPLASH로 오판되는 원인이었음).
# 대신 물기둥의 대표 밀도(초기 씬의 입자수/부피)라는 "고정" 기준값으로 나눠 정규화한다.
_COLUMN_VOLUME = (
    (COLUMN_MAX[0] - COLUMN_MIN[0]) * (COLUMN_MAX[1] - COLUMN_MIN[1]) * (COLUMN_MAX[2] - COLUMN_MIN[2])
)
def density_rate_ref(n_particles):
    """Return the reference density for the actual particle count."""
    return n_particles / _COLUMN_VOLUME


DENSITY_RATE_REF = density_rate_ref(N_PARTICLES)

# STREAM: A_stream / N_density
STREAM_ALIGN_ENTER_THRESHOLD = 0.85
STREAM_ALIGN_EXIT_THRESHOLD = 0.6
STREAM_DENSITY_HIGH_THRESHOLD = 60.0  # 단위: 입자수 / 부피, 씬에 맞게 조정

# POOL: S_pool (시간창 필요)
POOL_WINDOW_FRAMES = 8       # N_win
POOL_VEL_THRESHOLD = 0.05    # v_th
POOL_HEIGHT_THRESHOLD = 0.01  # h_th
POOL_ENTER_STREAK = 4        # S_pool==True가 연속으로 몇 프레임이어야 진입하는지
POOL_EXIT_STREAK = 1         # S_pool==False가 연속으로 몇 프레임이면 이탈하는지

# ---------------------------------------------------------------------------
# 라우팅 갱신 주기 / 전파 마진
# ---------------------------------------------------------------------------
FULL_RECLASSIFY_PERIOD = 4   # 프레임마다 전체 재분류
# SPLASH 팽창 시 이웃 정의: 6(면 인접) 또는 26(면+모서리+꼭짓점).
# 주의: 기본 씬의 물기둥은 격자 셀 크기(0.5) 기준으로 폭이 2~3셀밖에 안 되는 작은 덩어리다.
# 26-이웃 팽창은 한 번에 사방으로 한 겹씩 다 덮기 때문에, 국소적으로만 SPLASH가 발생해도
# 팽창 몇 번(재분류 주기 4프레임 단위) 만에 물기둥 전체를 SPLASH로 뒤덮어버려서 STREAM/POOL과
# 공존하지 못하고 사실상 전역이 하나의 상태로만 움직이는 것처럼 보이게 만든다.
# 6-이웃(면 인접)으로 바꾸면 팽창 범위가 좁아져 국소적인 상태 공존이 더 잘 관찰된다.
DILATION_NEIGHBORHOOD = 6
DILATION_MIN_D_SPLASH = 2.5

# ---------------------------------------------------------------------------
# 경계 블렌딩
# ---------------------------------------------------------------------------
BLEND_WINDOW_K = 6

# ---------------------------------------------------------------------------
# 깜빡임(flicker) 시각화 감지 (검증용, 물리에는 영향 없음)
# ---------------------------------------------------------------------------
FLICKER_WINDOW_FRAMES = 60   # 이 프레임 수마다 recent_transition_count를 리셋
FLICKER_COUNT_THRESHOLD = 3  # 윈도우 내 전환 횟수가 이 값 이상이면 깜빡임 의심

# ---------------------------------------------------------------------------
# STREAM 솔버 (간이 SPH: 밀도 추정 + 상태방정식 압력 + 점성력)
# ---------------------------------------------------------------------------
# 질량-스프링 모델(정해진 "자연 길이"로 돌아가려는 탄성력)은 유체가 아니라 고체 역학
# 모델이라 압력/밀도 개념이 없다. 그래서 압력 기반의 최소 SPH(Müller et al. 2003 스타일)로
# 교체한다: 이웃 밀도를 커널로 추정하고, 기준밀도 대비 편차로 압력을 구해 압력기울기 힘을
# 가하고, 약한 점성력을 더한다.
#
# 평균 입자 간격 ((부피/입자수)^(1/3)) 기준으로 스무딩 반경을 잡는다.
# 주의: h가 작을수록 커널 정규화 상수(~1/h^6)가 급격히 커져서 근접 입자쌍의 힘이
# 뻣뻣해진다. 처음엔 SPH 관례대로 이웃 20~40개를 노리고 h=간격*2로 잡았는데
# (예전 스프링 모델의 탐색 반경 0.3, 간격의 5배보다는 훨씬 좁힌 값), 실측해보니
# 극심하게 불안정했다(중력만으로는 나올 수 없는 속도가 관측됨). h=간격*4(이웃 ~250개
# 안팎)까지 넓히고 나서야 안정적이었다 — 계산량은 늘지만 안정성이 우선이다.
def stream_particle_spacing(n_particles):
    """Return the representative spacing for the actual particle count."""
    return (_COLUMN_VOLUME / n_particles) ** (1 / 3)


_STREAM_PARTICLE_SPACING = stream_particle_spacing(N_PARTICLES)
STREAM_SPH_RADIUS_FACTOR = 3.0
STREAM_SPH_RADIUS = _STREAM_PARTICLE_SPACING * STREAM_SPH_RADIUS_FACTOR
STREAM_SPH_REST_DENSITY = DENSITY_RATE_REF  # 물기둥의 대표 밀도(입자수/부피)를 기준밀도로 재사용
STREAM_SPH_STIFFNESS = 5.5    # 상태방정식 계수 k: pressure = k * (rho - rho0)
STREAM_SPH_VISCOSITY = 0.4    # 물처럼 흐르도록 과도한 속도 결합을 줄인 점성 계수
STREAM_SURFACE_TENSION = 0.06
STREAM_SURFACE_DENSITY_RATIO = 0.9
STREAM_SPH_WARMUP_FRAMES = 24
STREAM_MAX_ACCELERATION = 55.0
# 근접쌍 힘 폭발 방지용 유효 거리 하한 (h에 대한 비율). stream_solver.py의 spiky_grad 참고.
STREAM_SPH_MIN_DIST_RATIO = 0.3
# SPH 압력힘은 프레임 간격(DT=1/60s)에 비해 뻣뻣해서(stiff), 한 번에 통째로 적분하면
# 불안정하다. 한 프레임을 이만큼의 작은 스텝으로 쪼개 적분해야 안정적이었다(실측 확인).
STREAM_SPH_SUBSTEPS = 4

# ---------------------------------------------------------------------------
# SPLASH 임시 솔버 (추후 GNN으로 교체)
# ---------------------------------------------------------------------------
# 반발 반경이 실제 평균 입자 간격(~0.057)보다 훨씬 크면(예전 값 0.12, 간격의 2배 이상),
# 정상적으로 촘촘히 패킹된 물기둥을 "겹쳐있다(overlap)"고 착각해 막 SPLASH로 분류된
# 순간 확 밀어내며 터지는 원인이 된다 (stream_solver.py의 예전 스프링 자연길이 버그와
# 동일한 종류). 실제 간격에 맞춰 재보정한다.
SPLASH_REPULSION_RADIUS = _STREAM_PARTICLE_SPACING * 1.5
SPLASH_REPULSION_STIFFNESS = 60.0
SPLASH_COHESION_RADIUS_FACTOR = 2.2
SPLASH_COHESION_STIFFNESS = 3.0
# Continuous damping rate (1/s), applied as exp(-rate * dt).
SPLASH_VELOCITY_DAMPING_RATE = 0.7
SPLASH_FLOOR_RESTITUTION = 0.28
SPLASH_MAX_UPWARD_SPEED = 2.5
SPLASH_LATERAL_TRANSFER = 0.18
SPLASH_MAX_LATERAL_KICK = 0.8

# ---------------------------------------------------------------------------
# POOL 솔버 (shallow water)
# ---------------------------------------------------------------------------
POOL_FLOOR_Y = DOMAIN_MIN[1]
POOL_MIN_DEPTH = 1e-3
POOL_VELOCITY_DAMPING = 0.995
POOL_HEIGHT_RELAXATION_RATE = 8.0  # 1/s; avoids snapping particles to the surface

# ---------------------------------------------------------------------------
# 경계 충돌
# ---------------------------------------------------------------------------
BOUNDARY_RESTITUTION = 0.1
