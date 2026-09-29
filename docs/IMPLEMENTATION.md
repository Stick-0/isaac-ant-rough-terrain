# 구현: 원본 대비 변경

비교 기준은 `cailab-hy/IsaacLab_RS`의 `e83a5d2f11ca1b5f03b690e1978479e620c500e2`입니다. 원본 스냅샷은 [`reference/original`](../reference/original), 전체 변경은 [`isaaclab.patch`](../patches/isaaclab.patch)에 있습니다.

## 지형

원본은 `terrain_type="plane"`입니다. 수정본은 `AntTerrainImporter`와 procedural height-field generator를 사용합니다.

| 지형 | 선택 비율 | 설정 |
| --- | ---: | --- |
| 랜덤 요철 | 50% | 높이 노이즈 -0.05~0.05m, 양자화 0.01m |
| 파도 | 30% | amplitude 설정 범위 0.05~0.15m, 타일당 4개 wave |
| 경사 | 10% | slope 비율 0.05~0.15, 중앙 platform 2m |
| 역경사 | 10% | slope 비율 0.05~0.15, 중앙 platform 2m |

각 타일은 6×6m이며 120×120개를 연결합니다. 수평 해상도 0.2m, 수직 단위 0.005m, 테두리 폭 0, `slope_threshold=None`입니다. 난이도 curriculum은 사용하지 않습니다. 한 실행 안에서는 지형이 고정되고 **새 terrain seed로 생성할 때 지형이 바뀝니다**. 에피소드마다 새 메시를 만드는 방식이 아닙니다.

## 원본과 같은 격자 배치

TerrainImporter의 기본 generated-terrain 배치는 tile origin을 샘플링합니다. 이번 구현은 원본의 `_compute_env_origins_grid(num_envs, env_spacing)`를 사용해 XY 시작 위치를 유지합니다. 시작점 주변 1×1m의 5×5 ray 중 가장 높은 지면으로 기준 Z를 맞춥니다. 실제 로봇 초기 몸통 높이는 그 위에 로봇의 기본 높이를 더합니다.

원본과 같은 배치는 모든 개미가 동일한 한 점에서 출발한다는 의미가 아닙니다. 각 환경이 **5m 간격 격자**에 놓입니다. 환경 수를 바꾸면 격자 범위도 달라집니다.

## 높이 관측과 낙상 판정

기존 `base_pos_z` 대신 `body_z - ground_z`를 사용합니다. torso 아래를 향한 1개 ray로 지면을 읽으며 관측 차원은 60D로 유지합니다. 낙상도 이 상대 높이가 0.31m 아래인지로 판단합니다.

한 개의 아래 방향 ray는 앞쪽 지형을 미리 보는 height map이나 실제 RGB-D 카메라가 아닙니다. 이 프로젝트는 시뮬레이터의 이상적인 지면 질의 값을 사용합니다. 관측 차원은 같지만 높이 항목의 의미가 바뀐 점을 구분해야 합니다.

RayCaster의 환경별 reset buffer가 실제 모든 복제 환경을 포함하도록 `clone_in_fabric=False`를 사용합니다. 이를 유지하지 않은 상태에서 sensor 환경 수가 1로 잡혀 환경 인덱싱 오류가 발생했던 문제를 수정했습니다.

## 충돌 메시와 관측 메시 분리

최종 지형에는 **25,920,000개 삼각형**이 있습니다. 이 전체 메시를 하나의 PhysX collider로 cooking하면 실패할 수 있어 다음처럼 처리합니다.

1. 타일 경계의 공통 정점을 병합합니다.
2. 전체 메시 하나는 렌더링과 Warp ray 질의에 사용합니다. 이 prim에는 collision을 적용하지 않습니다.
3. face의 공간 위치를 기준으로 120m 단위 영역으로 묶습니다. chunk는 최대 1,000,000개 face를 가집니다.
4. 최종 설정에서는 720,000개 삼각형씩 36개 collider를 만듭니다. 각 collider의 원점을 해당 영역 중심으로 이동합니다.
5. convex 근사가 아닌 원래 triangle surface를 충돌면으로 사용합니다.

단순히 face 배열을 일정 길이로 자르면 지형 행 끝과 다음 행 처음이 같은 chunk에 들어가 bounding box가 지도 전체를 가로지를 수 있습니다. 반대로 너무 잘게 나누면 전역 collider와 수천 환경의 조합 수가 커져 GPU 메모리 부담이 증가합니다. 공간별 분할은 이 두 문제를 함께 고려한 구현입니다.

테두리 폭이 0일 때는 TerrainGenerator의 border mesh 생성을 건너뛰어 면적이 0인 삼각형이 추가되지 않게 했습니다. **ray가 지면을 읽는 것과 PhysX 충돌면이 정상 생성된 것은 별개**이므로 `missing_ground=0`만으로 충돌 문제를 배제하지 않았습니다.

## 안정성 실험과 도구

- `Isaac-Ant-Stable-v0`: 기존 전진 보상 등에 action-rate 제곱합 `-0.01`, 몸통 roll/pitch 각속도 제곱합 `-0.025`를 추가합니다.
- PPO: 학습률 `1e-4` 고정, entropy 계수 `0.001`, 300회 추가 학습. 기존 actor/critic 가중치를 읽고 옵티마이저는 새로 시작합니다.
- `train.py --warm_start`: 위 가중치 초기화 경로를 지원하며 `--resume`과 동시에 사용할 수 없습니다.
- `play.py --diagnostics --max_steps`: 이동·속도·액션·종료·지면 질의를 제한된 시간 동안 관찰할 수 있습니다.
- `evaluate_ant.py`: 체크포인트별로 같은 seed로 리셋하고, 각 개미의 첫 에피소드만 집계합니다.

강한 패널티 실험은 action-rate `-0.05`, body-sway `-0.1`, 200회 추가 학습을 사용했습니다. 검증 지형에서 회귀했으므로 현재 기본 태스크 설정에는 반영하지 않았습니다.

## 후속 보상 재설계

`ant_recovery_env_cfg.py`와 `recovery_mdp.py`를 추가하고 `Isaac-Ant-Recovery-v0`로 등록했다. 4.5m/s 전진 보상 상한, 낙상 사건 -10, 몸높이·기울기 위험 보상을 포함한다. 기존 두 task의 설정은 보존했다. [수식과 의도·동일 예산 비교](REWARD_DESIGN.md).
