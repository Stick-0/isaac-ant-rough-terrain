# 재현 방법

## 실행 환경

측정 환경은 RTX 4090 24GB, Python 3.11, Isaac Sim 5.1.0, Isaac Lab 릴리스 2.3.0, PyTorch 2.7.0+cu128, RSL-RL 3.0.1입니다. 설치 패키지 `isaaclab`의 metadata 버전은 0.47.2로, 저장소의 릴리스 VERSION과 표기가 다릅니다. 정확한 값은 [`manifest.json`](../manifest.json)에 있습니다.

Isaac Sim 실행 환경과 로봇 에셋은 이 저장소에 포함하지 않습니다. [기반 저장소](https://github.com/cailab-hy/IsaacLab_RS/tree/e83a5d2f11ca1b5f03b690e1978479e620c500e2)의 설치 안내에 따라 준비한 환경을 사용합니다. 기존 사용자 환경 이름은 `lerobot-arena`였지만 같은 이름의 Conda 환경이 필수인 것은 아닙니다.

## 변경 코드 적용

이미 수정한 작업 폴더 위에 다시 적용하지 말고 별도 checkout을 사용합니다. 적용 스크립트는 기반 커밋과 `git apply --check`를 먼저 검사합니다.

```bash
git clone https://github.com/Stick-0/isaac-ant-rough-terrain.git
git clone https://github.com/cailab-hy/IsaacLab_RS.git IsaacLab_RS_ant
git -C IsaacLab_RS_ant checkout e83a5d2f11ca1b5f03b690e1978479e620c500e2
python isaac-ant-rough-terrain/scripts/apply_overlay.py IsaacLab_RS_ant --check
python isaac-ant-rough-terrain/scripts/apply_overlay.py IsaacLab_RS_ant
```

이후 공개 저장소 루트에서 환경 경로를 설정합니다.

```bash
export ISAACLAB_ROOT="$(cd ../IsaacLab_RS_ant && pwd)"
export ANT_REPORT_ROOT="$PWD"
```

`overlay/`의 파일은 패치 적용 후 만들어지는 소스와 동일합니다. Python 파일만 덮어쓰는 것보다 위 패치 명령으로 기반 버전을 확인하는 편이 재현에 적합합니다.

## 저장한 모델 재생

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \
  --task Isaac-Ant-v0 --num_envs 64 \
  --checkpoint "$ANT_REPORT_ROOT/artifacts/models/rough.pt" --diagnostics
```

추가 보상 모델은 같은 명령에서 체크포인트만 `stable.pt`로 바꿔 비교할 수 있습니다. 보상 설정까지 추가 학습과 동일하게 실행하려면 태스크를 `Isaac-Ant-Stable-v0`로 바꿉니다. 평가 스크립트는 보상 총합의 차이를 피하고 같은 종료 조건으로 비교하기 위해 항상 기본 험지 태스크를 사용합니다.

## 동일 조건 평가

```bash
bash scripts/evaluate.sh 2001 2002
```

새 결과는 `results/reproduced/`에 저장됩니다. 저장한 측정값을 덮어쓰지 않습니다.

```bash
python -m pip install -r requirements-analysis.txt
python scripts/analyze.py --results-dir results/reproduced --output-dir results/reproduced_summary
```

집계 JSON/CSV는 지정한 output 폴더에 생성되며 그림은 `artifacts/figures/`에 다시 생성됩니다. 다른 새 지형은 예를 들어 `bash scripts/evaluate.sh 3001 3002`로 평가할 수 있습니다. 지형 seed를 더 평가하는 것과 학습 seed를 여러 번 반복하는 것은 다른 실험입니다.

## 원본 평지 학습

**패치를 적용하지 않은 별도의 기반 checkout**에서 실행합니다. 패치가 적용된 checkout에서는 동일한 이름 `Isaac-Ant-v0`가 험지 태스크가 됩니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-v0 --headless --seed 42 \
  --num_envs 4096 --max_iterations 1000 --run_name flat_seed42
```

## 험지 학습

패치를 적용한 checkout에서 실행합니다.

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-Ant-v0 --headless --seed 42 \
  --num_envs 4096 --max_iterations 1000 --run_name rough_seed42
```

학습 결과의 저장 위치는 실행 working directory 아래 `logs/rsl_rl/ant/`입니다. 새 학습 결과로 표를 만들려면 해당 체크포인트로 평가 명령을 다시 실행해야 합니다. 재학습이 공개 모델과 bitwise 같은 가중치를 보장하지는 않습니다.

## 안정성 추가 학습

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-Ant-Stable-v0 --headless --seed 42 \
  --num_envs 4096 --max_iterations 300 \
  --warm_start "$ANT_REPORT_ROOT/artifacts/models/rough.pt" \
  --run_name stability_finetune
```

`--warm_start`는 actor·critic 가중치와 정책 noise parameter를 가져오고, 옵티마이저와 반복 카운터는 새로 시작합니다. 실행을 중단했다가 같은 설정으로 이어갈 때 사용하는 `--resume`과는 다릅니다.

강한 패널티 비교 실험을 재현하려면:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-Ant-Stable-v0 --headless --seed 42 \
  --num_envs 4096 --max_iterations 200 \
  --warm_start "$ANT_REPORT_ROOT/artifacts/models/rough.pt" \
  --run_name stronger_penalties \
  agent.experiment_name=ant_stable_ablation \
  env.rewards.action_rate.weight=-0.05 env.rewards.body_sway.weight=-0.1
```

## 공개 파일 검증

```bash
python scripts/verify_package.py
```

모델 SHA-256, 평가 집계의 일관성, 공개 경로, Python 문법을 검사합니다. 전체 소스 패치를 원본에서 확인하려면 `apply_overlay.py ... --check`를 별도로 실행합니다. 공개 작업에서는 고정한 원본 파일로 만든 임시 checkout에서 패치를 적용한 뒤 결과 12개 파일이 `overlay/`와 byte 단위로 같은지도 확인했습니다.

## 새 안정성 보상과 동일 예산 대조 실험

새 태스크는 `Isaac-Ant-Recovery-v0`이다. 이전 Stable 설정은 기존 결과 재현을 위해 보존했다. [보상 설계와 실행 명령](REWARD_DESIGN.md)을 참고한다.

```bash
# 같은 초기 모델에서 원래 보상과 새 보상을 각각 600회 추가 학습
bash scripts/train_rewards.sh

# 공개 모델 4개를 같은 두 지도에서 비교하고 에피소드별 결과 저장
bash scripts/evaluate_rewards.sh 4001 4002
python scripts/analyze_rewards.py --results-dir results/reproduced_rewards \
  --output-dir results/reproduced_rewards_summary

# 새 보상 모델 재생
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \
  --task Isaac-Ant-Recovery-v0 --num_envs 64 \
  --checkpoint "$ANT_REPORT_ROOT/artifacts/models/recovery.pt" --diagnostics
```
