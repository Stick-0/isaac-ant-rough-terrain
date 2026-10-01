# 세 모델 실행과 재현

본문의 모델은 평지 `flat.pt`, 험지 `rough.pt`, 추가 학습 `recovery.pt`다. 중간 실험 명령은 [실험 기록](archive/reproduce_all.md)에 모았다.

## 공통 준비

측정 환경은 RTX 4090 24GB, Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3, RSL-RL 3.0.1이다. 정확한 패키지 버전은 [manifest.json](../manifest.json)에 있다. Isaac Sim 환경과 외부 로봇 에셋은 별도로 준비한다.

험지와 추가 학습 모델은 고정한 원본 커밋에 패치를 적용한 checkout을 사용한다.

```bash
git clone https://github.com/Stick-0/isaac-ant-rough-terrain.git
git clone https://github.com/cailab-hy/IsaacLab_RS.git IsaacLab_RS_ant
git -C IsaacLab_RS_ant checkout e83a5d2f11ca1b5f03b690e1978479e620c500e2
python isaac-ant-rough-terrain/scripts/apply_overlay.py IsaacLab_RS_ant --check
python isaac-ant-rough-terrain/scripts/apply_overlay.py IsaacLab_RS_ant
```

Isaac Lab용 Python 환경을 활성화하고 공개 저장소 루트에서 경로를 설정한다.

```bash
export ANT_REPORT_ROOT="$PWD"
export ISAACLAB_ROOT="$(cd ../IsaacLab_RS_ant && pwd)"
```

이미 수정한 checkout에 패치를 다시 적용하지 않는다. `overlay/`는 패치 적용 후의 소스와 동일하다.

## 1. 평지 모델

원본 평지에서 학습·재생할 때는 패치를 적용하지 않은 기반 checkout과 그 checkout에 연결된 Isaac Lab 실행 환경을 사용한다. 패치된 checkout의 `Isaac-Ant-v0`는 험지 태스크다.

```bash
export ISAACLAB_ORIGINAL_ROOT=/path/to/original/IsaacLab_RS

# 저장된 평지 모델 재생
"$ISAACLAB_ORIGINAL_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ORIGINAL_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \
  --task Isaac-Ant-v0 --num_envs 64 \
  --checkpoint "$ANT_REPORT_ROOT/artifacts/models/flat.pt"

# 원본 평지에서 새로 학습
"$ISAACLAB_ORIGINAL_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ORIGINAL_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-Ant-v0 --headless --seed 42 --num_envs 4096 \
  --max_iterations 1000 --run_name flat_seed42
```

패치된 환경에서도 원본 평지 설정을 직접 읽어 영상을 촬영할 수 있다. [촬영 명령](MEDIA.md)을 참고한다.

## 2. 험지 모델

```bash
# 저장된 험지 모델 재생
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \
  --task Isaac-Ant-v0 --num_envs 64 \
  --checkpoint "$ANT_REPORT_ROOT/artifacts/models/rough.pt" --diagnostics

# 랜덤 험지에서 새로 학습
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-Ant-v0 --headless --seed 42 --num_envs 4096 \
  --max_iterations 1000 --run_name rough_seed42
```

평지 모델의 가중치를 가져오는 단계가 아니라 험지에서 별도로 학습하는 단계다.

## 3. 추가 학습 모델

```bash
# 최종 추가 학습 모델 재생
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \
  --task Isaac-Ant-Recovery-v0 --num_envs 64 \
  --checkpoint "$ANT_REPORT_ROOT/artifacts/models/recovery.pt" --diagnostics

# 험지 모델에서 개선한 리워드로 600회 추가 학습
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-Ant-Recovery-v0 --headless --seed 42 --num_envs 4096 \
  --max_iterations 600 --run_name reward_recovery \
  --warm_start "$ANT_REPORT_ROOT/artifacts/models/rough.pt"
```

`--warm_start`는 actor·critic·policy noise 파라미터를 읽고 optimizer와 반복 카운터는 새로 시작한다. 같은 실행을 중단 후 이어가는 `--resume`과 구분한다. 학습률은 1e-4 고정, entropy 계수는 0.001이다.

학습 로그는 실행한 working directory의 `logs/rsl_rl/` 아래에 저장된다. 험지는 `ant/`, 추가 학습은 `ant_recovery/` 폴더를 사용한다. 다시 학습한 결과가 공개 체크포인트와 비트 단위로 같음을 보장하지 않는다.

## 세 모델을 같은 새 지형에서 평가

본문에 공개한 수치는 아래 명령으로 세 모델을 함께 평가한 seed 5001~5005의 결과다. 저장된 모델을 다시 학습할 필요는 없다.

```bash
# 기본값: 5001 5002 5003 5004 5005, 각 512환경, 첫 16초 에피소드
bash scripts/evaluate_models.sh
```

체크포인트는 `flat.pt`, `rough.pt`, `recovery.pt`이며 모델마다 동일 seed로 리셋한다. 실행 스크립트가 초기 상태 해시의 일치를 확인하고 로컬 절대 경로를 저장소 상대 경로로 바꾼다. 원래 공개된 측정은 `results/models/`, 재실행 결과는 기본적으로 `results/reproduced_models/`에 저장된다.

분석 환경에서는 `requirements-analysis.txt`를 설치하고 다음을 실행한다.

```bash
# 새로 재실행한 결과 검증·집계
python scripts/analyze_models.py \
  --results-dir results/reproduced_models \
  --output-dir results/reproduced_models/summary

# 공개된 원자료의 집계와 메인 그래프 재생성
python scripts/analyze_models.py
python scripts/plot_survival.py
```

`plot_survival.py`는 공개된 다섯 시드의 원자료를 사용한다. 한글 글꼴 NanumGothic이 없으면 영문으로 표시한다. 개체별 기록, 모델·평가기 해시, 다섯 시드의 누락 여부를 검사하며 모든 시드가 있어야 최종 집계를 만든다. 평가 기준은 [고정한 프로토콜](../results/models/protocol.json)에 있다.

다른 시드를 추가로 평가하려면 `bash scripts/evaluate_models.sh 6001 6002`처럼 지정할 수 있다. 이 결과는 이번 본문 집계에 섞지 않는다. 과거 평가와 대조 실험을 재현하는 명령은 [기록 모음](archive/README.md)에 있다.

## 공개 파일 검증

```bash
python scripts/verify_package.py
# torch가 있는 환경에서 보상 함수 경계 조건 검사
python scripts/test_reward_terms.py
```

모델·미디어 해시, 원자료와 집계, Python 문법, 공개 경로를 검사한다. 원본 패치는 `apply_overlay.py ... --check`로 확인한다. 코드·원자료·이전 모델의 경로는 문서 정리 후에도 유지했다.
