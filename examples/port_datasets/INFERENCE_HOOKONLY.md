# Hookonly inference (ACT / Diffusion)

학습된 체크포인트를 변환된 `*_hookonly` LeRobot 데이터셋에 적용해
예측 action을 GT(ground-truth) action과 비교하는 스크립트 사용 가이드.

스크립트:
- `examples/port_datasets/inference_act_hookonly.py` — ACT용, 100-step chunk를 한 번에 예측
- `examples/port_datasets/inference_diffusion_hookonly.py` — Diffusion용, `select_action` 루프로 `n_action_steps`(기본 8)만큼 예측

## 동작 개요

### ACT
1. `ACTPolicy.from_pretrained(<체크포인트>)`로 정책 로드
2. `make_pre_post_processors(cfg, pretrained_path=<체크포인트>)`로 학습 시 저장된
   normalizer/unnormalizer 로드
3. 데이터셋에서 `--frame-index` 프레임을 한 개 꺼내 batch 차원 추가 후 preprocess
4. `policy.predict_action_chunk(batch)` 로 `chunk_size`(기본 100) 길이의 action 예측
5. postprocess(unnormalize) 후 같은 episode의 다음 N개 GT action과 비교

### Diffusion
1. `DiffusionPolicy.from_pretrained(<체크포인트>)` + 동일 processor 로드
2. `policy.reset()`으로 observation/action queue 초기화
3. `--frame-index`부터 `n_action_steps`(=8)개의 연속 프레임을 하나씩 `select_action`으로
   호출 — 내부적으로 `n_obs_steps=2` 크기의 obs queue를 유지하며 `horizon=16` chunk를 뽑아
   앞 `n_action_steps`개만 실행
4. postprocess 후 해당 8개 GT action과 비교

## 사용법

```bash
conda activate lerobot
cd /home/ngseo/remove_hook/lerobot

# --- ACT ---
# dg5f (state 163D, action 26D, chunk 100)
CUDA_VISIBLE_DEVICES=1 python examples/port_datasets/inference_act_hookonly.py \
  --checkpoint outputs/train/dg5f_act/checkpoints/last/pretrained_model \
  --dataset-root /home/ngseo/recorded_dataset/lerobot/dg5f_hookonly \
  --frame-index 100 \
  --show-first-n 5

# rh56f1 (state 141D, action 12D, chunk 100)
CUDA_VISIBLE_DEVICES=1 python examples/port_datasets/inference_act_hookonly.py \
  --checkpoint outputs/train/rh56f1_act/checkpoints/last/pretrained_model \
  --dataset-root /home/ngseo/recorded_dataset/lerobot/rh56f1_hookonly \
  --frame-index 100

# --- Diffusion ---
# dg5f (n_obs_steps=2, horizon=16, n_action_steps=8)
CUDA_VISIBLE_DEVICES=1 python examples/port_datasets/inference_diffusion_hookonly.py \
  --checkpoint outputs/train/dg5f_diffusion/checkpoints/last/pretrained_model \
  --dataset-root /home/ngseo/recorded_dataset/lerobot/dg5f_hookonly \
  --frame-index 100

# rh56f1
CUDA_VISIBLE_DEVICES=1 python examples/port_datasets/inference_diffusion_hookonly.py \
  --checkpoint outputs/train/rh56f1_diffusion/checkpoints/last/pretrained_model \
  --dataset-root /home/ngseo/recorded_dataset/lerobot/rh56f1_hookonly \
  --frame-index 100
```

## 옵션 (공통)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--checkpoint` | (필수) | `.../checkpoints/<step>/pretrained_model` 경로. `last` 심볼릭 링크 사용 가능 |
| `--dataset-root` | (필수) | `LeRobotDataset` 루트 (변환 결과 폴더) |
| `--repo-id` | `local/<루트 폴더명>` | 메타데이터용 식별자, 보통 자동 사용 |
| `--frame-index` | `0` | 입력으로 사용할 프레임 인덱스 (전역 인덱스) |
| `--show-first-n` | `10` | GT vs 예측을 출력할 첫 N개 step |
| `--device` | `cuda` (없으면 `cpu`) | 정책/preprocessor 실행 디바이스 |

Diffusion 전용:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--num-steps` | `cfg.n_action_steps` (=8) | `select_action`으로 뽑을 연속 step 수. episode 끝에 도달하면 잘림 |

## 출력 해석

```
loaded ACT  chunk_size=100  device=cuda  ckpt=...
frame 100  episode 0 (0..761)  obs.state=(163,)  img=(3, 240, 320)

pred chunk : shape=(100, 26)  mean|err|=0.0050  max|err|=0.1430

first 3 action steps  (left: GT, right: pred)
t=  0  gt=[ ... 26 floats ... ]
       pr=[ ... 26 floats ... ]
...
```

- `episode 0 (0..761)`: 해당 프레임이 속한 episode 번호와 데이터셋 내 [from..to) 인덱스 범위
- `mean|err|`, `max|err|`: 예측 chunk 전체에 대한 절대 오차
- `gt` / `pr`: 같은 timestep의 GT / 예측 action 벡터

## 검증 결과 (참고)

| 정책 | 데이터셋 | frame | 비교 길이 | mean\|err\| | max\|err\| |
|---|---|---|---|---|---|
| ACT       | dg5f_hookonly | 100 | 100 step chunk | 0.0050 | 0.1430 |
| Diffusion | dg5f_hookonly | 100 | 8 step rollout | 0.0039 | 0.0554 |

## 주의사항

- **GPU 메모리**: Diffusion 학습이 동시 진행 중이면 GPU 0/1 모두 점유 상태. 더 여유 있는 쪽을
  `CUDA_VISIBLE_DEVICES=`로 골라 사용하세요. ACT 모델 자체는 ~52M, 추가 ~1GB 정도 사용.
- **CPU 모드 미지원 이유**: preprocessor 안의 normalizer stats가 학습 시 저장된 device(cuda)
  에 묶여 있어 `--device cpu`만으로는 device mismatch가 발생합니다. 현재 스크립트는 cuda 사용
  전제. 꼭 CPU에서 돌리려면 preprocessor의 모든 텐서를 명시적으로 `.to("cpu")` 해주는 추가
  처리가 필요합니다.
- **chunk_size / rollout 길이**: ACT는 `chunk_size=100`이라 episode 끝 근처에서는 GT가 부족합니다.
  스크립트는 마지막 GT를 복제해 길이를 맞춥니다(평가용 참고치). Diffusion 스크립트는 episode
  끝에 도달하면 rollout을 줄여 잘라냅니다.
- **Diffusion의 open-loop vs closed-loop**: 실 로봇 제어에서는 `n_action_steps`(=8) 개의
  action을 실행한 뒤 다시 새로운 observation으로 `select_action`을 호출하는 닫힌 고리 방식을
  쓰세요. 본 스크립트는 **매 step마다** 새 observation을 주며 `select_action`을 호출하므로
  queue 덕분에 자연스럽게 8 step마다 새 chunk가 생성되는 closed-loop 시나리오를 모사합니다.
- **ACT vs Diffusion action 의미**: 둘 다 학습 시 `actions`(raw teleop) 필드를 예측합니다.
  실행 시 Isaac Lab의 scale/offset (`processed_actions = raw * scale + offset`)이 env 쪽에서
  추가 적용된다는 점 주의.
