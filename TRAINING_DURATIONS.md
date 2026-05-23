# 학습 시간 정리

`outputs/train/_logs/*.log`의 첫/마지막 INFO 타임스탬프, `outputs/train/_logs/chain_*.log`의 launch/finish 라인, 그리고 각 모델 체크포인트의 `train_config.json` (`steps`, `batch_size`)을 기준으로 산정.

같은 시각에 시작한 모델은 서로 다른 GPU에서 병렬로 돌린 것 (chain log 확인).
GPU는 모두 RTX 5000 Ada (32GB) × 2.

## dg5f

| 모델 | 백본 | 스케줄러 | cams | bs | steps | 시작 → 종료 | 소요 |
|---|---|---|---|---|---|---|---|
| dg5f_act_dinov2s | dinov2-small | — (ACT) | 1 | 8 | 800k | 04-29 18:13 → 04-30 01:40 | **7h 27m** |
| dg5f_diffusion_dinov2s | dinov2-small | DDIM | 1 | 64 | 200k | 04-29 18:09 → 04-30 02:45 | **8h 35m** |
| dg5f_act_dinov2s_dr | dinov2-small | — (ACT) | 1 | 8 | 800k | 04-30 03:26 → 04-30 12:04 | **8h 37m** |
| dg5f_diffusion_dinov2s_dr | dinov2-small | DDIM | 1 | 64 | 200k | 04-30 03:26 → 04-30 17:06 | **13h 39m** |
| dg5f_diffusion_dinov2s_flowmatch | dinov2-small | FlowMatch | 1 | 64 | 200k | 04-30 14:28 → 05-01 03:56 | **13h 28m** |
| dg5f_act_dinov2s_multicam | dinov2-small | — (ACT) | 2 | 8 | 800k | 04-30 21:27 → 05-01 09:32 | **12h 04m** |
| dg5f_diffusion_dinov2s_multicam | dinov2-small | DDIM | 2 | 64 | 200k | 05-01 17:14 → 05-02 18:12 | **24h 58m** |
| dg5f_diffusion_dinov2s_flowmatch_multicam | dinov2-small | FlowMatch | 2 | 64 | 200k | 05-01 17:14 → 05-02 18:20 | **25h 06m** |
| dg5f_diffusion_flowmatch_multicam | **resnet18** | FlowMatch | 2 | 64 | 200k | 05-13 15:30 → 진행 중 | **~27h 예상** (54k/200k, ETA 21h 잔여) |

## rh56f1

| 모델 | 백본 | 스케줄러 | cams | bs | steps | 시작 → 종료 | 소요 |
|---|---|---|---|---|---|---|---|
| rh56f1_act_dinov2s_dr | dinov2-small | — (ACT) | 1 | 8 | 800k | 05-04 00:43 → 05-04 08:56 | **8h 12m** |
| rh56f1_diffusion_dinov2s_dr | dinov2-small | DDIM | 1 | 64 | 200k | 05-04 00:43 → 05-04 14:29 | **13h 45m** |
| rh56f1_diffusion_dinov2s_flowmatch | dinov2-small | FlowMatch | 1 | 64 | 200k | 05-04 08:56 → 05-04 22:39 | **13h 43m** |
| rh56f1_act_dinov2s_multicam | dinov2-small | — (ACT) | 2 | 8 | 800k | 05-04 14:29 → 05-05 02:56 | **12h 26m** |
| rh56f1_diffusion_dinov2s_flowmatch_multicam | dinov2-small | FlowMatch | 2 | 64 | 200k | 05-04 22:39 → 05-05 23:29 | **24h 49m** |
| rh56f1_diffusion_dinov2s_multicam | dinov2-small | DDIM | 2 | 64 | 200k | 05-05 02:56 → 05-06 03:17 | **24h 20m** |

## 관찰

- **ACT vs Diffusion (1cam, 동일 데이터셋)**: ACT(bs=8, 800k step)와 Diffusion(bs=64, 200k step)이 비슷한 시간대 (7~9h). 두 설정 모두 약 1.6M sample을 본 셈.
- **Multicam은 거의 정확히 ×2**: 1cam → 2cam으로 가면 diffusion 기준 ~13.5h → ~25h, ACT 기준 ~8h → ~12h. 이미지 인코더 forward가 카메라 수에 선형.
- **FlowMatch ≈ DDIM (학습 시간)**: 학습 루프는 DDIM이나 FlowMatch나 UNet forward 1회로 동일하기 때문에 차이가 거의 없다 (12분 차이 수준). FlowMatch의 장점은 **추론 step 수**에서 나옴 (1-step vs 100-step).
- **DR (image_transforms) 비용**: dr 없는 dg5f_diffusion_dinov2s (8h 35m) vs dr 켠 dg5f_diffusion_dinov2s_dr (13h 39m). DR이 ~60% 추가 비용.
- **resnet18 vs dinov2-small (multicam flowmatch)**: 진행 중인 resnet 버전은 ~2.0 step/s로 dinov2 버전(~2.4 step/s)보다 약간 느림. dinov2-small은 frozen이고 resnet18은 학습되기 때문 (학습 파라미터 283M).
