# Hyundai Uiwang — 데이터 변환 & FlowMatch 학습

현대차 의왕 양팔(left/right) 매니퓰레이션 데이터를 LeRobot v3.0 데이터셋으로 변환하고,
Diffusion Policy의 **FlowMatch** 스케줄러로 학습하는 파이프라인 정리.

## 1. 데이터

| | 원본 | 변환본 (LeRobot v3.0) |
|---|---|---|
| 위치 | `~/hyundai_uiwang_data/{left,right}/datas` | `data/lerobot/hyundai_uiwang_{left,right}` |
| 형식 | `system/frame_data_{i}.bin` + `videos/{side}_{i}_{wrist_cam,zivid}_video.mp4` | parquet + h264 영상 + meta |
| left | 131 episodes | 131 ep / 110,568 frames |
| right | 118 episodes | 118 ep |

- **주파수: 30 Hz** (state·action·이미지 전부 동기)
- `observation.state` / `action`: 26차원 (arm joint 6 + hand joint 20)
- `action` = `target_robot_joint`(6) + `target_gripper_joint`(20)
- 추가 센서: `observation.gripper_sensor`[30], `observation.wrist_ft_sensor`[6]
  → `FeatureType.STATE`로 매핑되지만 Diffusion Policy는 `observation.state`만 입력에 사용 (나머지는 normalize만 됨, 무해)
- 카메라 2개 (둘 다 640×480으로 리사이즈):
  - `observation.images.front_rgb` ← zivid (원본 1944×1200)
  - `observation.images.wrist_rgb` ← wrist_cam (원본 848×480)

### 왜 재변환했나
팀의 `~/hyundai_uiwang_data/convert_data_aligned_video.py`는 parquet + info.json만 만들고
**영상 배치와 v3.0 메타(`meta/episodes`, `meta/tasks.parquet`, `meta/stats.json`)를 생성하지 않아** LeRobot으로 로드 불가였다.
또 info.json의 영상 스펙(h264·50fps·480×640)이 실제(mpeg4·30fps·카메라별 상이)와 불일치였다.
→ 공식 API(`LeRobotDataset.create/add_frame/save_episode`)로 재생성하여 영상 인코딩·메타를 한 번에 올바르게 만들었다.
bin 파싱과 영상-타임라인 정합 로직은 팀 변환기를 그대로 재사용한다.

## 2. 변환 실행

```bash
export PYTHONPATH=src
# 10워커 분할 변환 → aggregate 병합 (단일 프로세스는 1코어만 써서 ~2h, 병렬은 ~10min/side)
python3 scripts/convert_hyundai_uiwang_parallel.py --side left  --workers 10
python3 scripts/convert_hyundai_uiwang_parallel.py --side right --workers 10
```

- 변환기: `scripts/convert_hyundai_uiwang_to_lerobot.py` (`--side`, `--episodes`, `--root`, `--repo-id`, `--vcodec`, `--limit`)
- 병렬 드라이버: `scripts/convert_hyundai_uiwang_parallel.py` (에피소드 샤딩 → `aggregate_datasets`로 재인코딩 없이 병합)
- 검증: `LeRobotDataset('local/hyundai_uiwang_left', root='data/lerobot/hyundai_uiwang_left')` 로드 → 영상 `(3,480,640)` 디코딩 확인

## 3. FlowMatch 학습

`flowmatch policy`는 별도 정책이 아니라 **Diffusion Policy의 noise scheduler 옵션**이다:

```
--policy.type=diffusion --policy.noise_scheduler_type=FlowMatch --policy.num_inference_steps=1
```

| 스크립트 | 용도 |
|---|---|
| `scripts/run_train_uiwang_left_flowmatch_smoke.sh [STEPS]` | 파이프라인 검증 (기본 200 step, resnet18, 풀해상도) |
| `scripts/run_train_uiwang_left_flowmatch_full.sh [STEPS]` | 본 학습 (기본 200k step, resize+crop) |

- smoke 검증 완료: loss 1.41 → 1.00, ~3.7 step/s (RTX PRO 4500), 체크포인트 정상 저장
- 출력: `outputs/train/<job_name>/checkpoints/`

## 4. 환경

LeRobot이 pip 설치돼 있지 않아 **base conda python(`~/miniconda3`)에서 `PYTHONPATH=src`로 소스 실행**한다.
base에 추가 설치한 의존성:
`accelerate av opencv-python-headless termcolor wandb draccus pyserial deepdiff jsonlines imageio imageio-ffmpeg einops gymnasium diffusers`

- GPU: NVIDIA RTX PRO 4500 Blackwell (30GB)
- 영상 디코딩 백엔드: torchcodec 없음 → pyav fallback (정상 동작)
