# FlowMatch vs DDIM — Code-level Differences

이 문서는 `src/lerobot/policies/diffusion/` 안에서 `noise_scheduler_type="DDIM"`과
`noise_scheduler_type="FlowMatch"`가 코드 경로상 어떻게 갈라지는지 정리한다.

대상 파일:

- `src/lerobot/policies/diffusion/configuration_diffusion.py`
- `src/lerobot/policies/diffusion/modeling_diffusion.py`

---

## 1. 큰 그림

| 구분 | DDIM | FlowMatch |
|---|---|---|
| 패러다임 | Denoising Diffusion (epsilon/sample prediction) | Rectified Flow (velocity prediction) |
| 학습 시 t | 정수, `Uniform{0, …, num_train_timesteps-1}` | 실수, `Uniform[0, 1)` |
| Forward 과정 | `scheduler.add_noise` (β 스케줄 기반) | 선형 보간 `x_t = (1-t) x_0 + t x_1` |
| 학습 target | `eps` 또는 `sample` | 속도 `v = x_1 - x_0` |
| 샘플링 | `DDIMScheduler.step` 루프 (t: high→0) | Euler ODE 적분 (t: 0→1) |
| 기본 inference steps | `num_train_timesteps` (예: 100) | **1** |
| 외부 의존성 | `diffusers.DDIMScheduler` | 없음 (직접 구현) |
| 추가 옵션 | — | AdaFlow 조기 종료, RTC 실시간 추론 |

---

## 2. Configuration 차이

`configuration_diffusion.py` 기준.

```python
noise_scheduler_type: str = "DDIM"  # "DDPM" | "DDIM" | "FlowMatch"
num_train_timesteps: int = 100
beta_schedule: str = "squaredcos_cap_v2"
prediction_type: str = "epsilon"     # "epsilon" | "sample"
clip_sample: bool = True
clip_sample_range: float = 1.0
num_inference_steps: int | None = None

# FlowMatch 전용
use_adaflow_inference: bool = False
adaflow_min_steps: int = 1
adaflow_max_steps: int = 4
adaflow_convergence_threshold: float = 0.01
use_rtc: bool = False
```

`__post_init__`에서의 검증 (line 211~221):

- `noise_scheduler_type`은 `{"DDPM", "DDIM", "FlowMatch"}`만 허용.
- `use_adaflow_inference`, `use_rtc`는 **FlowMatch에서만** 사용 가능 (DDIM이면 `ValueError`).

DDIM에서만 의미 있는 필드 (FlowMatch에서는 사실상 무시):

- `beta_start`, `beta_end`, `beta_schedule`
- `prediction_type` (FlowMatch 손실은 항상 velocity)
- `clip_sample`, `clip_sample_range`

---

## 3. 스케줄러 초기화 (`DiffusionModel.__init__`)

`modeling_diffusion.py:197~226`.

### DDIM 경로

```python
self.noise_scheduler = _make_noise_scheduler(
    config.noise_scheduler_type,           # "DDIM"
    num_train_timesteps=...,
    beta_start=..., beta_end=..., beta_schedule=...,
    clip_sample=..., clip_sample_range=...,
    prediction_type=...,
)
self.num_inference_steps = (
    config.num_inference_steps
    if config.num_inference_steps is not None
    else self.noise_scheduler.config.num_train_timesteps   # 기본 = 100
)
self.rtc_processor = None
```

`_make_noise_scheduler`는 `diffusers`의 `DDIMScheduler` 인스턴스를 반환한다
(`modeling_diffusion.py:156~166`).

### FlowMatch 경로

```python
self.noise_scheduler = None    # diffusers scheduler 사용하지 않음
self.num_inference_steps = (
    config.num_inference_steps
    if config.num_inference_steps is not None
    else 1                      # 기본 = 1-step
)
self.rtc_processor = None
if config.use_rtc:
    self.rtc_processor = RTCProcessor(RTCConfig())   # 실시간 chunk 처리용
```

핵심: FlowMatch는 외부 스케줄러 객체 없이 **모델 안에서 직접 Euler 적분**으로 샘플링한다.

---

## 4. 학습 loss (`compute_loss`)

`modeling_diffusion.py:405~499`.

분기점은 line 433:

```python
if self.config.noise_scheduler_type == "FlowMatch":
    return self._compute_loss_flowmatch(batch, global_cond)
# 이하는 DDIM/DDPM 공통 경로
```

### DDIM loss (line 437~474)

```python
trajectory = batch[ACTION]                                    # x_0 (clean)
eps = torch.randn(trajectory.shape, device=trajectory.device)
timesteps = torch.randint(0, num_train_timesteps, (B,)).long()
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

if self.config.prediction_type == "epsilon":
    target = eps
elif self.config.prediction_type == "sample":
    target = trajectory

loss = F.mse_loss(pred, target, reduction="none").mean()
```

특징:

- 정수 timestep을 그대로 UNet에 넣는다.
- Forward noising은 β 스케줄에 따라 결정되는 `add_noise` 사용.
- UNet은 noise(`eps`) 또는 원본(`sample`)을 직접 예측.

### FlowMatch loss (`_compute_loss_flowmatch`, line 476~499)

```python
x_1 = batch[ACTION]                          # clean action
x_0 = torch.randn_like(x_1)                  # noise prior
t   = torch.rand(B, device=..., dtype=...)   # t ~ U[0,1)
t_b = t.view(-1, 1, 1)

x_t       = (1.0 - t_b) * x_0 + t_b * x_1    # 선형 보간
target_v  = x_1 - x_0                        # 속도 (rectified flow)

# UNet의 sinusoidal positional embedding이 정수 timestep 기준으로
# 학습됐기 때문에 t를 동일 스케일로 맞춰서 입력
t_emb_in = t * float(self.config.num_train_timesteps)
pred_v   = self.unet(x_t, t_emb_in, global_cond=global_cond)

loss = F.mse_loss(pred_v, target_v, reduction="none").mean()
```

특징:

- t는 연속값, β 스케줄 없음.
- UNet 출력은 **속도 벡터**로 해석. `prediction_type` 설정은 무시된다.
- `t * num_train_timesteps`로 스케일링하는 이유: UNet의 sinusoidal position
  embedding이 DDIM 학습 때 0~100 범위의 정수 t를 보고 학습됐기 때문에, 같은
  주파수 대역을 활성화시키기 위함.

---

## 5. Sampling (inference)

`modeling_diffusion.py:229~332`.

분기점은 `conditional_sample` 진입부:

```python
def conditional_sample(self, batch_size, global_cond=None, generator=None, noise=None):
    if self.config.noise_scheduler_type == "FlowMatch":
        return self._conditional_sample_flowmatch(batch_size, global_cond, generator, noise)
    # 이하 DDIM
```

### DDIM sampling (line 239~266)

```python
sample = noise or torch.randn((B, horizon, action_dim), ...)
self.noise_scheduler.set_timesteps(self.num_inference_steps)

for t in self.noise_scheduler.timesteps:        # 예: [99, 98, ..., 0] 또는 등간격 부분집합
    model_output = self.unet(
        sample,
        torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
        global_cond=global_cond,
    )
    sample = self.noise_scheduler.step(
        model_output, t, sample, generator=generator
    ).prev_sample                                # x_t → x_{t-1}

return sample
```

특징:

- timestep이 **high → low**로 진행 (노이즈 → 깨끗).
- 한 step의 업데이트 공식은 `DDIMScheduler.step` 안에 캡슐화 (β, α 계산 포함).
- step 수가 곧 UNet forward 횟수. 기본 100 step.

### FlowMatch sampling (`_conditional_sample_flowmatch`, line 268~332)

```python
x_t = noise or torch.randn((B, horizon, action_dim), ...)
n_steps = max(self.num_inference_steps, 1)      # 기본 1
dt = 1.0 / n_steps
train_T = float(self.config.num_train_timesteps)

for step in range(n_steps):
    t_value = step * dt                          # 0, dt, 2dt, ... (t: 0 → 1)
    t_emb = torch.full((B,), t_value * train_T, ...)   # 학습 때와 동일 스케일

    def denoise(input_x_t):
        return self.unet(input_x_t, t_emb, global_cond=global_cond)

    if self.config.use_rtc:
        v_t = self.rtc_processor.denoise_step(...)
    else:
        v_t = denoise(x_t)

    new_x = x_t + dt * v_t                       # Euler step

    # AdaFlow: variance-adaptive 조기 종료
    if self.config.use_adaflow_inference and (step + 1) >= self.config.adaflow_min_steps:
        rel_change = (new_x - x_t).abs().mean() / (x_t.abs().mean() + 1e-8)
        x_t = new_x
        if rel_change.item() < self.config.adaflow_convergence_threshold:
            break
    else:
        x_t = new_x

return x_t
```

특징:

- timestep이 **low → high**로 진행 (노이즈 → 깨끗). DDIM과 정반대 방향.
- 매 step은 단순한 Euler ODE update: `x_{t+dt} = x_t + dt · v(x_t, t)`.
  β 스케줄도, `scheduler.step` 같은 외부 호출도 없음.
- 기본 step 수가 **1**. 한 번의 UNet forward만으로 `x_t = noise + 1.0 * v(noise, 0)`
  결과를 사용. step 수를 늘리면 더 정확한 ODE 적분.
- `use_adaflow_inference=True`이면 `adaflow_min_steps` 이후부터 상대 변화율을
  체크해 수렴 시 break. `adaflow_max_steps`가 상한.
- `use_rtc=True`이면 UNet 호출을 `RTCProcessor.denoise_step`이 감싸 이전 chunk
  의 잔여분을 활용한 실시간 추론을 수행.

---

## 6. 한눈에 보는 수식 차이

| 항목 | DDIM | FlowMatch |
|---|---|---|
| Forward noising | $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ | $x_t = (1-t)\, x_0 + t\, x_1$ |
| Model output 의미 | $\hat\epsilon$ 또는 $\hat x_0$ | 속도 $\hat v$ |
| Loss target | $\epsilon$ 또는 $x_0$ | $x_1 - x_0$ |
| Sampling update | $x_{t-1} = \text{DDIMStep}(x_t, \hat\epsilon, t)$ | $x_{t+dt} = x_t + dt\cdot\hat v$ |
| t 도메인 | 이산, $\{0,\dots,T-1\}$, 역방향 | 연속, $[0,1]$, 정방향 |

---

## 7. 호환되는 옵션 / 호환 안 되는 옵션

`configuration_diffusion.py:217~221`:

```python
if self.noise_scheduler_type != "FlowMatch":
    if self.use_adaflow_inference:
        raise ValueError("`use_adaflow_inference` requires `noise_scheduler_type='FlowMatch'`.")
    if self.use_rtc:
        raise ValueError("`use_rtc` requires `noise_scheduler_type='FlowMatch'`.")
```

요약:

- AdaFlow, RTC는 **FlowMatch 전용**.
- DDIM/DDPM은 β 스케줄, `prediction_type`, sample clipping 옵션을 사용한다 —
  FlowMatch는 이 필드들을 읽지 않는다.

---

## 8. UNet 자체는 동일

두 경로 모두 동일한 `DiffusionConditionalUnet1d`(`modeling_diffusion.py:190`)를
공유한다. 차이는 어디까지나:

1. UNet에 어떤 t와 어떤 입력 x를 넣을지,
2. UNet 출력을 어떻게 해석할지 (noise vs velocity),
3. 한 step의 x 업데이트 공식.

따라서 같은 backbone(예: dinov2/resnet)과 같은 conditioning 코드를 그대로
재사용하면서 학습/샘플링 방식만 바꿔 끼울 수 있다.
