# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_LAZY_IMPORTS = {
    "ACTConfig": "lerobot.policies.act.configuration_act",
    "DiffusionConfig": "lerobot.policies.diffusion.configuration_diffusion",
    "GrootConfig": "lerobot.policies.groot.configuration_groot",
    "PI0Config": "lerobot.policies.pi0.configuration_pi0",
    "PI0FastConfig": "lerobot.policies.pi0_fast.configuration_pi0_fast",
    "PI05Config": "lerobot.policies.pi05.configuration_pi05",
    "SARMConfig": "lerobot.policies.sarm.configuration_sarm",
    "SmolVLAConfig": "lerobot.policies.smolvla.configuration_smolvla",
    "SmolVLANewLineProcessor": "lerobot.policies.smolvla.processor_smolvla",
    "TDMPCConfig": "lerobot.policies.tdmpc.configuration_tdmpc",
    "VQBeTConfig": "lerobot.policies.vqbet.configuration_vqbet",
    "WallXConfig": "lerobot.policies.wall_x.configuration_wall_x",
    "XVLAConfig": "lerobot.policies.xvla.configuration_xvla",
    "get_policy_class": "lerobot.policies.factory",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_LAZY_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
