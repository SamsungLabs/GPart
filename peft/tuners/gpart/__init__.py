# Copyright 2026 Samsung
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
from peft.utils import register_peft_method

from .config import GPartConfig
from .grouping import generate_implicit_group_ids
from .layer import Linear, GPartLayer
from .model import GPartModel

__all__ = [
    "GPartConfig",
    "GPartLayer",
    "Linear",
    "GPartModel",
    "generate_implicit_group_ids",
]
register_peft_method(name="gpart", config_cls=GPartConfig, model_cls=GPartModel)
