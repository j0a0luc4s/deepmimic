# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
from os import path

import numpy as np
import json

ROOT_PATH = path.dirname(__file__)

T1_MJP_JOYSTICK_PATH = path.join(ROOT_PATH, "t1_mjp_joystick.json")


def load_ref(ref_name: str):
    ref_path = {
        "t1_mjp_joystick": T1_MJP_JOYSTICK_PATH
    }[ref_name]

    ref = {}

    with open(ref_path) as ref_file:
        ref_raw = json.load(ref_file)

        ref["qposs"] = np.array(ref_raw["qposs"])
        ref["qvels"] = np.array(ref_raw["qvels"])
        ref["subtree_coms"] = np.array(ref_raw["subtree_coms"])
        ref["xquats"] = np.array(ref_raw["xquats"])
        ref["cvels"] = np.array(ref_raw["cvels"])

    return ref
