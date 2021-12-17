# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import paddle

from passl.utils import logger

from . adamw import AdamW
from . adafactor import Adafactor

def build_optimizer(config, lr_scheduler, model_list=None):
    config = copy.deepcopy(config)
    
    # step1 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        if 'weight_decay' in config:
            logger.warning(
                "ConfigError: Only one of regularizer and weight_decay can be set in Optimizer Config. \"weight_decay\" has been ignored."
            )
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name') + 'Decay'
        reg = getattr(paddle.regularizer, reg_name)(**reg_config)
        config["weight_decay"] = reg
        logger.debug("build regularizer ({}) success..".format(reg))
        
    # step2 build optimizer
    optim_name = config.pop('name')
    if 'clip_norm' in config:
        clip_norm = config.pop('clip_norm')
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    elif 'clip_global_norm' in config:
        clip_global_norm = config.pop('clip_global_norm')
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_global_norm)
    else:
        grad_clip = None
    optim = eval(optim_name)(learning_rate=lr_scheduler,
                                           grad_clip=grad_clip,
                                           **config)(model_list=model_list)
    logger.debug("build optimizer ({}) success..".format(optim))
    return optim