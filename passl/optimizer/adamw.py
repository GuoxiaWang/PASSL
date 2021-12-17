# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import optimizer as optim

from passl.utils import logger

class AdamW(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=None,
                 multi_precision=False,
                 grad_clip=None,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False,
                 **args):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.multi_precision = multi_precision
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None

        # TODO(gaotingquan): model_list is None when in static graph, "no_weight_decay" not work.
        if model_list is None:
            if self.one_dim_param_no_weight_decay or len(
                    self.no_weight_decay_name_list) != 0:
                msg = "\"AdamW\" does not support setting \"no_weight_decay\" in static graph. Please use dynamic graph."
                logger.error(Exception(msg))
                raise Exception(msg)

        self.no_weight_decay_param_name_list = [
            p.name for model in model_list for n, p in model.named_parameters()
            if any(nd in n for nd in self.no_weight_decay_name_list)
        ] if model_list else []

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += [
                p.name for model in model_list
                for n, p in model.named_parameters() if len(p.shape) == 1
            ] if model_list else []

        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            multi_precision=self.multi_precision,
            grad_clip=self.grad_clip,
            apply_decay_param_fun=self._apply_decay_param_fun)
        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list
