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

import errno
import os
import paddle
import time
import glob

from passl.utils import logger

__all__ = ['load_checkpoint', 'save_checkpoint']

def _mkdir_if_not_exist(path):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))

def load_checkpoint(config, net, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    checkpoint = config.get('checkpoint')
    if checkpoint and optimizer is not None:
        assert os.path.exists(checkpoint + ".pdparams"), \
            "Given dir {}.pdparams not exist.".format(checkpoint)
        assert os.path.exists(checkpoint + ".pdopt"), \
            "Given dir {}.pdopt not exist.".format(checkpoint)
        para_dict = paddle.load(checkpoint + ".pdparams")
        opti_dict = paddle.load(checkpoint + ".pdopt")
        metric_dict = paddle.load(checkpoint + ".pdstates")
        net.set_dict(para_dict)
        optimizer.set_state_dict(opti_dict)
        logger.info("Finish load checkpoint from {}".format(checkpoint))
        return metric_dict

def save_checkpoint(net,
               optimizer,
               metric_info,
               model_path,
               model_name="",
               prefix='passl',
               max_num_checkpoint=3):
    """
    save model to the target path
    """
    if paddle.distributed.get_rank() != 0:
        return
    
    metric_info.update(
        {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}
    )
    model_dir = os.path.join(model_path, model_name)
    _mkdir_if_not_exist(model_dir)
    model_prefix = os.path.join(model_dir, prefix)

    paddle.save(net.state_dict(), model_prefix + ".pdparams")
    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    paddle.save(metric_info, model_prefix + ".pdstates")
    logger.info("Already save {} model in {}".format(prefix, model_dir))
    
    keep_prefixs = ['best', 'latest']

    if all(p not in prefix for p in keep_prefixs) and max_num_checkpoint >= 0:
        pdstates_list = glob.glob(os.path.join(model_dir, '*.pdstates'))
        
        timestamp_to_path = {}
        for path in pdstates_list:
            if any(p in path for p in keep_prefixs):
                continue
            metric_dict = paddle.load(path)
            timestamp_to_path[metric_dict['timestamp']] = path[:-9]
        
        # sort by ascend
        timestamps = list(timestamp_to_path.keys())
        timestamps.sort()
        
        if max_num_checkpoint > 0:
            to_remove = timestamps[:-max_num_checkpoint]
        else:
            to_remove = timestamps
        for timestamp in to_remove:
            model_prefix = timestamp_to_path[timestamp]
            for ext in ['.pdparams', '.pdopt', '.pdstates']:
                path = model_prefix + ext
                if os.path.exists(path):
                    logger.info("Remove checkpoint {}.".format(path))
                    os.remove(path)
