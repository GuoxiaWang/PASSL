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

import os
import random
import platform
import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from visualdl import LogWriter

from passl.utils.misc import AverageMeter
from passl.utils import logger
from passl.utils.config import print_config
from passl.data import build_dataloader
from passl.models import build_model
from passl.loss import build_loss
from passl.metric import build_metrics
from passl.scheduler import build_lr_scheduler
from passl.optimizer import build_optimizer
from passl.utils import io

from . import classification


class Engine(object):
    def __init__(self, config, mode="train"):
        assert mode in ["train", "eval"]
        self.mode = mode
        self.config = config
        self.task_type = self.config["Global"].get("task_type", "classification")
        self.use_dali = self.config['Global'].get("use_dali", False)
        self.print_batch_step = self.config['Global'].get('print_batch_step', 10)
        self.save_interval = self.config["Global"].get("save_interval", 1)
        self.accum_steps = self.config["Global"].get("accum_steps", 1)
        
        # init distribution env
        self.config["Global"]["distributed"] = dist.get_world_size() != 1
        self.config["Global"]["rank"] = dist.get_rank()
        self.config["Global"]["world_size"] = dist.get_world_size()
        if self.config["Global"]["distributed"]:
            dist.fleet.init(is_collective=True)

        # set seed
        seed = self.config["Global"].get("seed", False)
        if seed:
            assert isinstance(seed, int), "The 'seed' must be a integer!"
            seed += self.config["Global"]["rank"]
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # init logger
        self.output_dir = self.config['Global']['output_dir']
        log_file = os.path.join(self.output_dir, self.config["Model"]["name"],
                                f"{mode}.log")
        logger.init_logger(name='root', log_file=log_file)
        print_config(config)

        # init train_func and eval_func
        train_epoch_func_name = self.config['Global'].get("train_epoch_func", 'defualt_train_one_epoch')
        self.train_epoch_func = getattr(eval('{}'.format(self.task_type)), train_epoch_func_name)
        
        eval_func_name = self.config['Global'].get("eval_func", 'default_classification_eval')
        self.eval_func = getattr(eval('{}'.format(self.task_type)), eval_func_name)

        # for visualdl
        self.vdl_writer = None
        if self.config['Global']['use_visualdl'] and mode == "train":
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)

        # set device
        assert self.config["Global"]["device"] in ["cpu", "gpu", "xpu", "npu"]
        self.device = paddle.set_device(self.config["Global"]["device"])
        logger.info('train with paddle {} and device {}'.format(
            paddle.__version__, self.device))

        class_num = config["Model"].get("class_num", None)
        self.config["DataLoader"].update({"class_num": class_num})
        # build dataloader
        if self.mode == 'train':
            self.train_dataloader = build_dataloader(
                self.config["DataLoader"], "Train", self.device, self.use_dali)
        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            self.eval_dataloader = build_dataloader(
                self.config["DataLoader"], "Eval", self.device, self.use_dali)

        # build loss
        if self.mode == "train":
            loss_info = self.config["Loss"]["Train"]
            self.train_loss_func = build_loss(loss_info)
        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            loss_config = self.config.get("Loss", None)
            if loss_config is not None:
                loss_config = loss_config.get("Eval")
                if loss_config is not None:
                    self.eval_loss_func = build_loss(loss_config)
                else:
                    self.eval_loss_func = None
            else:
                self.eval_loss_func = None

        # build metric
        if self.mode == 'train':
            metric_config = self.config.get("Metric")
            if metric_config is not None:
                metric_config = metric_config.get("Train")
                if metric_config is not None:
                    self.train_metric_func = build_metrics(metric_config)
                else:
                    self.train_metric_func = None
        else:
            self.train_metric_func = None

        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            metric_config = self.config.get("Metric")
            if metric_config is not None:
                metric_config = metric_config.get("Eval")
                if metric_config is not None:
                    self.eval_metric_func = build_metrics(metric_config)
        else:
            self.eval_metric_func = None

        # build model
        self.model = build_model(self.config["Model"])

        # load pretrained model
        if self.config["Global"]["pretrained_model"] is not None:
            assert isinstance(self.config["Global"]["pretrained_model"], str), "pretrained_model type is not available. Please use `string`."
            self.model.load_pretrained(self.config["Global"]["pretrained_model"],
                                       self.config["Global"].get("finetune", False))

        # build optimizer
        if self.mode == 'train':
            self.lr_decay_unit = self.config["LRScheduler"].get('decay_unit', 'step')
            self.lr_scheduler = build_lr_scheduler(self.config["LRScheduler"],
                self.config["Global"]["epochs"], len(self.train_dataloader))
            
            self.optimizer = build_optimizer(
                self.config["Optimizer"], self.lr_scheduler, [self.model])
            
        # FP16 training
        self.fp16 = True if "FP16" in self.config else False
        config_fp16 = self.config.get('FP16', {})
        assert config_fp16 is not None
        self.fp16_level = config_fp16.get("level", 'O2')
        self.fp16_custom_white_list = config_fp16.get("fp16_custom_white_list", None)
        self.fp16_custom_black_list = config_fp16.get("fp16_custom_black_list", None)
        
        if self.fp16:
            config_gradscaler = config_fp16.get('GradScaler', {})
            self.scaler = paddle.amp.GradScaler(
                enable=True,
                **config_gradscaler,
            )
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.fp16_level,
                master_weight=config_fp16.get("master_weight", True),
                save_dtype=config_fp16.get("save_dtype", 'float32'),
            )
        else:
            self.scaler = paddle.amp.GradScaler(enable=False)

        # for distributed
        if self.config["Global"]["distributed"]:
            # config DistributedStrategy
            if self.config.get("DistributedStrategy", None) is not None:
                if self.config["DistributedStrategy"].get("data_sharding", False):
                    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import ShardingOptimizerStage2
                    from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage2 import ShardingStage2
                    hcg = dist.fleet.get_hybrid_communicate_group()
                    # group = hcg.get_check_parallel_group()
                    group = paddle.distributed.new_group([0, 1, 2, 3])
                    
                    # First, we need to split optimizer
                    self.optimizer = ShardingOptimizerStage2(
                        params=self.model.parameters(), optim=self.optimizer, group=group)

                    # Second, warpper the origin model to have gradient sharding function
                    self.model = ShardingStage2(
                        self.model, 
                        self.optimizer, 
                        group=group, 
                        accumulate_grads=self.accum_steps>1,
                        device=self.config["Global"]["device"],
                    )
            else:
                # we always use pure data parallel default
                self.model = paddle.DataParallel(self.model)      

    def train(self):
        assert self.mode == "train"
        best_metric = {
            "metric": 0.0,
            "epoch": 0,
        }
        # key:
        # val: metrics list word
        self.output_info = dict()
        self.time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }
        # global iter counter
        self.global_step = 0

        if self.config["Global"]["checkpoint"] is not None:
            metric_info = io.load_checkpoint(self.config["Global"], self.model,
                                     self.optimizer)
            if metric_info is not None:
                best_metric.update(metric_info)

        self.max_iter = len(self.train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.train_dataloader)
        for epoch_id in range(best_metric["epoch"] + 1,
                              self.config["Global"]["epochs"] + 1):
            acc = 0.0
            # for one epoch train
            self.train_epoch_func(self, epoch_id)

            if self.use_dali:
                self.train_dataloader.reset()
            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, self.output_info[key].avg)
                for key in self.output_info
            ])
            logger.info("[Train][Epoch {}/{}][Avg]{}".format(
                epoch_id, self.config["Global"]["epochs"], metric_msg))
            self.output_info.clear()

            # eval model and save model if possible
            if self.config["Global"][
                    "eval_during_train"] and epoch_id % self.config["Global"][
                        "eval_interval"] == 0:
                acc = self.eval(epoch_id)
                if acc > best_metric["metric"]:
                    best_metric["metric"] = acc
                    best_metric["epoch"] = epoch_id
                    io.save_checkpoint(
                        self.model,
                        self.optimizer,
                        best_metric,
                        self.output_dir,
                        model_name=self.config["Model"]["name"],
                        prefix="best_model",
                        max_num_checkpoint=self.config["Global"]["max_num_latest_checkpoint"],
                    )
                logger.info("[Eval][Epoch {}][best metric: {}]".format(
                    epoch_id, best_metric["metric"]))
                logger.scaler(
                    name="eval_acc",
                    value=acc,
                    step=epoch_id,
                    writer=self.vdl_writer)

            # save model
            if epoch_id % self.save_interval == 0:
                if self.config["Global"]["max_num_latest_checkpoint"] != 0:
                    io.save_checkpoint(
                        self.model,
                        self.optimizer, {"metric": acc,
                                         "epoch": epoch_id},
                        self.output_dir,
                        model_name=self.config["Model"]["name"],
                        prefix="epoch_{}".format(epoch_id),
                        max_num_checkpoint=self.config["Global"]["max_num_latest_checkpoint"],
                    )
                # save the latest model
                io.save_checkpoint(
                    self.model,
                    self.optimizer, {"metric": acc,
                                     "epoch": epoch_id},
                    self.output_dir,
                    model_name=self.config["Model"]["name"],
                    prefix="latest",
                    max_num_checkpoint=self.config["Global"]["max_num_latest_checkpoint"],
                )

        if self.vdl_writer is not None:
            self.vdl_writer.close()

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        assert self.mode in ["train", "eval"]
        self.model.eval()
        eval_result = self.eval_func(self, epoch_id)
        self.model.train()
        return eval_result
