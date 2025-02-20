#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Test training benchmark for a model.
# Usage: bash ./tests/test_tipc/classification/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${bs_item} ${run_mode} ${device_num} ${sample_ratio} ${yaml_path} ${epochs} 2>&1;
function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item
    fp_item=${2:-"fp32"}            # (必选) fp32|fp16
    global_batch_size=${3:-"128"}    # （必选）global_batch_size
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP2-MP8-PP2|DP1-MP8-PP4|DP4-MP8-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    yaml_path=${6:-"./tasks/classification/vit/configs/ViT_base_patch16_224_in1k_1n8c_dp_fp16o2.yaml"}
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PASSL"          # (必选) 模型套件的名字
    speed_unit="images/sec"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=${7:-150}                      # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件；或使用max_epoch参数
    num_workers=0                  # (可选)
    base_batch_size=$global_batch_size
    accum_steps=${8:-1}
    pretrained_model=${9:-null}
    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${global_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    #
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed

    OUTPUT_PATH=${run_log_path}/output
}

function _train(){
    batch_size=${local_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs

    if [ -d $OUTPUT_PATH ]; then
        rm -rf $OUTPUT_PATH
    fi
    mkdir $OUTPUT_PATH

    echo "current model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    if [[ ${model_item} =~ "ft" ]];then # finetune
        pretrained=" -o Global.pretrained_model=${pretrained_model} "
    else
        pretrained=""
    fi

    train_cmd="-o Global.print_batch_step=1 \
               -o Global.max_train_step=${max_iter} \
               -o Global.flags.FLAGS_cudnn_exhaustive_search=0 \
               -o Global.flags.FLAGS_cudnn_deterministic=1 \
               -o Global.accum_steps=${accum_steps} \
               ${pretrained} "
    if [ ${PADDLE_TRAINER_ID} ]
    then
        PADDLE_RANK_OPTION=" --rank ${PADDLE_TRAINER_ID}"
    else
        PADDLE_RANK_OPTION=""
    fi
    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP1-MP1) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --nnodes=1 --master=127.0.0.1:12538 \
        --devices=0 ${PADDLE_RANK_OPTION} passl-train -c ${yaml_path} \
        ${train_cmd}"
        workerlog_id=0
        ;;
    DP8-MP8|DP8-MP1) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --nnodes=1 --master=127.0.0.1:12538 \
        --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION} passl-train -c ${yaml_path} \
        ${train_cmd}"
        workerlog_id=0
        ;;
    *) echo "choose run_mode "; exit 1;
    esac
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    timeout 100m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.${workerlog_id} ${log_file}
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
