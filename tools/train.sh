#!/usr/bin/env bash

# for single card train
# python tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
#TRAINER_IP_LIST=127.0.0.1
python -m paddle.distributed.launch --ips=$TRAINER_IP_LIST --gpus="0,1,2,3,4,5,6,7" tools/train.py -c ./passl/configs/VisionTransformer/ViT_base_patch16_224_4n32c.yaml
