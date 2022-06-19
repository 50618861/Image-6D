#!/bin/bash
#$ -l gpu=1 -l rmem=40G
#$ -P acsehpc
#$ -q acsehpc.q

module load apps/python/conda
source activate my_env_1
module load libs/cudnn/7.6.5.32/binary-cuda-10.0.130

cd /data/coq18yj/Mask_RCNN/samples/cam/
python3 valid.py