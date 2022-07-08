#!/usr/bin/env bash
NETSCRATCH="/netscratch/schulz"
DATAROOT="$NETSCRATCH/question_answering_german"
DATADIR="$DATAROOT/exp"
HF_DATASETS_CACHE="$NETSCRATCH/hf_datasets_cache"
TRANSFORMERS_CACHE="$NETSCRATCH/transformers_cache"
ROOTDIR="`dirname \"$0\"`"
ROOTDIR="`readlink -f ${ROOTDIR}`"  # this is the directory that contains this script
BASE_IMAGE="nvcr.io_nvidia_pytorch_22.02-py3.sqsh"
ENROOT_DIR="/netscratch/enroot"
srun --container-image=$ENROOT_DIR/$BASE_IMAGE \
  --container-workdir="`pwd`" \
  --container-mounts=$NETSCRATCH:$NETSCRATCH,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-save=$NETSCRATCH/docker_images/$BASE_IMAGE \
  ./env.sh && pip install -r --no-cache-dir requirements.txt && python germanQuAD_train.py
