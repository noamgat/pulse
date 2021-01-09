#!/bin/bash

# probst = 000082
# woman with glasses: 000451
INPUT_PREFIX=$1
GPU_ID=$2
export CUDA_VISIBLE_DEVICES=$GPU_ID

for attr in {0..39}
do
  for attr_val in 0 1
  do
python run.py \
 -face_comparer_config configs/arcface_adv.yml \
 -output_dir paper/attrs/$INPUT_PREFIX/attr_${attr}_${attr_val} \
 -overwrite \
 -duplicates=4 \
 -loss_str=100*L2+0.05*GEOCROSS+10.0*ATTR_${attr}_IS_${attr_val} \
 -input_prefix=$INPUT_PREFIX \
 -input_dir=CelebA_small/celeba/img_align_celeba \
 -gpu_id=0 \
 -eps=0.005
  done
done


