GPU_ID=2
export CUDA_VISIBLE_DEVICES=$GPU_ID

python run.py \
 -face_comparer_config configs/arcface_basic.yml \
 -output_dir paper/jeff_vanilla \
 -overwrite \
 -duplicates=64 \
 -loss_str=100*L2+0.05*GEOCROSS \
 -input_prefix=000082 \
 -input_dir=CelebA_small/celeba/img_align_celeba \
 -targets_dir=CelebA_large/celeba/img_align_celeba \
 -gpu_id=0 \
 -copy_target \
 -eps=0.005 \
 -celeba_pairs
