python run.py \
 -face_comparer_config configs/linear_basic.yml \
 -output_dir runs/pulse_vanilla \
 -overwrite \
 -duplicates=8 \
 -loss_str=100*L2+0.05*GEOCROSS \
 -input_prefix=000 \
 -input_dir=CelebA_small/celeba/img_align_celeba \
 -targets_dir=CelebA_large/celeba/img_align_celeba \
 -gpu_id=7 \
 -copy_target

