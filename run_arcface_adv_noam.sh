python run.py \
 -face_comparer_config configs/arcface_adv.yml \
 -output_dir runs/arcface_adv_insert_noam \
 -overwrite \
 -duplicates=8 \
 -loss_str=100*L2+0.05*GEOCROSS+10.0*IDENTITY_SCORE \
 -input_prefix=000 \
 -input_dir=CelebA_small/celeba/img_align_celeba \
 -targets_dir=targets/noam1_0.png \
 -gpu_id=7 \
 -copy_target

