# probst = 000082
# woman with glasses: 000451
INPUT_PREFIX=$1
GPU_ID=$2
export CUDA_VISIBLE_DEVICES=$GPU_ID

python run.py \
 -face_comparer_config configs/arcface_adv.yml \
 -output_dir paper/$INPUT_PREFIX/adv \
 -overwrite \
 -duplicates=64 \
 -loss_str=100*L2+0.05*GEOCROSS+10.0*IDENTITY_SCORE \
 -input_prefix=$INPUT_PREFIX \
 -input_dir=CelebA_small/celeba/img_align_celeba \
 -targets_dir=CelebA_large/celeba/img_align_celeba \
 -gpu_id=0 \
 -copy_target \
 -eps=0.005 \
 -celeba_pairs

python run.py \
 -face_comparer_config configs/arcface_basic.yml \
 -output_dir paper/$INPUT_PREFIX/noadv \
 -overwrite \
 -duplicates=64 \
 -loss_str=100*L2+0.05*GEOCROSS+10.0*IDENTITY_SCORE \
 -input_prefix=$INPUT_PREFIX \
 -input_dir=CelebA_small/celeba/img_align_celeba \
 -targets_dir=CelebA_large/celeba/img_align_celeba \
 -gpu_id=0 \
 -copy_target \
 -eps=0.005 \
 -celeba_pairs


python run.py \
 -face_comparer_config configs/arcface_basic.yml \
 -output_dir paper/$INPUT_PREFIX/vanilla \
 -overwrite \
 -duplicates=64 \
 -loss_str=100*L2+0.05*GEOCROSS \
 -input_prefix=$INPUT_PREFIX \
 -input_dir=CelebA_small/celeba/img_align_celeba \
 -targets_dir=CelebA_large/celeba/img_align_celeba \
 -gpu_id=0 \
 -copy_target \
 -eps=0.005 \
 -celeba_pairs