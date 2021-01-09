# probst = 000082
# woman with glasses: 000451
INPUT_PREFIX=$1
GPU_ID=$2
export CUDA_VISIBLE_DEVICES=$GPU_ID

python run.py \
 -face_comparer_config configs/arcface_fairface.yml \
 -output_dir paper/fairfacecompfull/adv \
 -overwrite \
 -loss_str=100*L2+0.05*GEOCROSS+1.0*ATTR_SOURCE_IS_1 \
 -input_prefix=$INPUT_PREFIX \
 -input_dir=fairface_val \
 -targets_dir=fairface_val_large \
 -gpu_id=0 \
 -eps=0.005 \
 -fairface_csv_path=fairface/fairface_label_val.csv

python run.py \
 -face_comparer_config configs/arcface_basic.yml \
 -output_dir paper/fairfacecompfull/noadv \
 -overwrite \
 -loss_str=100*L2+0.05*GEOCROSS+10.0*IDENTITY_SCORE \
 -input_prefix=$INPUT_PREFIX \
 -input_dir=fairface_val \
 -targets_dir=fairface_val_large \
 -gpu_id=0 \
 -eps=0.005

python run.py \
 -face_comparer_config configs/arcface_basic.yml \
 -output_dir paper/fairfacecompfull/vanilla \
 -overwrite \
 -loss_str=100*L2+0.05*GEOCROSS \
 -input_prefix=$INPUT_PREFIX \
 -input_dir=fairface_val \
 -targets_dir=fairface_val_large \
 -gpu_id=0 \
 -eps=0.005
