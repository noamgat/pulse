import glob
import os
import re
import shutil
from pathlib import Path

import pandas as pd


job_datas = [
    ('paper/fairfacecomp/vanilla', 'fairface/comparison_vanilla', 'fairface/fairface_label_comparison_vanilla.csv'),
    ('paper/fairfacecomp/noadv', 'fairface/comparison_noadv', 'fairface/fairface_label_comparison_noadv.csv'),
    ('paper/fairfacecomp/adv', 'fairface/comparison_adv', 'fairface/fairface_label_comparison_adv.csv'),
]
RACE_NAMES = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

if __name__ == '__main__':
    base_dataframe = pd.read_csv('fairface/fairface_label_val.csv')
    input_races = {}
    for input_filename, input_race in zip(base_dataframe['file'], base_dataframe['race']):
        input_races[Path(input_filename).stem] = input_race

    for src_dir, dst_dir, dst_file in job_datas:
        os.makedirs(dst_dir, exist_ok=True)
        lines = ["img_path,age,gender,race,service_test"]
        for relative_path in glob.glob(src_dir + '/*.*', recursive=True):
            filename = Path(relative_path).stem + relative_path[-4:]
            image_name = filename.split('_')[0]
            race_name = input_races[image_name]
            shutil.copy(relative_path, os.path.join(dst_dir, filename))
            line_parts = [dst_dir.split('/')[-1] + '/' + filename, '3-9', 'Male', race_name, 'False']
            lines.append(",".join(line_parts))
        output_csv = open(dst_file, "w").write("\r\n".join(lines))
        print(f"Wrote {dst_file} with {len(lines)-1} images")

