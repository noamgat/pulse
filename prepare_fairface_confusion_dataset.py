import glob
import os
import re
import shutil

src_dir = 'paper/fairface'
dst_dir = 'fairface/confusion_test'
dst_file = 'fairface/fairface_label_confusion.csv'

RACE_NAMES = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

if __name__ == '__main__':
    os.makedirs(dst_dir, exist_ok=True)
    lines = ["img_path,age,gender,race,service_test"]
    for relative_path in glob.glob(src_dir + '/*/*.*', recursive=True):
        filename = os.path.split(relative_path)[-1]
        match = re.match('.+attr_(\d).+', relative_path)
        race = int(match.groups()[0])
        print(relative_path, filename, race)
        shutil.copy(relative_path, os.path.join(dst_dir, filename))
        race_name = RACE_NAMES[race]
        line_parts = ['confusion_test/' + filename, '3-9', 'Male', race_name, 'False']
        lines.append(",".join(line_parts))
    output_csv = open(dst_file, "w").write("\r\n".join(lines))
    print(f"Done with {len(lines)-1} images")

