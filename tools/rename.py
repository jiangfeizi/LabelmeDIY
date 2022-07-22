from datetime import datetime
import argparse
import glob
import os
import shutil
import json

import utils



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rename images and jsons.")
    parser.add_argument("root_dir", help="The directory of images that need to be renamed.")
    parser.add_argument("-o", "--output", default="./output",help="The path of weights.")
    args = parser.parse_args()

    file_list = glob.glob(f'{args.root_dir}/**/*', recursive=True)
    image_list = [item for item in file_list if utils.is_image(item)]

    for image_path in image_list:
        pre, ext = os.path.splitext(image_path)
        label_path = pre + ".json"

        strftime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        shutil.copy(image_path, os.path.join(args.output, strftime + ext))
        if os.path.exists(label_path):
            data = json.load(open(label_path, encoding='utf8'))
            data['imagePath'] = strftime + ext
            json.dump(data, open(os.path.join(args.output, strftime + '.json'), 'w', encoding='utf8'))