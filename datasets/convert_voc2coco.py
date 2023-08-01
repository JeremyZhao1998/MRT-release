import os
import json
from tqdm import tqdm
import xml.dom.minidom as minidom


def convert_split(split_name, anno_file_names=None):
    new_file_name = src_dir_name + '_' + split_name + '_cocostyle.json'
    new_annotation_path = os.path.join(new_annotation_dir, new_file_name)
    anno_file = {"images": [], "categories": categories, "annotations": []}
    type_cnt = {cid: 0 for cid in categories_dict.values()}
    if anno_file_names is None:
        print("Converting " + split_name + " split")
    else:
        print("Converting " + split_name + " split, size: " + str(len(anno_file_names)))
    for root, dirs, files in os.walk(os.path.join(old_annotation_dir)):
        for img_id, file in tqdm(enumerate(files), total=len(files)):
            if anno_file_names is not None and file[:-4] not in anno_file_names:
                continue
            xml_path = os.path.join(root, file)
            dom_root = minidom.parse(xml_path).documentElement
            size = dom_root.getElementsByTagName('size')[0]
            anno_file['images'].append({
                "id": img_id,
                "width": size.getElementsByTagName('width')[0].childNodes[0].data,
                "height": size.getElementsByTagName('height')[0].childNodes[0].data,
                "file_name": dom_root.getElementsByTagName('filename')[0].childNodes[0].data
            })
            for obj in dom_root.getElementsByTagName('object'):
                category = obj.getElementsByTagName('name')[0].childNodes[0].data
                if category not in categories_dict:
                    continue
                xmin = float(obj.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymin = float(obj.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmax = float(obj.getElementsByTagName('xmax')[0].childNodes[0].data)
                ymax = float(obj.getElementsByTagName('ymax')[0].childNodes[0].data)
                anno_file['annotations'].append({
                    "id": len(anno_file['annotations']),
                    "image_id": img_id,
                    "category_id": categories_dict[category],
                    "iscrowd": 0,
                    "area": (xmax - xmin) * (ymax - ymin),
                    "bbox": [xmin, ymax, xmax - xmin, ymax - ymin]
                })
                type_cnt[categories_dict[category]] += 1
    print("Type count: " + str(type_cnt))
    print("Writing new annotations to " + new_annotation_path)
    with open(new_annotation_path, 'w', encoding='utf-8') as fp:
        json.dump(anno_file, fp)


def main():
    convert_all = not os.path.exists(os.path.join(src_dir, 'ImageSets/Main'))
    if convert_all:
        convert_split('train')
    else:
        for root, dirs, files in os.walk(os.path.join(src_dir, 'ImageSets/Main')):
            for file in files:
                if file in ['test.txt', 'train.txt', 'val.txt']:
                    names = [line.strip() for line in open(os.path.join(root, file), 'r', encoding='utf-8').readlines()]
                    convert_split(file[:-4], names)


if __name__ == '__main__':
    data_root = '/network_space/storage43/zhaozijing/datasets/'
    src_dir_name = 'voc0712'  # 'clipart'  # 'watercolor'  # 'sim10k'
    # Set source directory
    src_dir = os.path.join(data_root, src_dir_name)
    # Set old annotation directory
    old_annotation_dir = os.path.join(src_dir, 'Annotations')
    # Set image directory
    image_dir = os.path.join(src_dir, 'JPEGImages')
    assert os.path.exists(src_dir) and os.path.exists(old_annotation_dir) and os.path.exists(image_dir)
    # Read label names
    assert os.path.exists(os.path.join(src_dir, 'labels.txt'))
    label_names = [label
                   for label in open(os.path.join(src_dir, 'labels.txt'), 'r', encoding='utf-8').read().split('\n')
                   if len(label) > 0]
    categories_dict = {label: idx + 1 for idx, label in enumerate(label_names)}  # 0 is reserved for background
    categories = [{"id": idx, "name": label} for label, idx in categories_dict.items()]
    # Create new annotation directory
    new_annotation_dir = os.path.join(src_dir, 'annotations')
    os.system('mkdir -p ' + new_annotation_dir)
    main()
