""" 
Process data.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

import sys
sys.path.append(str(root_dir))

from eval.data_utils import *

import json
import jsonlines
import io
from PIL import Image, ImageDraw
import shutil
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from tqdm import tqdm


spatial_words_dict = {
    "Adjacency": [
        "adjacent to", "alongside", "at the side of", "at the right side of", 
        "at the left side of", "attached to", "at the back of", "ahead of", 
        "against", "at the edge of"
    ],
    "Directional": [
        "off", "past", "toward", "down", "deep down", "up", "away from", 
        "along", "around", "from", "into", "to", "across", "across from", 
        "through", "down from"
    ],
    "Orientation": [
        "facing", "facing away from", "parallel to", "perpendicular to"
    ],
    "Projective": [
        "on top of", "beneath", "beside", "behind", "left of", "right of", 
        "under", "in front of", "below", "above", "over", "in the middle of"
    ],
    "Proximity": [
        "by", "close to", "near", "far from", "far away from"
    ],
    "Topological": [
        "connected to", "detached from", "has as a part", "part of", "contains", 
        "within", "at", "on", "in", "with", "surrounding", "among", "consists of", 
        "out of", "between", "inside", "outside", "touching"
    ],
    "Unallocated": [
        "beyond", "next to", "opposite to", "after", "among", "enclosed by"
    ]
}

def process_gqa():
    data_path = [
        data_dir / "GQA/questions/testdev_balanced_questions.json",
        data_dir / "GQA/images/images"
    ]
    text_path, image_path = data_path
    all_data = json.load(open(text_path, "r"))
    print("all_data", len(all_data))
    
    spatial_words = []
    for key, value in spatial_words_dict.items():
        spatial_words.extend(value)
    
    spatial_words = ["left", "right", "top", "bottom", "in front of", "behind", "above", "below", "next to", "beside", "between", "on top of", "under", "over"]
    
    gqa_spatial = []
    gqa_no_spatial = []
    for qn_id, content in all_data.items():
        question = content["question"]
        imageID = content["imageId"]
        image = os.path.join(image_path, imageID + ".jpg")
        answer = content["answer"]
        if not any(f" {word}" in question for word in spatial_words) and not any(word in answer for word in spatial_words):
            gqa_no_spatial.append({"id": qn_id, "question": question, "image": image, "answer": answer})
        else:
            gqa_spatial.append({"id": qn_id, "question": question, "image": image, "answer": answer})
    
    print("gqa_no_spatial", len(gqa_no_spatial))
    data_dir = root_dir / "utils/GQA_data"
    spatial_path = os.path.join(data_dir, "gqa_spatial.jsonl")
    no_spatial_path = os.path.join(data_dir, "gqa_no_spatial.jsonl")
    with jsonlines.open(spatial_path, mode='w') as writer:
        for item in gqa_spatial:
            writer.write(item)
    with jsonlines.open(no_spatial_path, mode='w') as writer:
        for item in gqa_no_spatial:
            writer.write(item)

def process_gqa_for_sft(data_num=20000):
    data_path = [
        data_dir / "GQA/questions/train_balanced_questions.json",
        data_dir / "GQA/images/images"
    ]
    text_path, image_path = data_path
    all_data = json.load(open(text_path, "r"))
    print("all_data", len(all_data))
    
    spatial_words = []
    for key, value in spatial_words_dict.items():
        spatial_words.extend(value)
    
    spatial_words = ["left", "right", "top", "bottom", "in front of", "behind", "above", "below", "next to", "beside", "between", "on top of", "under", "over"]
    
    gqa_spatial = []
    gqa_no_spatial = []
    for qn_id, content in all_data.items():
        question = content["question"]
        imageID = content["imageId"]
        image = os.path.join(image_path, imageID + ".jpg")
        answer = content["answer"]
        conversations = []
        conversations.append({"from": "human", "value": "<image>\n" + question})
        conversations.append({"from": "gpt", "value": answer})
        if not any(f" {word}" in question for word in spatial_words) and not any(word in answer for word in spatial_words):
            gqa_no_spatial.append({"image": image, "conversations": conversations})
        else:
            gqa_spatial.append({"image": image, "conversations": conversations})
    if data_num is not None:
        gqa_spatial = gqa_spatial[:data_num]
    
    print("gqa_no_spatial", len(gqa_no_spatial))
    data_dir = root_dir / "train/data"
    spatial_path = os.path.join(data_dir, "gqa_spatial.jsonl")
    no_spatial_path = os.path.join(data_dir, "gqa_no_spatial.jsonl")
    if data_num is not None:
        spatial_path = spatial_path.split(".jsonl")[0] + f"_{data_num}.jsonl"
    with jsonlines.open(spatial_path, mode='w') as writer:
        for item in gqa_spatial:
            writer.write(item)
    with jsonlines.open(no_spatial_path, mode='w') as writer:
        for item in gqa_no_spatial:
            writer.write(item)
            
def process_spare():
    spare_data_path = root_dir / "train/data/spare_300k.jsonl"
    with jsonlines.open(spare_data_path, 'r') as f:
        for line in f:            
            image_name = line["image_file"].split('/')[-1]
            samples = line["parsed"][:2]
            for sample in samples:
                qn = sample["question"]
                ans = sample["answer"]
                print(qn, ans, image_name)
            # break
    
def get_gqa_bbox():
    image_path = data_dir / "GQA/images"
    files = os.listdir(image_path)
    hd5_files = [file for file in files if file.endswith(".h5")]
    print("hd5_files", len(hd5_files))
    
    link_path = data_dir / "GQA/gqa_objects_info.json"
    with open(link_path, "r") as f:
        data = json.load(f)
    print("data", type(data))
    
    import h5py
    # https://docs.h5py.org/en/stable/quick.html
    path = data_dir / "GQA/objects/gqa_objects_7.h5"
    with h5py.File(path, 'r') as f:
        print(list(f.keys()))
        bbox_file = f["bboxes"]
        data = bbox_file[:]
        print("data", data)
        print("data.shape", data.shape)  # (ImagesNum, 100, 4)

def get_square_images():
    image_path = data_dir / "GQA/images/images"
    tgt_dir = root_dir / "test_figs/gqa"
    files = os.listdir(image_path)
    square_images = []
    for file in files:
        image_file = os.path.join(image_path, file)
        image = Image.open(image_file)
        width, height = image.size
        if width == height:
            square_images.append(file)
            shutil.copy(image_file, os.path.join(tgt_dir, file))
            print("square_image", file)
    
    print("square_images", len(square_images))

def get_scene_info():
    train_scene_path = data_dir / "GQA/scene/train_sceneGraphs.json"
    val_scene_path = data_dir / "GQA/scene/val_sceneGraphs.json"
    train_scene_data = json.load(open(train_scene_path, "r"))
    val_scene_data = json.load(open(val_scene_path, "r"))
    key = "2333052"  # 2354453 2332870
    print(train_scene_data[key])
    objects = train_scene_data[key]["objects"]
    names = [obj["name"] for obj in objects.values()]
    print("names", names)
    pass

def process_whatsup(dataset_name):
    root_dir=data_dir / "whatsup_vlms/data/whatsup_vlms_data"
    dataset1 = Controlled_Images(image_preprocess=None, subset="B", root_dir=root_dir, directions=[])
    for i in range(len(dataset1)):
        if "on " in dataset1[i]["caption_options"][0]:
            print("dataset1", dataset1[i])
     
    # dataset2 = Controlled_Images(image_preprocess=None, subset="B", root_dir=root_dir, directions=[])
    # for i in range(5):
    #     print("dataset1", dataset1[i])
    # for i in range(5):
    #     print("dataset2", dataset2[i])
    # print("dataset1", len(dataset1))
    # print("dataset2", len(dataset2))

def get_bbox_from_xml(xml_file):
    with open(xml_file, 'r') as f:
        xml_data = f.read()
        
    # Parse XML
    root = ET.fromstring(xml_data)

    # Extract bounding box information
    bbox_dict = {}
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox_dict[name] = (xmin, ymin, xmax, ymax)

    print("bbox_dict", bbox_dict)
    return bbox_dict

def draw_bbox(
    image_path,
    bboxes=None,
    save_dir="",
):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for name, bbox in bboxes.items():
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), name, fill="red")
    # save
    image_name = os.path.basename(image_path).split(".")[0]
    save_path = os.path.join(save_dir, f"{image_name}_objects_bbox.png")
    image.save(save_path)
    print(f"save_path: {save_path}")

def process_pope():
    
    data_type = "adversarial"  # random, popular, adversarial
    data_path = data_dir / f"POPE/Full/{data_type}-00000-of-00001.parquet"
    save_path = data_dir / f"POPE/{data_type}.jsonl"
    image_dir = os.path.join(data_dir / "POPE/images", data_type)
    os.makedirs(image_dir, exist_ok=True)
    
    df = pd.read_parquet(data_path)
    
    # print(df.info())
    # print("df", df.shape)  # (3000, 7)
    # print("df.columns", df.columns)  # ['id', 'question_id', 'question', 'answer', 'image_source', 'image','category']
    # print("df.head()", df.head())
    
    # sample = df.iloc[0]
    # print("sample", sample)

    # image = Image.open(io.BytesIO(sample['image']["bytes"]))
    # image.save(root_dir / "test_figs/pope/0.png")

    data_num = len(df)
    with jsonlines.open(save_path, 'w') as f:
        for idx in tqdm(range(data_num)):
            sample = df.iloc[idx]
            
            image_bytes = sample['image']["bytes"]
            image = Image.open(io.BytesIO(image_bytes))
            image_name = f"{sample['id']}.png"
            image_path = os.path.join(image_dir, image_name)
            image.save(image_path)
            
            info = {
                "id": sample['id'],
                "question_id": sample['question_id'],
                "question": sample['question'],
                "answer": sample['answer'],
                "image_source": sample['image_source'],
                "image": f"{sample['id']}.png",  # Save the path to the image
                "category": sample['category']
            }

            f.write(info)

def process_mme():
    image_data_dir = data_dir / "MME/data"
    image_path_0 = os.path.join(image_data_dir, "test-00000-of-00002.parquet")
    image_path_1 = os.path.join(image_data_dir, "test-00001-of-00002.parquet")
    df_0 = pd.read_parquet(image_path_0)
    df_1 = pd.read_parquet(image_path_1)
    df = pd.concat([df_0, df_1], ignore_index=True)
    # print("df", df.shape)  # (1187, 5)
    # print("df.columns", df.columns)  # 'question_id', 'image', 'question', 'answer', 'category'
    # print("df.head()", df.head())
    # print(df.iloc[0]["image"]["path"])
    
    data_num = len(df)
    new_data_dir = data_dir / "MME/MME_data"
    for idx in tqdm(range(data_num)):
        sample = df.iloc[idx]
        category = sample['category']
        data_dir = os.path.join(new_data_dir, category)
        image_dir = os.path.join(data_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        text_path = os.path.join(data_dir, "data.jsonl")
        
        image_bytes = sample['image']["bytes"]
        image = Image.open(io.BytesIO(image_bytes))
        image_name = sample['image']["path"]
        image.save(os.path.join(image_dir, image_name))
        
        info = {
            "question_id": sample['question_id'],
            "image": image_name,
            "question": sample['question'],
            "answer": sample['answer'],
            "category": sample['category']
        }
        with jsonlines.open(text_path, 'a') as f:
            f.write(info)
    
def process_vqa():
    qn_path = data_dir / "VQA_v2/v2_OpenEnded_mscoco_val2014_questions.json"
    annotation_path = data_dir / "VQA_v2/v2_mscoco_val2014_annotations.json"
    
    # open question file
    with open(qn_path, 'r') as f:
        qn_data = json.load(f)
    questions = qn_data["questions"]
    # print("qn_data", len(qn_data["questions"]))  # 214354
    # print("qn_data.keys()", qn_data.keys())  # ['info', 'task_type', 'data_type', 'license', 'data_subtype', 'questions']
    print("questions[0]", questions[0])  # {'question_id': 500000, 'image_id': 300000, 'question': 'What is the color of the car?'}

    # open annotation file
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)
    annotations = annotation_data["annotations"]
    # print("annotation_data", len(annotation_data["annotations"]))  # 214354
    # print("annotation_data.keys()", annotation_data.keys())  # ['info', 'task_type', 'data_type', 'license', 'data_subtype', 'annotations']
    print("annotations[0]", annotations[0])  # {'question_id': 500000, 'image_id': 300000, 'answers': [{'answer': 'red', 'answer_confidence':
    
    res_path = data_dir / "VQA_v2/val.jsonl"
    with jsonlines.open(res_path, 'w') as f:
        for idx in tqdm(range(len(questions))):
            question = questions[idx]
            annotation = annotations[idx]
            question_id = question["question_id"]
            image_id = question["image_id"]
            qn = question["question"]
            answers = annotation["answers"]
            answer_list = [ans["answer"] for ans in answers]
            
            info = {
                "question_id": question_id,
                "image_id": image_id,
                "question": qn,
                "answers": answer_list
            }
            f.write(info)

def process_textvqa():
    data_path = str(data_dir / "TextVQA/data")
    all_files = os.listdir(data_path)
    train_files = [file for file in all_files if "train" in file]
    val_files = [file for file in all_files if "val" in file and "parquet" in file]
    test_files = [file for file in all_files if "test" in file and "parquet" in file]

    # get all images
    images_dict = {}
    for file in val_files:
        file_path = os.path.join(data_path, file)
        df = pd.read_parquet(file_path)
        # print(f"Processing {file}: {df.shape}")
        # print(df.columns)  # ['image_id', 'question_id', 'question', 'question_tokens', 'image', 'image_width', 'image_height', 'flickr_original_url', 'flickr_300k_url', 'answers', 'image_classes', 'set_name', 'ocr_tokens']
        # print(df.head())
        # print("Sample image data:", df.iloc[0])
        # print("Sample image data:", df.iloc[0]["image"])
        # break

        # Sample image data: image_id                                                ac71817a98529f20
        # question_id                                                        29412
        # question                                                what time is it?
        # image                  {'bytes': b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x...
        # answers                [coca cola , 12:49, 1:54, 12:54, 12:55pm, 12:5...
        # image_classes                                 [Clock, Watch, Wall clock]


        for idx in tqdm(range(len(df))):
            sample = df.iloc[idx]
            image_id = sample["image_id"]
            image_data = sample["image"]
            if isinstance(image_data, dict):
                image_bytes = image_data["bytes"]
                image_name = image_data["path"]
                images_dict[image_id] = {
                    "bytes": image_bytes,
                    "path": image_name,
                    "question_id": int(sample["question_id"]),
                    "question": sample["question"],
                    "answers": list(sample["answers"])
                }
    # save images under a folder
    image_dir = os.path.join(data_path, "val_images")
    os.makedirs(image_dir, exist_ok=True)
    test_data_path = os.path.join(data_path, "val_data.jsonl")
    with jsonlines.open(test_data_path, mode='w') as f:
        for image_id, image_info in images_dict.items():
            image_bytes = image_info["bytes"]
            image_name = image_info["path"]
            image_path = os.path.join(image_dir, image_name)
            with open(image_path, 'wb') as img_file:
                img_file.write(image_bytes)
            sample = {
                "question_id": image_info["question_id"],
                "question": image_info["question"],
                "image": image_name,  # Save the path to the image
                "answers": image_info["answers"]
            }
            # print("sample", sample)
            # import pdb; pdb.set_trace()
            f.write(sample)  

def create_vision_decoder_val_dataset():
    coco_path = str(data_dir / "Hallucination/coco/train2014")
    textvqa_path = str(data_dir / "TextVQA/data/images")
    val_dataset = []
    all_samples = []
    val_dataset_path = root_dir / "train_unembedding/data/val_dataset.jsonl"
    for file in os.listdir(coco_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            all_samples.append({
                "image": os.path.join(coco_path, file),
                "question": "What is in this image?",
            })
    for file in os.listdir(textvqa_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            all_samples.append({
                "image": os.path.join(textvqa_path, file),
                "question": "What is in this image?",
            })
    import random
    random.shuffle(all_samples)
    for idx in range(1000):
        val_dataset.append(all_samples[idx])
    with jsonlines.open(val_dataset_path, mode='w') as writer:
        for item in val_dataset:
            writer.write(item)

def process_sqa():
    data_path = str(data_dir / "SQA/data")
    val_file_path = str(data_dir / "SQA/data/validation-00000-of-00001-6c7328ff6c84284c.parquet")
    test_file_path = str(data_dir / "SQA/data/test-00000-of-00001-f0e719df791966ff.parquet")
    
    # df = pd.read_parquet(test_file_path)
    # print("df", df.shape)
    # print("df.columns", df.columns)
    # print(df.iloc[10])
    # print(df.iloc[10]["image"])
    # for i in range(len(df)):
    #     print(df.iloc[i]["answer"])
    # return

    # get all images
    images_dict = {}
    df = pd.read_parquet(test_file_path)

    for idx in tqdm(range(len(df))):
        sample = df.iloc[idx]
        image_data = sample["image"]
        if isinstance(image_data, dict):
            image_bytes = image_data["bytes"]
            images_dict[idx] = {
                "question_id": idx,
                "bytes": image_bytes,
                "path": f"{idx}.png",
                "question": sample["question"],
                "choices": list(sample["choices"]),
                "answer": sample["answer"],
            }
        else:
            images_dict[idx] = {
                "question_id": idx,
                "path": None,
                "question": sample["question"],
                "choices": list(sample["choices"]),
                "answer": sample["answer"],
            }

    # save images under a folder
    image_dir = os.path.join(data_path, "test_images")
    os.makedirs(image_dir, exist_ok=True)
    test_data_path = os.path.join(data_path, "test_data.jsonl")
    with jsonlines.open(test_data_path, mode='w') as f:
        for image_id, image_info in images_dict.items():
            if "bytes" in image_info:
                image_bytes = image_info["bytes"]
                image_name = image_info["path"]
                image_path = os.path.join(image_dir, image_name)
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_bytes)
            sample = {
                "question_id": image_info["question_id"],
                "question": image_info["question"],
                "image": image_info["path"],  # Save the path to the image
                "choices": image_info["choices"],
                "answer": int(image_info["answer"])
            }
            # print("sample", sample)
            # import pdb; pdb.set_trace()
            f.write(sample)

def process_mmbench():
    data_path = str(data_dir / "MMBench/en")
    dev_file_path = os.path.join(data_path, "dev-00000-of-00001.parquet")
    df = pd.read_parquet(dev_file_path)
    print("df", df.shape)  # (1000, 6)
    print("df.columns", df.columns)
    print("df.head()", df.head())
    print("df.iloc[0]", df.iloc[0])
    # print("df.iloc[0]['image']", df.iloc[0]["image"])
    # return
    dev_image_dir = os.path.join(data_path, "dev_images")
    os.makedirs(dev_image_dir, exist_ok=True)
    dev_data_path = os.path.join(data_path, "dev_data.jsonl")
    with jsonlines.open(dev_data_path, mode='w') as f:
        for idx in tqdm(range(len(df))):
            sample = df.iloc[idx]
            image_data = sample["image"]
            if isinstance(image_data, dict):
                image_bytes = image_data["bytes"]
                image_name = f"{idx}.png"
                image_path = os.path.join(dev_image_dir, image_name)
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_bytes)
            else:
                image_name = None
            
            info = {
                "id": int(sample['index']),
                "question": sample['question'],
                "image": image_name,  # Save the path to the image
                "answer": sample['answer'],
                "choices": [sample["A"], sample["B"], sample["C"], sample["D"]],
            }
            f.write(info)


if __name__ == "__main__":
    # process_gqa()
    # process_gqa_for_sft(data_num=60000)
    # process_spare()
    
    # f = jsonlines.open(root_dir / "utils/GQA_data/gqa_spatial.jsonl", mode='r')
    # data = [line for line in f]
    # print(data[0])
    
    # get_gqa_bbox()
    
    # get_square_images()
    
    # process_whatsup("whatsup_a")
    # get_scene_info()
    
    # get_bbox_from_xml(root_dir / "eval/WhatsUp/bboxes/test_label/1.xml")
    def get_bboxes_from_xmls():
        xml_path = root_dir / "eval/WhatsUp/bboxes/label"
        xml_paths = os.listdir(xml_path)
        res = []
        for xml_file in xml_paths:
            xml_file_path = os.path.join(xml_path, xml_file)
            bbox_dict = get_bbox_from_xml(xml_file_path)
            data = {
                "image": xml_file.replace(".xml", ''),
                "bbox": bbox_dict
            }
            res.append(data)
        save_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
        with jsonlines.open(save_path, mode='w') as writer:
            for item in res:
                writer.write(item)
    
    # get_bboxes_from_xmls()
    
    # image_id = "book_behind_can"
    # bbox_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
    # with jsonlines.open(bbox_path, mode='r') as f:
    #     data = [line for line in f]
    # bbox = [item for item in data if item["image"] == image_id][0]["bbox"]
    # draw_bbox(
    #     image_path=fdata_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr/{image_id}.jpeg",
    #     bboxes=bbox,
    #     save_dir=root_dir / "test_figs/gqa"
    # )
    
    # annotation_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data"
    # annotation_path = os.path.join(annotation_dir, "controlled_clevr_dataset.json")
    # annotations = json.load(open(annotation_path))
    # print(annotations[0])

    # process_mme()
    
    # process_vqa()

    # process_textvqa()
    # create_vision_decoder_val_dataset()

    # process_sqa()

    process_mmbench()
