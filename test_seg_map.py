"""
Test directions. 
"""
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoModel,
    AutoConfig,
    AutoTokenizer, 
    AutoProcessor
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from qwen_vl_utils import process_vision_info as process_vision_info_qwen
from utils import load_image_intern
import clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import Counter

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

import sys
sys.path.append(str(root_dir))

from eval.data_utils import *
from eval.CHAIR.utils.chair_new import evaluate_chair
from eval.MME.mme import evaluate_mme
from eval.VQA.vqa import evaluate_vqa
from patch.monkey_patch import *
from model.unembedding import VisionTokenDecoder

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import svgwrite
from pathlib import Path
from tqdm import tqdm
import glob
import json
import jsonlines
import os
import copy
import shutil
import string
import re


MODEL_NAME_TO_PATH = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_vl_7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2_vl_2b": "Qwen/Qwen2-VL-2B-Instruct",
    "llava1_5_7b": "llava-hf/llava-1.5-7b-hf",  # "llava-hf/llava-1.5-7b-hf", "liuhaotian/llava-v1.5-7b"
    "internvl2_5_8b": "OpenGVLab/InternVL2_5-8B",
}
DATASET_NAME_TO_OFFICIAL = {
    "coco": "COCO",
    "GQA": "GQA",
    "mme": "MME",
}

def load_data(dataset_name, model_name, processor, data_num=1000, random=False, batch_size=8, **kwargs):
    
    if "qwen" in model_name:
        vision_process_func = process_vision_info_qwen
    elif "intern" in model_name:
        vision_process_func = load_image_intern
    else:
        # raise ValueError("Unsupported model type.")
        vision_process_func = None
    
    if "whatsup" in dataset_name:
        root_dir = data_dir / "whatsup_vlms" / "data" / "whatsup_vlms_data"
        if dataset_name == "whatsup_a":
            dataset = Controlled_Images(image_preprocess=None, subset="A", root_dir=root_dir)    
        elif dataset_name == "whatsup_b":
            dataset = Controlled_Images(image_preprocess=None, subset="B", root_dir=root_dir)
        else:
            raise ValueError("Invalid dataset name.")
        data_collator = Controlled_Images_collate_function(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "cocoqa" in dataset_name:
        root_dir = data_dir / "whatsup_vlms" / "data" / "whatsup_vlms_data"
        if dataset_name == "cocoqa_1":
            dataset = COCO_QA(image_preprocess=None, subset='one', root_dir=root_dir)
        elif dataset_name == "cocoqa_2":
            dataset = COCO_QA(image_preprocess=None, subset='two', root_dir=root_dir)
        else:
            raise ValueError("Invalid dataset name.")
        data_collator = COCO_QA_collate_function(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "gqa_" in dataset_name:
        if "no_spatial" in dataset_name:
            data_path = [
                str(data_dir / "GQA/utils/GQA_data/gqa_no_spatial.jsonl"),
                str(data_dir / "GQA/images/images")
            ]
            data_num  = None
            dataset = GQADataset(data_path, data_num=data_num, random_select=random)
            data_collator = GQACollator(processor, vision_process_func=vision_process_func)
        else:
            root_dir = data_dir / "whatsup_vlms" / "data" / "whatsup_vlms_data"
            if dataset_name == "gqa_1":
                dataset = VG_QA(image_preprocess=None, subset='one', root_dir=root_dir)
            elif dataset_name == "gqa_2":
                dataset = VG_QA(image_preprocess=None, subset='two', root_dir=root_dir)
            else:
                raise ValueError("Invalid dataset name.")
            data_collator = VG_QA_collate_function(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif dataset_name == "vsr":
        text_path = data_dir / "visual-spatial-reasoning/data/splits/random/test.jsonl"
        image_path = data_dir / "visual-spatial-reasoning/data/images/test"
        data_path = [text_path, image_path]
        dataset = VSRDataset(data_path=data_path, data_num=None, random_select=random)
        data_collator = VSRCollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "GQA" in dataset_name:
        data_path = [
            str(data_dir / "GQA/questions/testdev_balanced_questions.json"),
            str(data_dir / "GQA/images/images")
        ]
        qa_mode = kwargs.get("QA_mode", True)
        data_collator = GQACollator(processor, vision_process_func=vision_process_func, model_name=model_name, QA_mode=qa_mode)
        random=True
        dataset = GQADataset(data_path, data_num=data_num, random_select=random)
    elif "pope" in dataset_name:
        data_type = dataset_name.split("_")[-1]  # e.g., "random", "popular", "adversarial"
        data_path = [
            str(data_dir / f"POPE/{data_type}.jsonl"),
            str(data_dir / f"POPE/images/{data_type}")
        ]
        data_num = None
        dataset = POPEDataset(data_path, data_num=data_num, random_select=random)
        data_collator = POPECollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "coco" in dataset_name:
        data_path = [
            data_dir / "Hallucination/coco/annotations/captions_train2014.json",
            data_dir / "Hallucination/coco/train2014"
        ]
        dataset = COCODataset(data_path, data_num=data_num, random_select=random)
        data_collator = COCOCollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "textvqa" in dataset_name:
        data_path = data_dir / "TextVQA/data"
        dataset = TextVQADataset(data_path, data_num=data_num, random_select=random)
        data_collator = TextVQACollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "vqa" in dataset_name:
        data_path = [
            data_dir / "VQA_v2/v2_OpenEnded_mscoco_val2014_questions.json",
            data_dir / "VQA_v2/v2_mscoco_val2014_annotations.json",
            data_dir / "VQA_v2/images/val2014"
        ]
        dataset = VQADataset(data_path, data_num=data_num, random_select=random)
        data_collator = VQACollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "mme" in dataset_name:
        data_path = data_dir / "MME/MME_data"
        dataset = MMEDataset(data_path, data_num=data_num, random_select=random)
        data_collator = MMECollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "mmb" in dataset_name:
        data_path = data_dir / "MMBench/en"
        dataset = MMBDataset(data_path, data_num=data_num, random_select=random)
        data_collator = MMBCollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "sqa" in dataset_name:
        data_path = data_dir / "SQA/data"
        dataset = SQADataset(data_path, data_num=data_num, random_select=random)
        data_collator = SQACollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    else:
        raise ValueError("Invalid dataset name.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    
    return dataloader

def load_model(model_name, device, use_flash_attention=True):
    model, processor, tokenizer = None, None, None
    
    if "qwen" in model_name:
        model_dir = MODEL_NAME_TO_PATH[model_name]
        model_class = Qwen2_5_VLForConditionalGeneration if "qwen2_5" in model_name else Qwen2VLForConditionalGeneration
        model = model_class.from_pretrained(
            model_dir, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
        )
        model.to(device)

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_dir, min_pixels=min_pixels, max_pixels=max_pixels, padding_side='left'
        )  # left padding: <|endoftext|> 151644
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side='left',
            use_fast=use_flash_attention,
        )
    elif "ViT" in model_name:
        model, processor = clip.load(model_name, device=device, download_root="/raid_sdd/lyy/hf/clip")
        tokenizer = None
    elif "llava" in model_name:
        model_dir = MODEL_NAME_TO_PATH[model_name]
        MODEL_CLASS = LlavaForConditionalGeneration if "llava1_5" in model_name else LlavaNextForConditionalGeneration
        model = MODEL_CLASS.from_pretrained(
            model_dir, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2" if use_flash_attention else None,
        ).to(device)
        try:
            processor = AutoProcessor.from_pretrained(model_dir, padding_side='left', use_fast=True)
        except:
            processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side='left',
            use_fast=True,
        )
    elif "intern" in model_name:
        model_dir = MODEL_NAME_TO_PATH[model_name]
        model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side='left',
            use_fast=True,
            trust_remote_code=True,
        )
        processor = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side='left',
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        raise ValueError("Invalid model name.")
    
    return model, processor, tokenizer

def get_model_inputs(model_name, processor, vision_process_func, image_paths, prompts):
    
    image_paths = map(str, image_paths)

    if any(name in model_name for name in ["qwen"]):
        batch_messages = []
        for img_path, prompt in zip(image_paths, prompts):
            batch_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt},
                    ]
                }
            ])
        texts = [processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ) for msg in batch_messages]
        
        all_images = []
        for msg in batch_messages:
            images, _ = vision_process_func(msg)
            if images:
                all_images.extend(images)
            
        inputs = processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    elif any(name in model_name for name in ["llava"]):
        batch_messages = []
        for img_path, prompt in zip(image_paths, prompts):
            batch_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ]
                }
            ])
        texts = [processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ) for msg in batch_messages]
        
        all_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        inputs = processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    elif any(name in model_name for name in ["intern"]):
        # question_length = [len(prompt) for prompt in prompts]
        block_wh_list = None
        if vision_process_func.__name__ == "load_image_intern_return_wh":
            pixel_values = [vision_process_func(image_path)[0].to(torch.bfloat16) for image_path in image_paths]
            block_wh_list = [vision_process_func(image_path)[1] for image_path in image_paths]
        else:
            pixel_values = [vision_process_func(image_path).to(torch.bfloat16) for image_path in image_paths]
        num_patches_list = [pixel_value.size(0) for pixel_value in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        inputs = {
            "questions": prompts,
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "block_wh_list": block_wh_list
            # "question_length": question_length,
        }
    else:
        raise ValueError("Model name not supported.")
    
    return inputs

def draw_bbox(
    image_path,
    bboxes=None,
    save_dir="",
    obj_names=None
):
    image_id = os.path.basename(image_path).split(".")[0]
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for name, bbox in bboxes.items():
        if name not in obj_names:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1-25, y1), name, fill="lime")
    # save
    image_name = os.path.basename(image_path).split(".")[0]
    save_path = os.path.join(save_dir, f"{image_name}_objects_bbox.png")
    image.save(save_path)
    print(f"save_path: {save_path}")

def draw_all_bboxes(
    image_path, 
    bbox_path= data_dir / "GQA/objects", 
    link_path= data_dir / "GQA/objects/gqa_objects_info.json",
    save_dir=""
    ):
    
    with open(link_path, "r") as f:
        data = json.load(f)
    
    image_id = os.path.basename(image_path).split(".")[0]
    image_info = data[image_id]
    print(f"image_info: {image_info}")
    
    file_num = image_info["file"]
    idx = image_info["idx"]
    object_num = image_info["objectsNum"]
    bbox_path = os.path.join(bbox_path, f"gqa_objects_{file_num}.h5")
    
    import h5py
    # https://docs.h5py.org/en/stable/quick.html
    with h5py.File(bbox_path, 'r') as f:
        bbox_file = f["bboxes"]
        data = bbox_file[:]
        # print("data", data)
        # print("data.shape", data.shape)  # (ImagesNum, 100, 4)
        bboxes = data[idx]
    
    # draw bbox
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    # save
    image_name = os.path.basename(image_path).split(".")[0]
    save_path = os.path.join(save_dir, f"{image_name}_bbox.png")
    image.save(save_path)
    print(f"save_path: {save_path}")

def get_scene_info(image_path):
    train_scene_path = data_dir / "GQA/scene/train_sceneGraphs.json"
    val_scene_path = data_dir / "GQA/scene/val_sceneGraphs.json"
    train_scene_data = json.load(open(train_scene_path, "r"))
    val_scene_data = json.load(open(val_scene_path, "r"))
    
    scene_data = None
    image_id = os.path.basename(image_path).split(".")[0]
    if image_id in train_scene_data:
        scene_data = train_scene_data[image_id]
    elif image_id in val_scene_data:
        scene_data = val_scene_data[image_id]
    else:
        # raise ValueError("Invalid image id.")
        return None
    objects = scene_data["objects"]
    # obj_ids = list(objects.keys())
    obj_names = [obj["name"] for obj in objects.values()]
    print(f"obj_names: {obj_names}")
    obj_bboxes = [(obj["x"], obj["y"], obj["w"], obj["h"]) for obj in objects.values()]
    obj_bboxes = [(x, y, x+w, y+h) for x, y, w, h in obj_bboxes]  # bbox: x1, y1, x2, y2
    obj_bboxes = {name: bbox for name, bbox in zip(obj_names, obj_bboxes)}
    
    return obj_bboxes

def generate_batch_responses(
    model,
    processor,
    dataset_name,
    batch_size=8,
    max_new_tokens=10,
    save_path="",
    data_num=1000, 
    random=False,
):
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        model_name = "qwen2_5_vl"
    elif isinstance(model, Qwen2VLForConditionalGeneration):
        model_name = "qwen2_vl"
    elif isinstance(model, LlavaForConditionalGeneration) or isinstance(model, LlavaNextForConditionalGeneration):
        model_name = "llava"
    elif "Intern" in str(type(model)):
        model_name = "intern"
    dataloader = load_data(dataset_name, model_name, processor, data_num=data_num, random=random, batch_size=batch_size)
    data_num = batch_size * len(dataloader)
    
    if "coco" in dataset_name:
        max_new_tokens = 256  # coco caption generation is longer

    preds = []
    all_samples = []
    for idx, batch in enumerate(tqdm(dataloader)):
        
        inputs, samples = batch
        all_samples.extend(samples)
        # input_0 = inputs.input_ids[0]
        # text = processor.decode(
        #     input_0,
        #     skip_special_tokens=False,
        #     clean_up_tokenization_spaces=False
        # )
        # print(f"idx: {idx}, text: {text}")
        # import pdb; pdb.set_trace()
        
        outputs = []
        if "intern" in model_name:
            responses = model.batch_chat(
                processor, 
                inputs["pixel_values"].to(model.device),
                num_patches_list=inputs["num_patches_list"],
                questions=inputs["questions"],
                generation_config=dict(max_new_tokens=5, do_sample=False),
            )
            for i in range(len(responses)):
                outputs.append(responses[i])
        else:
            inputs.to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            for i in range(len(inputs.input_ids)):
                output_ids = generated_ids[i][len(inputs.input_ids[i]):]
                outputs.append(processor.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ))
        
        for out, sample in zip(outputs, samples):
            if "coco" in dataset_name:
                print("----"*20)
                print(f"[caption_pred]: {out}")
                save_dir = save_path.rsplit("/", 1)[0]
                image_id = str(sample["image_id"])
                sample_save_dir = os.path.join(save_dir, f"cases_{data_num}", image_id)
                os.makedirs(sample_save_dir, exist_ok=True)
                shutil.copy(sample["image"], sample_save_dir)
                caption_save_path = os.path.join(sample_save_dir, f"{image_id}_caption.jsonl")
                with jsonlines.open(caption_save_path, "w") as f:
                    f.write({
                        "image_id": image_id,
                        "caption_gold": sample["caption_gold"],
                        "caption": out,
                    })
            elif "mme" in dataset_name:
                category = sample["category"]
                res_dir = save_path.split('/')[:-1]
                res_dir = '/'.join(res_dir)
                os.makedirs(res_dir, exist_ok=True)
                new_save_path = os.path.join(res_dir, f"{category}.txt")
                with open(new_save_path, "a") as f:
                    # write: Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
                    image_name = sample["image_name"]
                    qn = sample["question"]
                    gt = sample["answer"]
                    res = out.replace('\n', ' ')
                    f.write(f"{image_name}\t{qn}\t{gt}\t{res}\n")
            else:
                # print(f"image: {samples['image_path']}, choices: {samples['caption_options']}, pred: {out}")
                if "vqa" in dataset_name:
                    print(f"pred: [{out}], gold_ans: {sample["answers"]}")
                else:
                    print(f"pred: [{out}], gold_ans: {sample["answer"]}")
                
        preds.extend(outputs)  
        
    with jsonlines.open(save_path, "w") as f:
        for pred, sample in zip(preds, all_samples):
            if "coco" in dataset_name:
                sample.update({"caption": pred})
            elif dataset_name in ["vqa"]:
                sample.update({"answer": pred})
            elif "mme" in dataset_name:
                continue
            else:
                sample.update({"pred": pred})
            f.write(sample)
               
    return preds

def cluster(data, method, metric, k=None, max_k=10):
    
    if not k:
        sse = []
        silhouette_scores = []
        cluster_results = []
        for k in range(2, max_k+1):
            if method == 'kmeans':
                cluster = KMeans(n_clusters=k, random_state=41)
            elif method == 'dbscan':
                cluster = DBSCAN(eps=0.5, min_samples=5)
            elif method == 'agg':
                cluster = AgglomerativeClustering(n_clusters=k)
            elif method == 'gmm':
                cluster = GaussianMixture(n_components=k, random_state=41)
            else:
                raise ValueError("Unknown method")
            
            labels = cluster.fit_predict(data)
            cluster_results.append(labels)
            sse.append(cluster.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))
        
        if metric == 'sse':
            best_k = np.argmin(sse) + 2
        else:
            best_k = np.argmax(silhouette_scores) + 2
        best_results = cluster_results[best_k-2]
        print(f"Best k is {best_k} under the {metric} metric.")
    else:
        if method == 'kmeans':
            cluster = KMeans(n_clusters=k, random_state=41)
        elif method == 'dbscan':
            cluster = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'agg':
            cluster = AgglomerativeClustering(n_clusters=k)
        elif method == 'gmm':
            cluster = GaussianMixture(n_components=k, random_state=41)
        else:
            raise ValueError("Unknown method")
        
        best_results = cluster.fit_predict(data)
    
    
    # plt.plot(range(1, max_k+1), sse, 'bx-')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Sum of Squared Distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()
    
    # plt.plot(range(2, max_k+1), silhouette_scores, 'bx-')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Method For Optimal k')
    # plt.show()

    return best_results

def visualize_clusters(tokens, labels):
    # 使用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    tokens_2d = tsne.fit_transform(tokens)
    
    plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('Token Clusters Visualization')
    plt.show()

def seg_with_activations(
    device, 
    model_name, 
    image_path,
    dim_reduction=True, 
    standarization=True,
    normalization=False,
    original_size=True,
    tsne_dim=3,
):    
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    vit, llm = None, None
    
    if "llava" in model_name:
        vit, llm = model.vision_tower, model.language_model
        vision_process_func = None
    elif "qwen" in model_name:
        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
    elif "intern" in model_name:
        vision_process_func = load_image_intern
    else:
        # raise ValueError("Unsupported model type.")
        vision_process_func = None

    # path
    save_dir = root_dir / "figures/seg_with_activations"
    if standarization and normalization:
        raise ValueError("Cannot apply both standarization and normalization at the same time.")
    if normalization:
        save_dir += "_normalized"
    
    new_save_dir = os.path.join(save_dir, model_name)
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}_tsne_{tsne_dim}")
    os.makedirs(image_save_dir, exist_ok=True)
    shutil.copy(image_path, image_save_dir)
    
    # get image info
    obj_bboxes = get_scene_info(image_path)
    if not obj_bboxes:  # invalid image
        return
    num_object = len(obj_bboxes)
    
    # get ViT outputs
    model_inputs = get_model_inputs(
        model_name=model_name,
        processor=processor,
        vision_process_func=vision_process_func,
        image_paths=[image_path],
        prompts=["describe the image and tell me what is the main object in the image"]
    )
    
    ## forward
    vit_layer_num = 0
    if any(name in model_name for name in ["qwen"]):
        # replace_qwen2_5_vl_test_directions_processor_return_indices()
        replace_qwen2_5_vl_test_seg_vit_output_hidden_states()
        vit_layer_num = model.config.vision_config.depth
    elif any(name in model_name for name in ["llava"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
    elif any(name in model_name for name in ["intern"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
        
    with torch.no_grad():
        inputs = model_inputs
        inputs.to(model.device)
        if "llava" in model_name:
            pixel_values = inputs["pixel_values"]
            output_hidden_states = vit(pixel_values, output_hidden_states=True).hidden_states
        elif "qwen" in model_name:
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            pixel_values = pixel_values.type(vit.dtype)
            _, output_hidden_states = vit(pixel_values, grid_thw=image_grid_thw)
        else:
            pass
        
    # clustering
    output_hidden_states = output_hidden_states[1:]  # remove the embedding layer output
    
    for layer_id in range(vit_layer_num):
        save_path = os.path.join(image_save_dir, f"seg_img_{image_name}_layer_{layer_id}.svg")
        activations = output_hidden_states[layer_id][0].cpu().float()
        if "llava" in model_name:
            activations = activations[1:]  # remove cls token
        else:
            pass
        
        # normalization
        if normalization:
            activations = normalize(activations, norm='l2', axis=1)

        # standarization
        if standarization:
            print(f"activations shape: {activations.shape}")
            activations = StandardScaler().fit_transform(activations)
        # dimensionality reduction
        if dim_reduction:
            # pca = PCA(n_components=0.95)
            # activations = pca.fit_transform(activations)
            # tsne
            tsne = TSNE(n_components=3, random_state=41, perplexity=30, n_iter=1000)
            activations = tsne.fit_transform(activations)
        # best_labels = cluster(activations, method=cluster_method, metric=cluster_metric, k=cluster_k)
        kmeans = KMeans(n_clusters=num_object, random_state=14, n_init='auto')
        kmeans.fit(activations)
        best_labels = kmeans.labels_
        
        # print(type(activations), activations.shape)
        # print(f"len(best_labels): {len(best_labels)}")
        # print(f"size: {image_grid_thw[0].cpu().numpy()}, {image_grid_thw[0].cpu().numpy()[0]}")
        # import pdb; pdb.set_trace()
        # visualize_clusters(activations, best_results)
        
        # draw the semantic segmentation map, using the clustering results
        # 1. get the unique labels
        unique_labels = np.unique(best_labels)
        
        # 2. create a color map
        color_map = {}
        for i, label in enumerate(unique_labels):
            color_map[label] = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
            
        # 3. create a blank image
        # the original_height/width here is the h/w of the resized image, while in `test_seg_map.py` it is the original image size
        if original_size:
            if "llava" in model_name:
                original_height, original_width = inputs["pixel_values"][0].shape[1:]  # after resize (336*336)
                scale = model.config.vision_config.patch_size
                width, height = original_width // scale, original_height // scale  # after patching
            elif "qwen" in model_name:  # after resize & patching, before merging
                image_grid_thw = inputs["image_grid_thw"]
                _, height, width = image_grid_thw[0].cpu().numpy()
                scale = model.config.vision_config.spatial_patch_size
                original_height *= scale  # after resize, before patching
                original_width *= scale
            new_labels = np.array([[None for _ in range(original_width)] for _ in range(original_height)])
            for j in range(len(best_labels)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                x_range = slice(x * scale, (x + 1) * scale)
                y_range = slice(y * scale, (y + 1) * scale)
                # print(f"j: {j}, width: {width}, x: {x}, y: {y},  x_range: {x_range}, y_range: {y_range}")
                for row in range(y_range.start, y_range.stop):
                    for col in range(x_range.start, x_range.stop):
                        new_labels[row][col] = best_labels[j]
            best_labels = new_labels.flatten()
            # seg_map = Image.new("RGB", (original_width, original_height), (0, 0, 0))
            output_width, output_height = original_width, original_height
        else:
            # seg_map = Image.new("RGB", (width, height), (0, 0, 0))
            output_width, output_height = width, height
        
        # 4. draw the segmentation map
        cell_size = 10  # Size of each cell in pixels
        svg_width = output_width * cell_size
        svg_height = output_height * cell_size
        
        dwg = svgwrite.Drawing(
            size=(f"{svg_width}px", f"{svg_height}px"),
            viewBox=(f"0 0 {svg_width} {svg_height}")
        )
        
        # Add white background
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=("100%", "100%"),
            fill="white"
        ))
        
        for j in range(len(best_labels)):
            x = j % original_width if original_size else j % width
            y = j // original_width if original_size else j // width
            x, y = int(x), int(y)
            color = color_map[best_labels[j]]
            
            dwg.add(dwg.rect(
                insert=(x * cell_size, y * cell_size),
                size=(cell_size, cell_size),
                fill=color,
                stroke="none"
            ))
            
            # rgb_color = mcolors.to_rgb(color)
            # int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
            # seg_map.putpixel((x, y), int_color)
            
        # 5. save the segmentation map
        # seg_map.save(save_path)
        dwg.saveas(save_path)
        print(f"image: {image_path}, layer: {layer_id-1} Finished!")

def seg_with_self_attention(
    device, 
    model_name, 
    image_path,
    original_size=False,
):    
    # load modelxxx
    model, processor, tokenizer = load_model(model_name, device, use_flash_attention=False)
    vit, llm = None, None
    if "intern" in model_name:
        vision_process_func = load_image_intern
    else:
        vit, llm = model.vision_tower, model.language_model
        # raise ValueError("Unsupported model type.")
        vision_process_func = None

    # path
    save_dir = root_dir / "figures/seg_with_self_attention"
    new_save_dir = os.path.join(save_dir, model_name)
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}")
    os.makedirs(image_save_dir, exist_ok=True)
        
    ## forward ViT
    vit_layer_num = 0
    if any(name in model_name for name in ["llava"]):
        replace_llava1_5_processor_forward_return_image_size()
        vit_layer_num = model.config.vision_config.num_hidden_layers
    elif any(name in model_name for name in ["intern"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
        
    ## Preparation for inference
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path).convert('RGB')
    inputs, image_size = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs.to(model.device)
    
    all_attentions = vit(
        pixel_values=inputs["pixel_values"],
        output_attentions=True,
    ).attentions  # layer_num * (batch_size, num_heads, seq_len, seq_len)
    
    for layer_id in range(vit_layer_num):
        if layer_id != vit_layer_num - 1:
            continue    
        
        # clustering using self attention
        # 1. get attention scores
        attention_scores = all_attentions[layer_id][0]  # the first sample: (num_heads, len, len), len = 1([cls]) + width * height
        attention_scores = attention_scores[:, 1:, 1:]  # remove cls token
        
        reduced_attention = torch.sum(attention_scores, dim=0)  # (len, len)
        print(reduced_attention.shape)
        # print(reduced_attention[2], torch.sum(reduced_attention[2]))
        # import pdb; pdb.set_trace()
        
        # 2. compute distance matrix
        # for two image tokens, the distance between them is the sum of their attention scores to each others, i.e. attn(i,j) + attn(j,i)
        seq_len = reduced_attention.shape[0]
        distance_matrix = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                distance_matrix[i][j] = 1 / (reduced_attention[i][j].item() + reduced_attention[j][i].item()) + 0.0000001
        print(distance_matrix)
        print(distance_matrix.shape)
                
        # 3. clustering using DBSCAN
        Z = linkage(distance_matrix, method="average")
        last_10 = Z[-10:, 2]
        diffs = np.diff(last_10)
        max_diff_index = np.argmax(diffs)
        threshold = last_10[max_diff_index]
    
        # 划分簇
        labels = fcluster(Z, t=threshold, criterion='distance')
        if len(np.unique(labels)) == 1:
            k = min(5, distance_matrix.shape[0]//2)
            labels = fcluster(Z, t=k, criterion='maxclust')
        
        print(labels)
        
        # draw the clustering map using self attention
        # 1. get the unique labels
        unique_labels = np.unique(labels)
        print(f"unique_labels: {unique_labels}")
        
        # 2. create a color map
        color_map = {}
        for i, label in enumerate(unique_labels):
            color_map[label] = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
        # 3. create a blank image
        height, width = image_size  # after patching
        print(f"image_size: {image_size}")
        # import pdb; pdb.set_trace()
        
        if original_size:
            scale = model.config.vision_config.patch_size
            original_width = width * scale
            original_height = height * scale
            new_labels = np.array([[None for _ in range(original_width)] for _ in range(original_height)])
            for j in range(len(labels)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                x_range = slice((x - 1) * scale, x * scale)
                y_range = slice(y * scale, (y + 1) * scale)
                # print(f"j: {j}, width: {width}, x: {x}, y: {y},  x_range: {x_range}, y_range: {y_range}")
                for row in range(y_range.start, y_range.stop):
                    for col in range(x_range.start, x_range.stop):
                        new_labels[row][col] = labels[j]
            labels = new_labels.flatten()
            # print(text_tokens)
            seg_map = Image.new("RGB", (original_width, original_height), (0, 0, 0))
        else:
            seg_map = Image.new("RGB", (width, height), (0, 0, 0))
        
        # 4. draw the segmentation map
        for j in range(len(labels)):
            x = j % original_width if original_size else j % width
            y = j // original_width if original_size else j // width
            x, y = int(x), int(y)
            color = color_map[labels[j]]
            rgb_color = mcolors.to_rgb(color)
            int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
            seg_map.putpixel((x, y), int_color)
        # 5. save the segmentation map
        if original_size:
            save_path = os.path.join(image_save_dir, f"seg_img-origin_size-{image_name}-vit_{layer_id}-unembed_llm.png")
        else:
            save_path = os.path.join(image_save_dir, f"seg_img_{image_name}-vit_{layer_id}-unembed_llm.png")
        seg_map.save(save_path)
    
        print(f"image: {image_path}, layer: {layer_id-1} Finished!")

def seg_with_unembedding_tokens_qwen_old(
    device, 
    model_name, 
    image_path,
    write_labels=False,
):
    
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    vit, llm = None, None
    if "qwen" in model_name:
        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
    elif "intern" in model_name:
        vision_process_func = load_image_intern
    else:
        # raise ValueError("Unsupported model type.")
        vision_process_func = None

    # path
    save_dir = root_dir / "/figures/seg_with_unembedding_tokens"
    new_save_dir = os.path.join(save_dir, model_name)
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}")
    os.makedirs(image_save_dir, exist_ok=True)
    
    # get ViT outputs
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path, 
                },
                {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
            ],
        }
    ]

    ## Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = vision_process_func(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs.to(model.device)
    
    ## forward
    vit_layer_num = 0
    discard_vit_layers_func = None
    if any(name in model_name for name in ["qwen"]):
        # replace_qwen2_5_vl_test_directions_processor_return_indices()  # for batch input
        # replace_qwen2_5_vl_test_seg_vit_output_hidden_states()  # for obtain of ViT outputs
        discard_vit_layers_func = replace_qwen2_5_vl_test_directions_vit_discard_layers
        replace_qwen2_5_vl_return_image_mask()
        vit_layer_num = model.config.vision_config.depth
    elif any(name in model_name for name in ["llava"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
    elif any(name in model_name for name in ["intern"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
        
    # with torch.no_grad():
        # pixel_values = inputs["pixel_values"]
        # image_grid_thw = inputs["image_grid_thw"]
        # pixel_values = pixel_values.type(vit.dtype)
        # _, output_hidden_states = vit(pixel_values, grid_thw=image_grid_thw)
        
        
    # segmentation based on text       
    # if any(name in model_name for name in ["qwen"]):
    #     output_hidden_states = output_hidden_states[:-1]  # exclude the merger
    
    for layer_id in range(vit_layer_num):
        if layer_id != vit_layer_num - 1:
            continue
        
        text_tokens = []
        unembed_llm_layer = -1
        with torch.no_grad():
            layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
            if "intern" in model_name:
                discard_vit_layers_func(model, layer_ids_to_delete)
            else:
                discard_vit_layers_func(layer_ids_to_delete)
            outputs, image_mask = model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # activations = outputs.logits
            activations = model.lm_head(outputs.hidden_states[unembed_llm_layer])  # maybe the last LLM layer performs the best, maybe not
        
            text_ids = torch.argmax(activations, dim=-1)  # (seq_len)
            # import pdb; pdb.set_trace()
            text_ids = text_ids[0]
            all_tokens = [processor.tokenizer.decode(text_id) for text_id in text_ids]
            # print(all_tokens)
            image_mask = image_mask[0]
            assert len(all_tokens) == len(image_mask)
            text_tokens = [all_tokens[i] for i in range(len(all_tokens)) if image_mask[i]]
            print(text_tokens)
            
        
        # if apply_merger:
        #     activations = vit.merger(output_hidden_states[layer_id])
        #     activations = activations.cpu().float()
        # else:
        #     activations = output_hidden_states[layer_id]
        # logits = llm.lm_head(activations)
        # text_ids = torch.argmax(logits, dim=-1)
        # text_tokens = [processor.tokenizer.convert_ids_to_tokens(text_id) for text_id in text_ids]
        
        
        # draw the semantic segmentation map, using the clustering results
        # 1. get the unique labels
        unique_labels = np.unique(text_tokens)
        print(f"unique_labels: {unique_labels}")
        
        if not write_labels:
            # 2. create a color map
            color_map = {}
            for i, label in enumerate(unique_labels):
                color_map[label] = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
            # 3. create a blank image
            image_grid_thw = inputs["image_grid_thw"]
            time, height, width = image_grid_thw[0].cpu().numpy()
            # if apply_merger:
            #     height = height // model.config.vision_config.spatial_merge_size
            #     width  = width // model.config.vision_config.spatial_merge_size
            height = height // model.config.vision_config.spatial_merge_size
            width  = width // model.config.vision_config.spatial_merge_size
            seg_map = Image.new("RGB", (width, height), (0, 0, 0))
            # 4. draw the segmentation map
            for j in range(len(text_tokens)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                color = color_map[text_tokens[j]]
                rgb_color = mcolors.to_rgb(color)
                int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
                seg_map.putpixel((x, y), int_color)
            # 5. save the segmentation map
            save_path = os.path.join(image_save_dir, f"seg_img_{image_name}-vit_{layer_id}-unembed_llm_{unembed_llm_layer}.png")
            seg_map.save(save_path)
        else:
            # create a blank image
            image_grid_thw = inputs["image_grid_thw"]
            time, height, width = image_grid_thw[0].cpu().numpy()
            height = height // model.config.vision_config.spatial_merge_size
            width = width // model.config.vision_config.spatial_merge_size
            cell_width = 50
            table_width, table_height = width * cell_width, height * cell_width
            cell_width, cell_height = cell_width, cell_width

            seg_map = Image.new("RGB", (table_width, table_height), "white")
            draw = ImageDraw.Draw(seg_map)

            font_path = "/raid_sdd/lyy/font/SimHei.ttf"
            try:
                font = ImageFont.truetype(font_path, size=15)
            except IOError:
                font = None

            # draw lines
            for x in range(0, table_width, cell_width):
                draw.line((x, 0, x, table_height), fill="black")
            for y in range(0, table_height, cell_height):
                draw.line((0, y, table_width, y), fill="black")

            # draw text
            for j in range(len(text_tokens)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                text = text_tokens[j]
                left, top, right, bottom = draw.textbbox((0, 0), text)
                text_width, text_height = right - left, top - bottom
                
                text_x = x * cell_width + (cell_width - text_width) / 2
                text_y = y * cell_height + (cell_height - text_height) / 2
                
                draw.text((text_x, text_y), text, fill="black", font=font)
            
            save_path = os.path.join(image_save_dir, f"seg_token_img_{image_name}-vit_{layer_id}-unembed_llm_{unembed_llm_layer}.png")
            seg_map.save(save_path)
            print(f"image: {image_path}, layer: {layer_id-1} Finished!")

# ===========================================================================
def select_unembedding_layer(
    device, 
    model_name, 
    image_path,
    check_emergence=True,
):
    # path
    save_dir = root_dir / "figures/llm_layer_logit_lens"
    new_save_dir = os.path.join(save_dir, model_name)
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(new_save_dir, f"img_{image_name}")
    os.makedirs(image_save_dir, exist_ok=True)
    shutil.copy(image_path, image_save_dir)
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None
            replace_llava1_5_return_image_mask()
            replace_llava1_5_processor_forward_return_image_size()
        elif "qwen" in model_name:
            vision_process_func = process_vision_info_qwen
            if "qwen2_5" in model_name:
                replace_qwen2_5_vl_return_image_mask()
            elif "qwen2" in model_name:
                replace_qwen2_vl_return_image_mask()
        elif "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            pass
            
    ## Preparation for inference
    if check_emergence:
        obj_bboxes = get_scene_info(image_path)
    
    model_inputs = get_model_inputs(model_name, processor, vision_process_func, [image_path], ["describe the image and tell me what is the main object in the image"])
    if "llava" in model_name:
        inputs, image_size = model_inputs
    elif "qwen" in model_name:
        inputs = model_inputs
    else:
        pass
    inputs.to(model.device)
    
    with torch.no_grad():
        outputs, image_mask = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )
    
    # start
    image_mask = image_mask[0].squeeze(-1)
    all_text_tokens = []
    if "llava" in model_name:
        llm_layer_num = model.language_model.config.num_hidden_layers
    elif "qwen" in model_name:
        llm_layer_num = model.config.num_hidden_layers
    else:
        raise ValueError("Unsupported model type.")
    
    for layer_id in range(llm_layer_num):
        
        # get activations
        if "llava" in model_name:
            activations = llm.lm_head(outputs.hidden_states[layer_id + 1])
        elif "qwen" in model_name:
            activations = model.lm_head(outputs.hidden_states[layer_id + 1])
        else:
            pass
        
        # get image size
        if "llava" in model_name:
            height, width = image_size  # after patching
        elif "qwen" in model_name:
            image_grid_thw = inputs["image_grid_thw"]
            time, height, width = image_grid_thw[0].cpu().numpy()
            # set to int32
            height, width = int(height), int(width)
            height = height // model.config.vision_config.spatial_merge_size
            width = width // model.config.vision_config.spatial_merge_size
        else:
            pass
        # print(f"image_size: {image_size}")
        # import pdb; pdb.set_trace()
        cell_width = 50
        table_width, table_height = width * cell_width, height * cell_width
        cell_width, cell_height = cell_width, cell_width
        cell_size = cell_width 
        svg_width = table_width
        svg_height = table_height
        
        
        # check norm (to see if register tokens exist)
        # compute the norm of the activations
        norms = torch.norm(activations[0], dim=-1)  # (seq_len)
        norms = [norms[i] for i in range(len(norms)) if image_mask[i]]
        norms = torch.tensor(norms, device=device)
        norms = (norms - norms.min()) / (norms.max() - norms.min())
        
        # draw norm heat map
        norm_save_dir = os.path.join(image_save_dir, "norms")
        os.makedirs(norm_save_dir, exist_ok=True)
        norm_save_path = os.path.join(norm_save_dir, f"norm_llm_{layer_id}.svg")
        dwg = svgwrite.Drawing(
            size=(f"{svg_width}px", f"{svg_height}px"),
            viewBox=(f"0 0 {svg_width} {svg_height}")
        )
        # draw each cell
        for j in range(len(norms)):
            x = j % width
            y = j // width
            x, y = int(x), int(y)
            norm_val = norms[j].item()
            color = plt.cm.viridis(norm_val)
            color_hex = mcolors.to_hex(color)
            dwg.add(dwg.rect(
                insert=(x * cell_size, y * cell_size),
                size=(cell_size, cell_size),
                fill=color_hex,
                stroke="none"
            ))
        # save the norm heat map
        dwg.saveas(norm_save_path)  
        
        
        # get text tokens
        # text_ids = torch.argmax(activations[0], dim=-1)  # (seq_len)
        text_vals_top_3, text_ids_top_3 = torch.topk(activations[0], k=3, dim=-1)  # (seq_len, top_k)
        text_ids = text_ids_top_3[:, 0]
        text_ids_second = text_ids_top_3[:, 1]
        text_ids_third = text_ids_top_3[:, 2]
        
        all_tokens = [processor.tokenizer.decode(text_id) for text_id in text_ids]
        all_tokens_second = [processor.tokenizer.decode(text_id) for text_id in text_ids_second]
        all_tokens_third = [processor.tokenizer.decode(text_id) for text_id in text_ids_third]
        
        text_tokens = [all_tokens[i] for i in range(len(all_tokens)) if image_mask[i]]
        text_tokens_second = [all_tokens_second[i] for i in range(len(all_tokens_second)) if image_mask[i]]
        text_tokens_third = [all_tokens_third[i] for i in range(len(all_tokens_third)) if image_mask[i]]
        
        all_text_tokens.append(text_tokens)
        assert len(all_tokens) == len(image_mask)
        
        # draw text token map
        if True:
            for probs_id in range(3):

                # Create SVG canvas
                dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"))
                
                # Add white background
                dwg.add(dwg.rect(
                    insert=(0, 0),
                    size=(f"{svg_width}px", f"{svg_height}px"),
                    fill="white"
                ))
                
                # Draw grid
                for x in range(0, svg_width + 1, cell_size):
                    dwg.add(dwg.line(start=(x, 0), end=(x, svg_height), stroke="black", stroke_width=1))
                for y in range(0, svg_height + 1, cell_size):
                    dwg.add(dwg.line(start=(0, y), end=(svg_width, y), stroke="black", stroke_width=1))
                
                font_path = "/raid_sdd/lyy/font/SimHei.ttf"
                try:
                    font = ImageFont.truetype(font_path, size=15)
                except IOError:
                    font = None
                    
                # Add text
                text_tokens_to_draw = None
                if probs_id == 0:
                    text_tokens_to_draw = text_tokens
                elif probs_id == 1:
                    text_tokens_to_draw = text_tokens_second
                elif probs_id == 2:
                    text_tokens_to_draw = text_tokens_third
                else:
                    pass
                for j, text in enumerate(text_tokens_to_draw):
                    x = j % width
                    y = j // width
                    x_pos = x * cell_size + cell_size / 2
                    y_pos = y * cell_size + cell_size / 2
                    
                    dwg.add(dwg.text(
                        text,
                        insert=(x_pos, y_pos),
                        fill="black",
                        font_family=font,  # Use your font
                        font_size="12px",
                        text_anchor="middle",
                        dominant_baseline="middle"
                    ))
                
                # Save SVG
                text_token_save_dir = os.path.join(image_save_dir, "text_tokens")
                os.makedirs(text_token_save_dir, exist_ok=True)            
                save_path = os.path.join(text_token_save_dir, f"token-llm_{layer_id}-probs_{probs_id}.svg")
                dwg.saveas(save_path)
            
    # check emergence
    if check_emergence:
        image_emergence = []
        for layer_id, text_tokens in enumerate(all_text_tokens):
            num_tokens = 0
            counter = Counter(text_tokens)
            for obj_name in obj_bboxes.keys():
                for token, cnt in counter.items():
                    if token in obj_name:
                        num_tokens += cnt
            image_emergence.append(num_tokens / len(text_tokens))
        
        # plot emergence
        plt.figure()
        llm_layer_ids = list(range(1, llm_layer_num + 1))
        plt.plot(
            llm_layer_ids,
            image_emergence,
            color="blue",
            label='Ratio of the emerging tokens',
            alpha=0.55, 
            marker='o',
            markersize=5, 
            markeredgecolor='blue',
            markeredgewidth=1.5
        )
        
        plt.xlabel("Layer ID", loc="right")
        plt.ylabel("Image Emergence")
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        plt.legend()
        
        plt.savefig(os.path.join(image_save_dir, "select_unembedding_layer.pdf"), bbox_inches='tight')

def select_unembedding_layer_batch(
    device, 
    model_name, 
    batch_size=8,
    bi_dir_att=False,  # whether to use bidirectional attention at image positions at the LLM stage
):
    # path
    save_dir = root_dir / "figures/seg_with_unembedding_tokens"
    new_save_dir = os.path.join(save_dir, model_name)
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None
        
        if any(name in model_name for name in ["llava"]):
            replace_llava1_5_return_image_mask()
            if bi_dir_att:
                replace_llava1_5_llm_image_bidirectional_attention_and_return_image_mask()
        
    ## Preparation for inference
    # dataloader = load_data("GQA", model_name, processor, data_num=100, random=False, batch_size=batch_size, QA_mode=False)
    
    dataset = GQASquareImages(root_dir / "test_figs/gqa", data_num=500)
    data_collator = GQACollator(processor, vision_process_func=vision_process_func, model_name=model_name, QA_mode=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    
    llm_layer_num = model.language_model.config.num_hidden_layers
    all_image_emergence = [[] for _ in range(llm_layer_num)]
    for idx, batch in enumerate(tqdm(dataloader)):
        
        inputs, samples = batch
        inputs.to(model.device)
        # image_ids = [os.path.basename(sample["image"]).split(".")[0] for sample in samples]
        image_bboxes = [get_scene_info(sample["image"]) for sample in samples]
            
        with torch.no_grad():
            outputs, image_masks = model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
            )
        
        for layer_id in range(llm_layer_num):
            activations = llm.lm_head(outputs.hidden_states[layer_id])
            all_text_ids = torch.argmax(activations, dim=-1)  # (bsz, seq_len)
            
            layer_image_emergence = []  # a batch of image emergence in this layer
            for batch_id in range(len(samples)):
                # get text tokens
                text_ids = all_text_ids[batch_id]
                text_tokens = [processor.tokenizer.decode(text_id) for text_id in text_ids]
                image_mask = image_masks[batch_id].squeeze(-1)
                # print(text_tokens)
                # print(image_mask)
                assert len(text_tokens) == len(image_mask)
                text_tokens = [text_tokens[i] for i in range(len(text_tokens)) if image_mask[i]]
            
                # check emergence
                num_tokens = 0
                counter = Counter(text_tokens)
                for obj_name in image_bboxes[batch_id].keys():
                    for token, cnt in counter.items():
                        if f" {token}" in f" {obj_name}":
                            num_tokens += cnt
                layer_image_emergence.append(num_tokens / len(text_tokens))
            all_image_emergence[layer_id].extend(layer_image_emergence)
            
    all_image_emergence = [np.mean(image_emergence) for image_emergence in all_image_emergence]
    std_error = np.std(all_image_emergence) / np.sqrt(len(all_image_emergence))
    
    # plot emergence
    plt.figure()
    llm_layer_ids = list(range(1, llm_layer_num + 1))
    plt.plot(
        llm_layer_ids,
        all_image_emergence,
        color="blue",
        label='Ratio of the emerging tokens',
        alpha=0.55, 
        marker='o',
        markersize=5, 
        markeredgecolor='blue',
        markeredgewidth=1.5
    )
    plt.fill_between(
        llm_layer_ids, 
        all_image_emergence - std_error, 
        all_image_emergence + std_error, 
        color='b', 
        alpha=0.1, 
        # label=f'Standard Error for "{labels[i]}"'
    )
    
    plt.xlabel("Layer ID", loc="right")
    plt.ylabel("Image Emergence")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    plt.legend()
    
    fig_path = os.path.join(new_save_dir, "select_unembedding_layer.pdf")
    if bi_dir_att:
        fig_path = os.path.join(new_save_dir, "select_unembedding_layer_bi_dir.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    
    data_path = os.path.join(new_save_dir, f"select_unembedding_layer_batch.jsonl")
    if bi_dir_att:
        data_path = os.path.join(new_save_dir, f"select_unembedding_layer_batch_bi_dir.jsonl")
    with jsonlines.open(data_path, "w") as f:
        for layer_id, image_emergence in enumerate(all_image_emergence):
            f.write({
                "layer_id": layer_id,
                "image_emergence": image_emergence,
            })
    
def seg_with_unembedding_tokens_qwen(
    device, 
    model_name, 
    image_path,
    original_size=False,
    unembed_llm_layer=-1,
):
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    vit, llm = model.visual, model.model
    vision_process_func = process_vision_info_qwen

    # path
    save_dir = root_dir / "figures/seg_with_unembedding_tokens"
    new_save_dir = os.path.join(save_dir, model_name)
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}")
    os.makedirs(image_save_dir, exist_ok=True)
    shutil.copy(image_path, image_save_dir)
    
    # get ViT outputs
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path, 
                },
                {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
            ],
        }
    ]

    
    ## forward
    discard_vit_layers_func = replace_qwen2_5_vl_test_directions_vit_discard_layers
    replace_qwen2_5_vl_return_image_mask()
    vit_layer_num = model.config.vision_config.depth
        
    ## Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = vision_process_func(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs.to(model.device)
        
    
    for layer_id in range(vit_layer_num):
        # if layer_id != vit_layer_num - 1:
        #     continue
        
        # get text tokens
        text_tokens = []
        with torch.no_grad():
            layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
            discard_vit_layers_func(layer_ids_to_delete)
            outputs, image_mask = model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # activations = outputs.logits
            activations = model.lm_head(outputs.hidden_states[unembed_llm_layer])  # maybe the last LLM layer performs the best, maybe not
        
            text_ids = torch.argmax(activations, dim=-1)  # (seq_len)
            # import pdb; pdb.set_trace()
            text_ids = text_ids[0]
            all_tokens = [processor.tokenizer.decode(text_id) for text_id in text_ids]
            # print(all_tokens)
            
            image_mask = image_mask[0].squeeze(-1)
            # print(image_mask)
            # print(image_mask.shape)
            # import pdb; pdb.set_trace()
            assert len(all_tokens) == len(image_mask)
            
            text_tokens = [all_tokens[i] for i in range(len(all_tokens)) if image_mask[i]]
        
        # ==============================================================================
        # draw original tokens map
        # create a blank image
        image_grid_thw = inputs["image_grid_thw"]   # after patching
        time, height, width = image_grid_thw[0].cpu().numpy()
        height = height // model.config.vision_config.spatial_merge_size
        width  = width // model.config.vision_config.spatial_merge_size
        # import pdb; pdb.set_trace()
        cell_width = 50
        table_width, table_height = width * cell_width, height * cell_width
        cell_width, cell_height = cell_width, cell_width

        seg_map = Image.new("RGB", (table_width, table_height), "white")
        draw = ImageDraw.Draw(seg_map)

        font_path = "/raid_sdd/lyy/font/SimHei.ttf"
        try:
            font = ImageFont.truetype(font_path, size=15)
        except IOError:
            font = None

        # draw lines
        for x in range(0, table_width, cell_width):
            draw.line((x, 0, x, table_height), fill="black")
        for y in range(0, table_height, cell_height):
            draw.line((0, y, table_width, y), fill="black")

        # draw text
        for j in range(len(text_tokens)):
            x = j % width
            y = j // width
            x, y = int(x), int(y)
            text = text_tokens[j]
            left, top, right, bottom = draw.textbbox((0, 0), text)
            text_width, text_height = right - left, top - bottom
            
            text_x = x * cell_width + (cell_width - text_width) / 2
            text_y = y * cell_height + (cell_height - text_height) / 2
            
            draw.text((text_x, text_y), text, fill="black", font=font)
        
        save_path = os.path.join(image_save_dir, f"token-vit_{layer_id}-unembed_llm_{unembed_llm_layer}.pdf")
        seg_map.save(save_path)
        
        # save text tokens
        text_token_path = os.path.join(image_save_dir, f"text_tokens.jsonl")
        with jsonlines.open(text_token_path, "a") as f:
            data = {"image_id": image_name, "processed": False, "vit_layer": layer_id, "unembed_llm_layer": unembed_llm_layer, "text_tokens": text_tokens}
            f.write(data)
        
        # ==============================================================================
        # clean the text tokens
        # 1. get image info
        image_info = None
        with jsonlines.open(root_dir / "test_figs/info_gqa.jsonl", "r") as f:
            for line in f:
                if line["image_id"] in image_path:
                    image_info = line
                    break
        
        # 2. merge text tokens according to the keywords of the objects
        if image_info:
            objects_info = image_info["objects"]
            class_names = image_info["objects"].keys()
            for i in range(len(text_tokens)):
                possible_classes = {class_name: 0 for class_name in class_names}
                for obj_name, obj_info in objects_info.items():
                #     if any(f"_{text_tokens[i]}" in f"_{val}" for val in obj_info["names"]):
                #         text_tokens[i] = obj_name
                #         break
                    for val in obj_info["names"]:
                        if f"_{text_tokens[i]}" in f"_{val}":
                            possible_classes[obj_name] = max(len(text_tokens[i]) / len(val), possible_classes[obj_name])
                if not all([val == 0 for val in possible_classes.values()]):
                    max_class = max(possible_classes, key=possible_classes.get)
                    text_tokens[i] = max_class
                    
            # 3. change the name of the tokens
            # get the unique labels and their counts
            # label_counts = Counter(text_tokens)  # {label: count, ...}
            for i in range(len(text_tokens)):
                if text_tokens[i] in class_names:
                    continue
                # if label_counts[text_tokens[i]] < 5 or text_tokens[i] in []:
                else:
                    x = i % width
                    y = i // width
                    x, y = int(x), int(y)
                    around_tokens = []
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if j == 0 and k == 0:
                                continue
                            x_ = x + j
                            y_ = y + k
                            if 0 <= x_ < width and 0 <= y_ < height:
                                around_token = text_tokens[y_ * width + x_]
                                around_token_dist = abs(j) + abs(k)
                                punctuations = string.punctuation + string.whitespace
                                invalid_names = list(punctuations) + ["\n", "in", "on"]
                                if around_token not in invalid_names:
                                    around_tokens.append((around_token, around_token_dist))
                            else:
                                continue
                    # sort the around tokens by appearance times
                    token_names = [token[0] for token in around_tokens]
                    # case 1: around tokens are all nonsense
                    if not token_names:
                        text_tokens[i] = "BG"
                        continue
                    
                    token_name_counter = Counter(token_names)
                    possible_choices = token_name_counter.most_common(2)  # [("name", count), ...]
                    # case 2: there exists only 1 class with most counts
                    if len(possible_choices) == 1 or (len(possible_choices) > 1 and possible_choices[0][1] > possible_choices[1][1]):
                        text_tokens[i] = possible_choices[0][0] if possible_choices[0][0] in class_names else "others"
                    else:
                        # case 3: choose the name with the shortest distance
                        dist_0 = sum([token[1] for token in around_tokens if token[0] == possible_choices[0][0]])
                        dist_1 = sum([token[1] for token in around_tokens if token[0] == possible_choices[1][0]])
                        if dist_0 < dist_1:
                            text_tokens[i] = possible_choices[0][0] if possible_choices[0][0] in class_names else "others"
                        else:
                            text_tokens[i] = possible_choices[1][0] if possible_choices[1][0] in class_names else "others"
            print(Counter(text_tokens))
        
            # save new text tokens
            text_token_path = os.path.join(image_save_dir, f"text_tokens.jsonl")
            with jsonlines.open(text_token_path, "a") as f:
                data = {"image_id": image_name, "processed": True, "vit_layer": layer_id, "unembed_llm_layer": unembed_llm_layer, "text_tokens": text_tokens}
                f.write(data)
                   
        # =============================================================
        # draw new tokens map
        # create a blank image
        cell_width = 50
        table_width, table_height = width * cell_width, height * cell_width
        cell_width, cell_height = cell_width, cell_width

        seg_map = Image.new("RGB", (table_width, table_height), "white")
        draw = ImageDraw.Draw(seg_map)

        font_path = "/raid_sdd/lyy/font/SimHei.ttf"
        try:
            font = ImageFont.truetype(font_path, size=15)
        except IOError:
            font = None

        # draw lines
        for x in range(0, table_width, cell_width):
            draw.line((x, 0, x, table_height), fill="black")
        for y in range(0, table_height, cell_height):
            draw.line((0, y, table_width, y), fill="black")

        # draw text
        for j in range(len(text_tokens)):
            x = j % width
            y = j // width
            x, y = int(x), int(y)
            text = text_tokens[j]
            left, top, right, bottom = draw.textbbox((0, 0), text)
            text_width, text_height = right - left, top - bottom
            
            text_x = x * cell_width + (cell_width - text_width) / 2
            text_y = y * cell_height + (cell_height - text_height) / 2
            
            draw.text((text_x, text_y), text, fill="black", font=font)
        

        save_path = os.path.join(image_save_dir, f"new_token-vit_{layer_id}-unembed_{unembed_llm_layer}.png")
        seg_map.save(save_path)
        
        # =============================================================
        # draw the semantic segmentation map, using the clustering results
        # 1. get the unique labels
        unique_labels = np.unique(text_tokens)
        print(f"unique_labels: {unique_labels}")
        
        # 2. create a color map
        color_map = {}
        if image_info:
            for i, label in enumerate(unique_labels):
                if label in image_info["objects"] and image_info["objects"][label]["color"] is not None:
                    color_map[label] = image_info["objects"][label]["color"]
                else:
                    # choose a random color other than existing colors
                    color = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
        else:
            for i, label in enumerate(unique_labels):
                color_map[label] = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
            
        # 3. create a blank image        
        if original_size:
            scale = model.config.vision_config.patch_size
            original_width = width * scale
            original_height = height * scale
            new_text_tokens = np.array([[None for _ in range(original_width)] for _ in range(original_height)])
            for j in range(len(text_tokens)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                x_range = slice(x * scale, (x + 1) * scale)
                y_range = slice(y * scale, (y + 1) * scale)
                # print(f"j: {j}, width: {width}, x: {x}, y: {y},  x_range: {x_range}, y_range: {y_range}")
                for row in range(y_range.start, y_range.stop):
                    for col in range(x_range.start, x_range.stop):
                        new_text_tokens[row][col] = text_tokens[j]
            text_tokens = new_text_tokens.flatten()
            # print(text_tokens)
            seg_map = Image.new("RGB", (original_width, original_height), (0, 0, 0))
        else:
            seg_map = Image.new("RGB", (width, height), (0, 0, 0))
        print(text_tokens)
            
        # 4. draw the segmentation map
        for j in range(len(text_tokens)):
            x = j % original_width if original_size else j % width
            y = j // original_width if original_size else j // width
            x, y = int(x), int(y)
            color = color_map[text_tokens[j]]
            rgb_color = mcolors.to_rgb(color)
            int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
            seg_map.putpixel((x, y), int_color)
            
        # 5. save the segmentation map
        if original_size:
            save_path = os.path.join(image_save_dir, f"img-vit_{layer_id}-unembed_{unembed_llm_layer}.png")
        else:
            save_path = os.path.join(image_save_dir, f"img-vit_{layer_id}-unembed_{unembed_llm_layer}.png")
        seg_map.save(save_path)
        
        print(f"image: {image_path}, layer: {layer_id-1} Finished! \n" + "----" * 20)     

def seg_with_unembedding_tokens(
    device, 
    model_name, 
    image_path,
    original_size=False,
    unembed_llm_layer=-1,
    delete_pos_embed=False,
):  
    # path
    save_dir = root_dir / "figures/seg_with_unembedding_tokens"
    if delete_pos_embed:
        save_dir += "_delete_pos_embed"
    new_save_dir = os.path.join(save_dir, model_name)
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}")
    os.makedirs(image_save_dir, exist_ok=True)
    shutil.copy(image_path, image_save_dir)
    
    text_token_path = os.path.join(image_save_dir, f"text_tokens.jsonl")
    with jsonlines.open(text_token_path, "w") as f:
        pass
    
    # get image info
    obj_bboxes = get_scene_info(image_path)
    if not obj_bboxes:  # invalid image
        return
    draw_bbox(image_path, obj_bboxes, image_save_dir)
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        model_config = model.config
        vit, llm = None, None
        if "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None
        
        # get ViT outputs
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
                ],
            }
        ]
        
        ## forward
        vit_layer_num = 0
        discard_vit_layers_func = None
        if any(name in model_name for name in ["llava"]):
            discard_vit_layers_func = replace_llava1_5_test_directions_vit_discard_layers
            replace_llava1_5_return_image_mask()
            replace_llava1_5_processor_forward_return_image_size()
            vit_layer_num = model.config.vision_config.num_hidden_layers
        elif any(name in model_name for name in ["intern"]):
            vit_layer_num = model.config.vision_config.num_hidden_layers
        
    ## Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path).convert('RGB')
    inputs, image_size = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs.to(model.device)
        
    # intervene
    if delete_pos_embed:
        replace_llava1_5_vl_delete_vit_pos_embed()
    
    for layer_id in range(vit_layer_num):
        # if layer_id != vit_layer_num - 1:
        #     continue
        
        # get text tokens
        text_tokens = []
        with torch.no_grad():
            layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
            if "intern" in model_name:
                discard_vit_layers_func(model, layer_ids_to_delete)
            else:
                discard_vit_layers_func(layer_ids_to_delete)
            outputs, image_mask = model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # activations = outputs.logits
            activations = llm.lm_head(outputs.hidden_states[unembed_llm_layer])  # maybe the last LLM layer performs the best, maybe not
        
            text_ids = torch.argmax(activations, dim=-1)  # (seq_len)
            # import pdb; pdb.set_trace()
            text_ids = text_ids[0]
            all_tokens = [processor.tokenizer.decode(text_id) for text_id in text_ids]
            # print(all_tokens)
            
            image_mask = image_mask[0].squeeze(-1)
            # print(image_mask)
            # print(image_mask.shape)
            # import pdb; pdb.set_trace()
            assert len(all_tokens) == len(image_mask)
            
            text_tokens = [all_tokens[i] for i in range(len(all_tokens)) if image_mask[i]]
        
        # ==============================================================================
        # draw original tokens map
        if True:
            # create a blank image
            height, width = image_size  # after patching
            # print(f"image_size: {image_size}")
            # import pdb; pdb.set_trace()
            cell_width = 50
            table_width, table_height = width * cell_width, height * cell_width
            cell_width, cell_height = cell_width, cell_width

            seg_map = Image.new("RGB", (table_width, table_height), "white")
            draw = ImageDraw.Draw(seg_map)

            font_path = "/raid_sdd/lyy/font/SimHei.ttf"
            try:
                font = ImageFont.truetype(font_path, size=15)
            except IOError:
                font = None

            # draw lines
            for x in range(0, table_width, cell_width):
                draw.line((x, 0, x, table_height), fill="black")
            for y in range(0, table_height, cell_height):
                draw.line((0, y, table_width, y), fill="black")

            # draw text
            for j in range(len(text_tokens)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                text = text_tokens[j]
                left, top, right, bottom = draw.textbbox((0, 0), text)
                text_width, text_height = right - left, top - bottom
                
                text_x = x * cell_width + (cell_width - text_width) / 2
                text_y = y * cell_height + (cell_height - text_height) / 2
                
                draw.text((text_x, text_y), text, fill="black", font=font)
            
            text_token_save_dir = os.path.join(image_save_dir, "text_tokens")
            os.makedirs(text_token_save_dir, exist_ok=True)
            save_path = os.path.join(text_token_save_dir, f"token-vit_{layer_id}-unembed_llm_{unembed_llm_layer}.pdf")
            seg_map.save(save_path)
            
            # save text tokens
            with jsonlines.open(text_token_path, "a") as f:
                data = {"image_id": image_name, "processed": False, "vit_layer": layer_id, "unembed_llm_layer": unembed_llm_layer, "text_tokens": text_tokens}
                f.write(data)
        
        # ==============================================================================
        # clean the text tokens
        if True:
            # 1. get image info
            image_info = None
            with jsonlines.open(root_dir / "test_figs/info_gqa.jsonl", "r") as f:
                for line in f:
                    if line["image_id"] in image_path:
                        image_info = line
                        break
            
            # 2. merge text tokens according to the keywords of the objects
            if image_info:
                objects_info = image_info["objects"]
                class_names = image_info["objects"].keys()
                for i in range(len(text_tokens)):
                    possible_classes = {class_name: 0 for class_name in class_names}
                    for obj_name, obj_info in objects_info.items():
                    #     if any(f"_{text_tokens[i]}" in f"_{val}" for val in obj_info["names"]):
                    #         text_tokens[i] = obj_name
                    #         break
                        for val in obj_info["names"]:
                            if f" {text_tokens[i]}" in f" {val}":
                                possible_classes[obj_name] = max(len(text_tokens[i]) / len(val), possible_classes[obj_name])
                    if not all([val == 0 for val in possible_classes.values()]):
                        max_class = max(possible_classes, key=possible_classes.get)
                        text_tokens[i] = max_class
                        
                # 3. change the name of the tokens the number of which is less than 3
                # get the unique labels and their counts
                # label_counts = Counter(text_tokens)  # {label: count, ...}
                
                # if "2354453" in image_path and not delete_pos_embed:
                #     text_tokens_copy = text_tokens
                # else:
                #     text_tokens_copy = copy.deepcopy(text_tokens)
                text_tokens_copy = copy.deepcopy(text_tokens)
                for i in range(len(text_tokens)):
                    if text_tokens[i] in class_names:
                        continue
                    # if label_counts[text_tokens[i]] < 5 or text_tokens[i] in []:
                    else:
                        height, width = image_size
                        x = i % width
                        y = i // width
                        x, y = int(x), int(y)
                        around_tokens = []
                        for j in range(-1, 2):
                            for k in range(-1, 2):
                                if j == 0 and k == 0:
                                    continue
                                x_ = x + j
                                y_ = y + k
                                if 0 <= x_ < width and 0 <= y_ < height:
                                    around_token = text_tokens_copy[y_ * width + x_]
                                    around_token_dist = abs(j) + abs(k)
                                    punctuations = string.punctuation + string.whitespace
                                    invalid_names = list(punctuations) + ["\n", "in", "on", "and"]
                                    if around_token not in invalid_names:
                                        around_tokens.append((around_token, around_token_dist))
                                else:
                                    continue                        
                        token_names = [token[0] for token in around_tokens]
                        # if not token_names:
                        #         text_tokens[i] = "BG"
                        #         continue
                        
                        # case 1: around tokens are all nonsense
                        if not token_names:
                            # text_tokens[i] = "BG"
                            # continue
                            around_tokens = []
                            for j in range(-2, 3):
                                for k in range(-2, 3):
                                    if j == 0 and k == 0:
                                        continue
                                    x_ = x + j
                                    y_ = y + k
                                    if 0 <= x_ < width and 0 <= y_ < height:
                                        around_token = text_tokens_copy[y_ * width + x_]
                                        around_token_dist = abs(j) + abs(k)
                                        punctuations = string.punctuation + string.whitespace
                                        invalid_names = list(punctuations) + ["\n", "in", "on", "and"]
                                        if around_token not in invalid_names:
                                            around_tokens.append((around_token, around_token_dist))
                                    else:
                                        continue
                            token_names = [token[0] for token in around_tokens]
                            if not token_names:
                                text_tokens[i] = "BG"
                                continue
                            
                        # sort the around tokens by appearance times
                        token_name_counter = Counter(token_names)
                        # if the count of each tokens is 1, then check if one of the token is the class name
                        if all([val == 1 for val in token_name_counter.values()]):
                            for val in token_name_counter:
                                for cls_name in class_names:
                                    if f" {val}" in f" {cls_name}":
                                        text_tokens[i] = cls_name
                                        break
                        possible_choices = token_name_counter.most_common(2)  # [("name", count), ...]
                        # case 2: there exists only 1 class with most counts
                        if len(possible_choices) == 1 or (len(possible_choices) > 1 and possible_choices[0][1] > possible_choices[1][1]):
                            for cls_name in class_names:
                                if f" {possible_choices[0][0]}" in f" {cls_name}":
                                    text_tokens[i] = cls_name
                                    break
                            else:
                                text_tokens[i] = "others"
                        else:
                            # case 3: choose the name with the shortest distance
                            dist_0 = sum([token[1] for token in around_tokens if token[0] == possible_choices[0][0]])
                            dist_1 = sum([token[1] for token in around_tokens if token[0] == possible_choices[1][0]])
                            selected_token = possible_choices[0][0] if dist_0 < dist_1 else possible_choices[1][0]
                            for cls_name in class_names:
                                if f" {selected_token}" in f" {cls_name}":
                                    text_tokens[i] = cls_name
                                    break
                            else:
                                text_tokens[i] = "others"
                print(Counter(text_tokens))
            
                # save new text tokens
                text_token_path = os.path.join(image_save_dir, f"text_tokens.jsonl")
                with jsonlines.open(text_token_path, "a") as f:
                    data = {"image_id": image_name, "processed": True, "vit_layer": layer_id, "unembed_llm_layer": unembed_llm_layer, "text_tokens": text_tokens}
                    f.write(data)
                   
        # =============================================================
        # draw new tokens map
        if True:
            # create a blank image
            height, width = image_size
            # print(f"image_size: {image_size}")
            # import pdb; pdb.set_trace()
            cell_width = 50
            table_width, table_height = width * cell_width, height * cell_width
            cell_width, cell_height = cell_width, cell_width

            seg_map = Image.new("RGB", (table_width, table_height), "white")
            draw = ImageDraw.Draw(seg_map)

            font_path = "/raid_sdd/lyy/font/SimHei.ttf"
            try:
                font = ImageFont.truetype(font_path, size=15)
            except IOError:
                font = None

            # draw lines
            for x in range(0, table_width, cell_width):
                draw.line((x, 0, x, table_height), fill="black")
            draw.line((table_width, 0, table_width, table_height), fill="black")
            for y in range(0, table_height, cell_height):
                draw.line((0, y, table_width, y), fill="black")
            draw.line((0, table_height, table_width, table_height), fill="black")

            # draw text
            for j in range(len(text_tokens)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                text = text_tokens[j]
                left, top, right, bottom = draw.textbbox((0, 0), text)
                text_width, text_height = right - left, top - bottom
                
                text_x = x * cell_width + (cell_width - text_width) / 2
                text_y = y * cell_height + (cell_height - text_height) / 2
                
                draw.text((text_x, text_y), text, fill="black", font=font)
            
            new_text_token_save_dir = os.path.join(image_save_dir, "new_text_tokens")
            os.makedirs(new_text_token_save_dir, exist_ok=True)
            save_path = os.path.join(new_text_token_save_dir, f"new_token-vit_{layer_id}-unembed_{unembed_llm_layer}.pdf")
            seg_map.save(save_path)
            
        # =============================================================
        # draw the semantic segmentation map, using the clustering results
        if True:
            # 1. get the unique labels
            unique_labels = np.unique(text_tokens)
            print(f"unique_labels: {unique_labels}")
            
            # 2. create a color map
            color_map = {}
            if image_info:
                for i, label in enumerate(unique_labels):
                    if label in image_info["objects"] and image_info["objects"][label]["color"] is not None:
                        color_map[label] = image_info["objects"][label]["color"]
                    else:
                        colors_exists = [obj_info["color"] for obj_info in image_info["objects"].values() if obj_info["color"] is not None]
                        # choose a random color other than existing colors
                        color = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
            else:
                for i, label in enumerate(unique_labels):
                    color_map[label] = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
                
            # 3. create a blank image
            height, width = image_size
            print(f"image_size: {image_size}")
            # import pdb; pdb.set_trace()
            
            if original_size:
                scale = model.config.vision_config.patch_size
                original_width = width * scale
                original_height = height * scale
                new_text_tokens = np.array([[None for _ in range(original_width)] for _ in range(original_height)])
                for j in range(len(text_tokens)):
                    x = j % width
                    y = j // width
                    x, y = int(x), int(y)
                    x_range = slice(x * scale, (x + 1) * scale)
                    y_range = slice(y * scale, (y + 1) * scale)
                    # print(f"j: {j}, width: {width}, x: {x}, y: {y},  x_range: {x_range}, y_range: {y_range}")
                    for row in range(y_range.start, y_range.stop):
                        for col in range(x_range.start, x_range.stop):
                            new_text_tokens[row][col] = text_tokens[j]
                text_tokens = new_text_tokens.flatten()
                # print(text_tokens)
                seg_map = Image.new("RGB", (original_width, original_height), (0, 0, 0))
            else:
                seg_map = Image.new("RGB", (width, height), (0, 0, 0))
            print(text_tokens)
                
            # 4. draw the segmentation map
            for j in range(len(text_tokens)):
                x = j % original_width if original_size else j % width
                y = j // original_width if original_size else j // width
                x, y = int(x), int(y)
                color = color_map[text_tokens[j]]
                rgb_color = mcolors.to_rgb(color)
                int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
                seg_map.putpixel((x, y), int_color)
                
            # 5. save the segmentation map
            seg_image_save_dir = os.path.join(image_save_dir, "seg_image")
            os.makedirs(seg_image_save_dir, exist_ok=True)
            if original_size:
                save_path = os.path.join(seg_image_save_dir, f"img-vit_{layer_id}-unembed_{unembed_llm_layer}.pdf")
            else:
                save_path = os.path.join(seg_image_save_dir, f"img-vit_{layer_id}-unembed_{unembed_llm_layer}.pdf")
            seg_map.save(save_path)
            
            print(f"image: {image_path}, layer: {layer_id-1} Finished! \n" + "----" * 20)
    # analyze_logit_lens
    analyze_logit_lens(model_name, image_path, model_config, inputs, delete_pos_embed=delete_pos_embed) 

def seg_with_unembedding_tokens_svg(
    device, 
    model_name, 
    image_path,
    original_size=False,
    unembed_llm_layer=-1,
    delete_pos_embed=False,
):  
    # path
    save_dir = str(root_dir / "figures/seg_with_unembedding_tokens")
    if delete_pos_embed:
        save_dir += "_delete_pos_embed"
    new_save_dir = os.path.join(save_dir, model_name)
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}")
    os.makedirs(image_save_dir, exist_ok=True)
    shutil.copy(image_path, image_save_dir)
    
    text_token_path = os.path.join(image_save_dir, f"text_tokens.jsonl")
    with jsonlines.open(text_token_path, "w") as f:
        pass
    
    # get image info
    if False:
        obj_bboxes = get_scene_info(image_path)
        if not obj_bboxes:  # invalid image
            return
        draw_bbox(image_path, obj_bboxes, image_save_dir)
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        model_config = model.config
        vit, llm = None, None
        if "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None
        
        # get ViT outputs
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
                ],
            }
        ]
        
        ## forward
        vit_layer_num = 0
        discard_vit_layers_func = None
        if any(name in model_name for name in ["llava"]):
            discard_vit_layers_func = replace_llava1_5_test_directions_vit_discard_layers
            replace_llava1_5_return_image_mask()
            replace_llava1_5_processor_forward_return_image_size()
            vit_layer_num = model.config.vision_config.num_hidden_layers
        elif any(name in model_name for name in ["intern"]):
            vit_layer_num = model.config.vision_config.num_hidden_layers
        
    ## Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path).convert('RGB')
    inputs, image_size = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs.to(model.device)
        
    # intervene
    if delete_pos_embed:
        replace_llava1_5_vl_delete_vit_pos_embed()
    
    for layer_id in range(vit_layer_num):
        # if layer_id != vit_layer_num - 1:
        #     continue
        
        # get text tokens
        text_tokens = []
        with torch.no_grad():
            layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
            if "intern" in model_name:
                discard_vit_layers_func(model, layer_ids_to_delete)
            else:
                discard_vit_layers_func(layer_ids_to_delete)
            outputs, image_mask = model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            
            # activations = outputs.logits
            activations = llm.lm_head(outputs.hidden_states[unembed_llm_layer])  # maybe the last LLM layer performs the best, maybe not
        
            text_ids = torch.argmax(activations, dim=-1)  # (seq_len)
            # import pdb; pdb.set_trace()
            text_ids = text_ids[0]
            all_tokens = [processor.tokenizer.decode(text_id) for text_id in text_ids]
            # print(all_tokens)
            
            image_mask = image_mask[0].squeeze(-1)
            # print(image_mask)
            # print(image_mask.shape)
            # import pdb; pdb.set_trace()
            assert len(all_tokens) == len(image_mask)
            
            text_tokens = [all_tokens[i] for i in range(len(all_tokens)) if image_mask[i]]
        
        # ==============================================================================
        # draw original tokens map
        if True:
            # create a blank image
            height, width = image_size  # after patching
            # print(f"image_size: {image_size}")
            # import pdb; pdb.set_trace()
            cell_width = 50
            table_width, table_height = width * cell_width, height * cell_width
            cell_width, cell_height = cell_width, cell_width

            # Create SVG canvas
            cell_size = cell_width 
            svg_width = table_width
            svg_height = table_height
            dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"))
            
            # Add white background
            dwg.add(dwg.rect(
                insert=(0, 0),
                size=(f"{svg_width}px", f"{svg_height}px"),
                fill="white"
            ))
            
            # Draw grid
            for x in range(0, svg_width + 1, cell_size):
                dwg.add(dwg.line(start=(x, 0), end=(x, svg_height), stroke="black", stroke_width=1))
            for y in range(0, svg_height + 1, cell_size):
                dwg.add(dwg.line(start=(0, y), end=(svg_width, y), stroke="black", stroke_width=1))
            
            font_path = "/raid_sdd/lyy/font/SimHei.ttf"
            try:
                font = ImageFont.truetype(font_path, size=15)
            except IOError:
                font = None
                
            # Add text
            for j, text in enumerate(text_tokens):
                x = j % width
                y = j // width
                x_pos = x * cell_size + cell_size / 2
                y_pos = y * cell_size + cell_size / 2
                
                dwg.add(dwg.text(
                    text,
                    insert=(x_pos, y_pos),
                    fill="black",
                    font_family=font,  # Use your font
                    font_size="12px",
                    text_anchor="middle",
                    dominant_baseline="middle"
                ))
            
            # Save SVG
            text_token_save_dir = os.path.join(image_save_dir, "text_tokens")
            os.makedirs(text_token_save_dir, exist_ok=True)            
            save_path = os.path.join(text_token_save_dir, f"token-vit_{layer_id}-unembed_llm_{unembed_llm_layer}.svg")
            dwg.saveas(save_path)
            
            # save text tokens
            with jsonlines.open(text_token_path, "a") as f:
                data = {"image_id": image_name, "processed": False, "vit_layer": layer_id, "unembed_llm_layer": unembed_llm_layer, "text_tokens": text_tokens}
                f.write(data)
        
        # ==============================================================================
        # clean the text tokens
        if True:
            # 1. get image info
            image_info = None
            with jsonlines.open(root_dir / "test_figs/info_gqa.jsonl", "r") as f:
                for line in f:
                    if line["image_id"] in image_path:
                        image_info = line
                        break
            
            # 2. merge text tokens according to the keywords of the objects
            if image_info:
                objects_info = image_info["objects"]
                class_names = image_info["objects"].keys()
                for i in range(len(text_tokens)):
                    possible_classes = {class_name: 0 for class_name in class_names}
                    for obj_name, obj_info in objects_info.items():
                    #     if any(f"_{text_tokens[i]}" in f"_{val}" for val in obj_info["names"]):
                    #         text_tokens[i] = obj_name
                    #         break
                        for val in obj_info["names"]:
                            if f" {text_tokens[i]}" in f" {val}":
                                possible_classes[obj_name] = max(len(text_tokens[i]) / len(val), possible_classes[obj_name])
                    if not all([val == 0 for val in possible_classes.values()]):
                        max_class = max(possible_classes, key=possible_classes.get)
                        text_tokens[i] = max_class
                        
                # 3. change the name of the tokens the number of which is less than 3
                # get the unique labels and their counts
                # label_counts = Counter(text_tokens)  # {label: count, ...}
                
                # if "2354453" in image_path and not delete_pos_embed:
                #     text_tokens_copy = text_tokens
                # else:
                #     text_tokens_copy = copy.deepcopy(text_tokens)

                text_tokens_copy = copy.deepcopy(text_tokens)                

                for i in range(len(text_tokens)):
                    if text_tokens[i] in class_names:
                        continue
                    # if label_counts[text_tokens[i]] < 5 or text_tokens[i] in []:
                    else:
                        height, width = image_size
                        x = i % width
                        y = i // width
                        x, y = int(x), int(y)
                        around_tokens = []
                        for j in range(-1, 2):
                            for k in range(-1, 2):
                                if j == 0 and k == 0:
                                    continue
                                x_ = x + j
                                y_ = y + k
                                if 0 <= x_ < width and 0 <= y_ < height:
                                    around_token = text_tokens_copy[y_ * width + x_]
                                    around_token_dist = abs(j) + abs(k)
                                    punctuations = string.punctuation + string.whitespace
                                    invalid_names = list(punctuations) + ["\n", "in", "on", "and"]
                                    if around_token not in invalid_names:
                                        around_tokens.append((around_token, around_token_dist))
                                else:
                                    continue                        
                        token_names = [token[0] for token in around_tokens]
                        # if not token_names:
                        #         text_tokens[i] = "BG"
                        #         continue
                        
                        # case 1: around tokens are all nonsense
                        if not token_names:
                            # text_tokens[i] = "BG"
                            # continue
                            around_tokens = []
                            for j in range(-2, 3):
                                for k in range(-2, 3):
                                    if j == 0 and k == 0:
                                        continue
                                    x_ = x + j
                                    y_ = y + k
                                    if 0 <= x_ < width and 0 <= y_ < height:
                                        around_token = text_tokens_copy[y_ * width + x_]
                                        around_token_dist = abs(j) + abs(k)
                                        punctuations = string.punctuation + string.whitespace
                                        invalid_names = list(punctuations) + ["\n", "in", "on", "and"]
                                        if around_token not in invalid_names:
                                            around_tokens.append((around_token, around_token_dist))
                                    else:
                                        continue
                            token_names = [token[0] for token in around_tokens]
                            if not token_names:
                                text_tokens[i] = "BG"
                                continue
                            
                        # sort the around tokens by appearance times
                        token_name_counter = Counter(token_names)
                        # if the count of each tokens is 1, then check if one of the token is the class name
                        if all([val == 1 for val in token_name_counter.values()]):
                            for val in token_name_counter:
                                for cls_name in class_names:
                                    if f" {val}" in f" {cls_name}":
                                        text_tokens[i] = cls_name
                                        break
                        possible_choices = token_name_counter.most_common(2)  # [("name", count), ...]
                        # case 2: there exists only 1 class with most counts
                        if len(possible_choices) == 1 or (len(possible_choices) > 1 and possible_choices[0][1] > possible_choices[1][1]):
                            for cls_name in class_names:
                                if f" {possible_choices[0][0]}" in f" {cls_name}":
                                    text_tokens[i] = cls_name
                                    break
                            else:
                                text_tokens[i] = "others"
                        else:
                            # case 3: choose the name with the shortest distance
                            dist_0 = sum([token[1] for token in around_tokens if token[0] == possible_choices[0][0]])
                            dist_1 = sum([token[1] for token in around_tokens if token[0] == possible_choices[1][0]])
                            selected_token = possible_choices[0][0] if dist_0 < dist_1 else possible_choices[1][0]
                            for cls_name in class_names:
                                if f" {selected_token}" in f" {cls_name}":
                                    text_tokens[i] = cls_name
                                    break
                            else:
                                text_tokens[i] = "others"
                print(Counter(text_tokens))
            
                # save new text tokens
                text_token_path = os.path.join(image_save_dir, f"text_tokens.jsonl")
                with jsonlines.open(text_token_path, "a") as f:
                    data = {"image_id": image_name, "processed": True, "vit_layer": layer_id, "unembed_llm_layer": unembed_llm_layer, "text_tokens": text_tokens}
                    f.write(data)
                   
        # =============================================================
        # draw new tokens map
        if True:
            # create a blank image
            height, width = image_size
            # print(f"image_size: {image_size}")
            # import pdb; pdb.set_trace()
            cell_width = 50
            table_width, table_height = width * cell_width, height * cell_width
            cell_width, cell_height = cell_width, cell_width

            # Create SVG canvas
            cell_size = cell_width 
            svg_width = table_width
            svg_height = table_height
            dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"))
            
            # Add white background
            dwg.add(dwg.rect(
                insert=(0, 0),
                size=(f"{svg_width}px", f"{svg_height}px"),
                fill="white"
            ))
            
            # Draw grid
            for x in range(0, svg_width + 1, cell_size):
                dwg.add(dwg.line(start=(x, 0), end=(x, svg_height), stroke="black", stroke_width=1))
            for y in range(0, svg_height + 1, cell_size):
                dwg.add(dwg.line(start=(0, y), end=(svg_width, y), stroke="black", stroke_width=1))

            font_path = "/raid_sdd/lyy/font/SimHei.ttf"
            try:
                font = ImageFont.truetype(font_path, size=15)
            except IOError:
                font = None

            # Add text
            for j, text in enumerate(text_tokens):
                x = j % width
                y = j // width
                x_pos = x * cell_size + cell_size / 2
                y_pos = y * cell_size + cell_size / 2
                
                dwg.add(dwg.text(
                    text,
                    insert=(x_pos, y_pos),
                    fill="black",
                    font_family=font,  # Use your font
                    font_size="12px",
                    text_anchor="middle",
                    dominant_baseline="middle"
                ))
                
            
            new_text_token_save_dir = os.path.join(image_save_dir, "new_text_tokens")
            os.makedirs(new_text_token_save_dir, exist_ok=True)
            save_path = os.path.join(new_text_token_save_dir, f"new_token-vit_{layer_id}-unembed_{unembed_llm_layer}.svg")
            dwg.saveas(save_path)
            
        # =============================================================
        # draw the semantic segmentation map, using the clustering results
        if True:
            # 1. get the unique labels
            unique_labels = np.unique(text_tokens)
            print(f"unique_labels: {unique_labels}")
            
            # 2. create a color map
            color_map = {}
            if image_info:
                for i, label in enumerate(unique_labels):
                    if label in image_info["objects"] and image_info["objects"][label]["color"] is not None:
                        color = image_info["objects"][label]["color"]
                        try:
                            color_map[label] = mcolors.to_hex(color)
                        except ValueError:
                            color_map[label] = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
            else:
                for i, label in enumerate(unique_labels):
                    color_map[label] = mcolors.to_hex(plt.cm.viridis(i / len(unique_labels)))
                
            # 3. create a blank image
            height, width = image_size
            print(f"image_size: {image_size}")
            # import pdb; pdb.set_trace()
            
            if original_size:
                scale = model.config.vision_config.patch_size
                original_width = width * scale
                original_height = height * scale
                new_text_tokens = np.array([[None for _ in range(original_width)] for _ in range(original_height)])
                for j in range(len(text_tokens)):
                    x = j % width
                    y = j // width
                    x, y = int(x), int(y)
                    x_range = slice(x * scale, (x + 1) * scale)
                    y_range = slice(y * scale, (y + 1) * scale)
                    # print(f"j: {j}, width: {width}, x: {x}, y: {y},  x_range: {x_range}, y_range: {y_range}")
                    for row in range(y_range.start, y_range.stop):
                        for col in range(x_range.start, x_range.stop):
                            new_text_tokens[row][col] = text_tokens[j]
                text_tokens = new_text_tokens.flatten()
                output_width, output_height = original_width, original_height
            else:
                output_width, output_height = width, height
            print(text_tokens)
                
            # 4. draw the segmentation map
            cell_size = 10  # Size of each cell in pixels
            svg_width = output_width * cell_size
            svg_height = output_height * cell_size
            
            dwg = svgwrite.Drawing(
                size=(f"{svg_width}px", f"{svg_height}px"),
                viewBox=(f"0 0 {svg_width} {svg_height}")
            )
            
            # Add white background
            dwg.add(dwg.rect(
                insert=(0, 0),
                size=("100%", "100%"),
                fill="white"
            ))

            # 4. Draw colored cells
            for j in range(len(text_tokens)):
                if text_tokens[j] is None:
                    continue
                    
                x = j % output_width
                y = j // output_width
                x, y = int(x), int(y)
                
                # # Flip y-coordinate to match image coordinates
                # y_svg = output_height - 1 - y
                
                color = color_map[text_tokens[j]]
                
                dwg.add(dwg.rect(
                    insert=(x * cell_size, y * cell_size),
                    size=(cell_size, cell_size),
                    fill=color,
                    stroke="none"
                ))
                
            # 5. save the segmentation map
            seg_image_save_dir = os.path.join(image_save_dir, "seg_image")
            os.makedirs(seg_image_save_dir, exist_ok=True)
            save_path = os.path.join(
                seg_image_save_dir, 
                f"img-vit_{layer_id}-unembed_{unembed_llm_layer}.svg"
            )
            dwg.saveas(save_path)            
            print(f"image: {image_path}, layer: {layer_id-1} Finished! \n" + "----" * 20)
            
    # analyze_logit_lens
    # analyze_logit_lens(model_name, image_path, model_config, inputs, delete_pos_embed=delete_pos_embed) 
    
def analyze_logit_lens(model_name, image_path, model_config=None, inputs=None, delete_pos_embed=False):
    # load results
    exp_dir = root_dir / "figures/seg_with_unembedding_tokens"
    if delete_pos_embed:
        exp_dir += "_delete_pos_embed"
    image_name = os.path.basename(image_path).split(".")[0]
    image_save_dir = os.path.join(exp_dir, model_name, f"seg_img_{image_name}")
    text_tokens_path = os.path.join(image_save_dir, f"text_tokens.jsonl")
    all_text_tokens = []
    with jsonlines.open(text_tokens_path, "r") as f:
        for line in f:
            if not line["processed"]:
                all_text_tokens.append(line["text_tokens"])
    
    # load model
    if model_config is None or inputs is None:
        model_dir = MODEL_NAME_TO_PATH[model_name]
        processor = AutoProcessor.from_pretrained(model_dir, padding_side='left', use_fast=True)
        model_config = AutoConfig.from_pretrained(model_dir)
            
        ## Preparation for inference
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert('RGB')
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
    
    # get objects info and filter the objects
    if True:
        obj_bboxes = get_scene_info(image_path)
        # 1. get resized bbox
        # height, width = image_size  # after patching
        height, width = inputs["pixel_values"][0].shape[1:]  # after resize, before patching
        original_image = Image.open(image_path).convert('RGB')
        original_width, original_height = original_image.size  # square image
        scale = width / original_width
        print(f"original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
        # import pdb; pdb.set_trace()
        resized_bboxes = {}
        for obj_name, obj_bbox in obj_bboxes.items():
            x1, y1, x2, y2 = obj_bbox  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)  # 
            resized_bboxes[obj_name] = (x1, y1, x2, y2)
        print(f"image_size: {width}, {height}")
        
        # 2. the resized image is split into patches with self.patch_size as size. Now let's get the obj patches
        # get the coordinates of the patches in the whole image
        object_patch_ids = {}
        patch_size = model_config.vision_config.patch_size
        for obj_name, obj_bbox in resized_bboxes.items():
            obj_patch_ids = []
            x1, y1, x2, y2 = obj_bbox
            for i in range(int(width // patch_size)):
                for j in range(int(height // patch_size)):
                    x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                    if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                    # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                        obj_patch_ids.append(j * int(width // patch_size) + i)
            object_patch_ids[obj_name] = obj_patch_ids
    
    # For each obj, get text tokens for each layer
    object_text_tokens = {obj_name: [] for obj_name in object_patch_ids.keys()}
    for obj_name, obj_patch_ids in object_patch_ids.items():
        obj_text_tokens = []
        for layer_id in range(len(all_text_tokens)):
            layer_text_tokens = all_text_tokens[layer_id]
            obj_text_tokens = [layer_text_tokens[patch_id] for patch_id in obj_patch_ids]
            obj_text_tokens = Counter(obj_text_tokens)
            # items = {token: count for token, count in obj_text_tokens.items()}
            items = obj_text_tokens.most_common(10)
            object_text_tokens[obj_name].append(items)
    
    # save
    save_path = os.path.join(image_save_dir, f"text_tokens_analysis.jsonl")
    with jsonlines.open(save_path, "w") as f:
        for obj_name, obj_text_tokens in object_text_tokens.items():
            f.write({"object": obj_name, "text_tokens": obj_text_tokens})
    
    # draw

def analyze_semantic_segmentation(image_ids=[], model_name="", delete_pos_embed=False):
    exp_dir = root_dir / "figures/seg_with_unembedding_tokens"
    if delete_pos_embed:
        exp_dir += "_delete_pos_embed"
    image_info_path = root_dir / "test_figs/info_gqa_0.jsonl"
    attribution_words_ratio = [[] for _ in range(24)]
    class_name_words_ratio = [[] for _ in range(24)]
    for image_id in image_ids:
        text_tokens_analysis_path = os.path.join(exp_dir, model_name, f"seg_img_{image_id}", "text_tokens_analysis.jsonl")
        # get the attribution words and class words for all objects
        objects_words = {}
        with jsonlines.open(image_info_path, "r") as f:
            for line in f:
                if line["image_id"] == image_id:
                    objects_words = line["objects"]
                    break
        # get the number of all tokens for the image and for each object
        all_tokens_num = 0
        objects_num = {}
        with jsonlines.open(text_tokens_analysis_path, "r") as f:
            for line in f:
                objects_num[line["object"]] = np.sum([pair[1] for pair in line["text_tokens"][0]])
        all_tokens_num = np.sum(cnt for name, cnt in objects_num.items() if name in objects_words)
        object_ratio = {obj_name: cnt / all_tokens_num for obj_name, cnt in objects_num.items() if obj_name in objects_words}
        
        # get the attribution ratio and class name ratio
        with jsonlines.open(text_tokens_analysis_path, "r") as f:
            image_attribution_words_ratio = [[] for _ in range(24)]
            image_class_name_words_ratio = [[] for _ in range(24)]
            for line in f:  # for each object
                obj_name = line["object"]
                if obj_name not in objects_words:
                    continue
                all_text_tokens = line["text_tokens"]
                for layer_id, layer_text_tokens in enumerate(all_text_tokens):
                    total_words_num = np.sum([pair[1] for pair in layer_text_tokens])
                    attr_words_num = 0
                    cls_words_num = 0
                    for token, count in layer_text_tokens:
                        if any([f" {token}" in f" {word}" for word in objects_words[obj_name]["attr"]]):
                            attr_words_num += count
                        elif any([f" {token}" in f" {word}" for word in objects_words[obj_name]["cls"]]):
                            cls_words_num += count
                        else:
                            continue
                    image_attribution_words_ratio[layer_id].append((attr_words_num / total_words_num) * object_ratio[line["object"]])
                    image_class_name_words_ratio[layer_id].append((cls_words_num / total_words_num) * object_ratio[line["object"]])
        for layer_id in range(24):
            attribution_words_ratio[layer_id].append(np.sum(image_attribution_words_ratio[layer_id]))
            class_name_words_ratio[layer_id].append(np.sum(image_class_name_words_ratio[layer_id]))
        
    # plot
    if delete_pos_embed:
        save_path = os.path.join(exp_dir, model_name, "semantic_segmentation_analysis_delete_pos_embed.pdf")
    else:
        save_path = os.path.join(exp_dir, model_name, "semantic_segmentation_analysis.pdf")
    layer_ids = np.arange(1, 25)
    attr_mean_acc = np.mean(attribution_words_ratio, axis=1)
    attr_std_err = np.std(attribution_words_ratio, axis=1) / np.sqrt(len(attribution_words_ratio[0]))
    cls_mean_acc = np.mean(class_name_words_ratio, axis=1)
    cls_std_err = np.std(class_name_words_ratio, axis=1) / np.sqrt(len(class_name_words_ratio[0]))
    plt.plot(
        layer_ids,
        attr_mean_acc,
        color="blue",
        label='object attribution words',
        alpha=1, 
        marker='o',
        markersize=5, 
        #  markerfacecolor=(0, 0, 1, 0.7),
        markeredgecolor='blue',
        markeredgewidth=1.5
    )
    plt.fill_between(
        layer_ids, 
        attr_mean_acc - attr_std_err, 
        attr_mean_acc + attr_std_err, 
        color='blue', 
        alpha=0.1
    )
    plt.plot(
        layer_ids,
        cls_mean_acc,
        color="green",
        label='object representative words',
        alpha=1, 
        marker='o',
        markersize=5, 
        #  markerfacecolor=(0, 0, 1, 0.7),
        markeredgecolor='green',
        markeredgewidth=1.5
    )
    plt.fill_between(
        layer_ids, 
        cls_mean_acc - cls_std_err, 
        cls_mean_acc + cls_std_err, 
        color='green', 
        alpha=0.1
    )
    
    plt.xlim(0, 25)
    
    plt.legend(framealpha=0.6)  # loc="upper right", 
    
    plt.xlabel("Layer ID", loc="right")
    plt.ylabel("Emergence of object-related tokens")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    plt.savefig(save_path, bbox_inches='tight')

def check_attention(
    device, 
    model_name, 
    image_path,
    object_names=None,
    colors=None,
):    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device, use_flash_attention=False)
        vit, llm = None, None
        if "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None

        image_name = os.path.basename(image_path).split(".")[0]

        ## forward ViT
        vit_layer_num = 0
        if any(name in model_name for name in ["llava"]):
            replace_llava1_5_processor_forward_return_image_size()
            vit_layer_num = model.config.vision_config.num_hidden_layers
        elif any(name in model_name for name in ["intern"]):
            vit_layer_num = model.config.vision_config.num_hidden_layers
            
        ## Preparation for inference
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert('RGB')
        inputs, image_size = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs.to(model.device)
        
        # get attention scores
        all_attentions = vit(
            pixel_values=inputs["pixel_values"],
            output_attentions=True,
        ).attentions  # layer_num * (batch_size, num_heads, seq_len, seq_len)
    
    # get objects info and filter the objects
    if True:
        obj_bboxes = get_scene_info(image_path)
        if object_names:
            obj_bboxes = {name: bbox for name, bbox in obj_bboxes.items() if name in object_names}
        
        # 1. get resized bbox
        # height, width = image_size  # after patching
        height, width = inputs["pixel_values"][0].shape[1:]  # after resize, before patching
        original_image = Image.open(image_path).convert('RGB')
        original_width, original_height = original_image.size  # square image
        scale = width / original_width
        print(f"original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
        # import pdb; pdb.set_trace()
        resized_bboxes = {}
        for obj_name, obj_bbox in obj_bboxes.items():
            x1, y1, x2, y2 = obj_bbox  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)  # 
            resized_bboxes[obj_name] = (x1, y1, x2, y2)
        print(f"image_size: {width}, {height}")
        
        # 2. the resized image is split into patches with self.patch_size as size. Now let's get the obj patches
        # get the coordinates of the patches in the whole image
        object_patch_ids = {}
        patch_size = model.config.vision_config.patch_size
        for obj_name, obj_bbox in resized_bboxes.items():
            obj_patch_ids = []
            x1, y1, x2, y2 = obj_bbox
            for i in range(int(width // patch_size)):
                for j in range(int(height // patch_size)):
                    x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                    if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                    # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                        obj_patch_ids.append(j * int(width // patch_size) + i)
            object_patch_ids[obj_name] = obj_patch_ids
    
    
    # compute attention scores
    obj_attentions_cls = {name: [] for name in obj_bboxes.keys()}
    obj_atentions_inside = {name: [] for name in obj_bboxes.keys()}
    obj_attentions_outside = {name: [] for name in obj_bboxes.keys()}
    for layer_id in range(vit_layer_num):
        # if layer_id != vit_layer_num - 1:
        #     continue    
        
        # check the [cls] token
        attention_scores = all_attentions[layer_id][0]
        reduced_attention_scores = torch.mean(attention_scores, dim=0)  # (len, len) (reduced over attention heads)
        for obj_name in obj_bboxes.keys():
            obj_patch_ids = object_patch_ids[obj_name]
            obj_patch_ids = [patch_id + 1 for patch_id in obj_patch_ids]
            obj_cls_attention_scores = reduced_attention_scores[obj_patch_ids][0]  
            obj_cls_attention_score = torch.sum(obj_cls_attention_scores, dim=0).item()
            # obj_cls_attention_score = torch.sum(obj_cls_attention_scores.flatten()).item()
            obj_attentions_cls[obj_name].append(obj_cls_attention_score)
        
        # clustering using self attention
        # 1. get attention scores
        attention_scores = all_attentions[layer_id][0]  # the first sample: (num_heads, len, len), len = 1([cls]) + width * height
        attention_scores = attention_scores[:, 1:, 1:]  # remove cls token
        
        reduced_attention = torch.mean(attention_scores, dim=0)  # (len, len) (reduced over attention heads)
        # print("Image shape", image_size)
        # print("Attention shape", reduced_attention.shape)
        # print(reduced_attention[2], torch.sum(reduced_attention[2]))
        # import pdb; pdb.set_trace()
        
        # 2. For each object, compute the attention scores inside and outside it
        seq_len = reduced_attention.shape[0]
        for obj_name in obj_bboxes.keys():
            
            # get attention scores between the patches of the obj
            obj_patch_ids = object_patch_ids[obj_name]
            # obj_patch_ids = torch.tensor(obj_patch_ids)
            # obj_self_attention_scores = reduced_attention[torch.ix_(obj_patch_ids, obj_patch_ids)]
            obj_self_attention_scores = reduced_attention[obj_patch_ids][:, obj_patch_ids]
            obj_self_attention_score = torch.sum(obj_self_attention_scores, dim=0).mean(dim=-1).item()
            # obj_self_attention_score = torch.sum(obj_self_attention_scores.flatten()).item()
            # print(obj_self_attention_scores)
            # print(f"obj_name: {obj_name}, obj_patch_ids: {obj_patch_ids}, obj_self_attention_scores: {obj_self_attention_scores.shape}, obj_self_attention_score: {obj_self_attention_score}")
            # import pdb; pdb.set_trace()
            
            # get attention scores between the patches of the obj and the rest of the image
            other_patch_ids = [i for i in range(seq_len) if i not in obj_patch_ids]
            # other_patch_ids = torch.tensor(other_patch_ids)
            # obj_other_attention_scores = reduced_attention[torch.ix_(obj_patch_ids, other_patch_ids)]
            obj_other_attention_scores = reduced_attention[obj_patch_ids][:, other_patch_ids]
            obj_other_attention_score = torch.sum(obj_other_attention_scores, dim=0).mean(dim=-1).item()
            # obj_other_attention_score = torch.sum(obj_other_attention_scores.flatten()).item()
            
            # save the attention scores
            obj_atentions_inside[obj_name].append(obj_self_attention_score)
            obj_attentions_outside[obj_name].append(obj_other_attention_score)
    
        print(f"image: {image_path}, layer: {layer_id-1} Finished!")
    # print("obj_atentions_inside: ", obj_atentions_inside)
    # print("obj_attentions_outside: ", obj_attentions_outside)
    
    # draw the attention scores for each object
    plt.figure()
    layer_ids = [i for i in range(vit_layer_num)]
    if not colors:
        # randomly select a color for each obj
        colors = {name: mcolors.to_hex(plt.cm.viridis(i / len(obj_bboxes))) for i, name in enumerate(obj_bboxes.keys())}
    for key, val in obj_atentions_inside.items():
        plt.plot(layer_ids, val, color=colors[key], label=key)
    for key, val in obj_attentions_outside.items():
        plt.plot(layer_ids, val, color=colors[key], linestyle="--", label=f"{key} outside")
    # for key, val in obj_attentions_cls.items():
    #     plt.plot(layer_ids, val, color=colors[key], linestyle="-.", label=f"{key} [cls]")
    
    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel("Attention", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 1))
    
    fig_dir = root_dir / "figures"
    save_dir = os.path.join(fig_dir, f"seg_with_self_attention", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{image_name}-attentions.pdf")
    plt.savefig(save_path, bbox_inches='tight')

def check_attention_each_obj(
    device, 
    model_name, 
    image_path,
    objects_to_check=None,
    colors=None,
):    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device, use_flash_attention=False)
        vit, llm = None, None
        if "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None

        # path
        image_name = os.path.basename(image_path).split(".")[0]
        fig_dir = root_dir / "figures"
        image_save_dir = os.path.join(fig_dir, f"seg_with_self_attention", model_name, f"seg_img_{image_name}")
        os.makedirs(image_save_dir, exist_ok=True)
        
        ## forward ViT
        vit_layer_num = 0
        if any(name in model_name for name in ["llava"]):
            replace_llava1_5_processor_forward_return_image_size()
            vit_layer_num = model.config.vision_config.num_hidden_layers
        elif any(name in model_name for name in ["intern"]):
            vit_layer_num = model.config.vision_config.num_hidden_layers
            
        ## Preparation for inference
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert('RGB')
        inputs, image_size = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs.to(model.device)
        
        # get attention scores
        with torch.no_grad():
            all_attentions = vit(
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
            ).attentions  # layer_num * (batch_size, num_heads, seq_len, seq_len)
    
    # get objects info and filter the objects
    obj_bboxes = get_scene_info(image_path)
    draw_bbox(image_path, obj_bboxes, image_save_dir)
    if True:
        # 1. get resized bbox
        # height, width = image_size  # after patching
        height, width = inputs["pixel_values"][0].shape[1:]  # after resize, before patching
        original_image = Image.open(image_path).convert('RGB')
        original_width, original_height = original_image.size  # square image
        scale = width / original_width
        print(f"original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
        # import pdb; pdb.set_trace()
        resized_bboxes = {}
        for obj_name, obj_bbox in obj_bboxes.items():
            x1, y1, x2, y2 = obj_bbox  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)  # 
            resized_bboxes[obj_name] = (x1, y1, x2, y2)
        print(f"image_size: {width}, {height}")
        
        # 2. the resized image is split into patches with self.patch_size as size. Now let's get the obj patches
        # get the coordinates of the patches in the whole image
        object_patch_ids = {}
        patch_size = model.config.vision_config.patch_size
        for obj_name, obj_bbox in resized_bboxes.items():
            obj_patch_ids = []
            x1, y1, x2, y2 = obj_bbox
            for i in range(int(width // patch_size)):
                for j in range(int(height // patch_size)):
                    x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                    if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                    # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                        obj_patch_ids.append(j * int(width // patch_size) + i)
            object_patch_ids[obj_name] = obj_patch_ids


    # compute attention scores
    obj_attentions_cls = {name: [] for name in obj_bboxes.keys()}
    obj_atentions_inside = {name: [] for name in obj_bboxes.keys()}
    obj_attentions_outside = {}
    for key in objects_to_check:
        obj_attentions_outside[key] = {name: [] for name in obj_bboxes.keys() if name != key}
    
    for layer_id in range(vit_layer_num):
        # check the [cls] token
        attention_scores = all_attentions[layer_id][0]
        reduced_attention_scores = torch.mean(attention_scores, dim=0)  # (len, len) (reduced over attention heads)
        for obj_name in objects_to_check:
            obj_patch_ids = object_patch_ids[obj_name]
            obj_patch_ids = [patch_id + 1 for patch_id in obj_patch_ids]
            obj_cls_attention_scores = reduced_attention_scores[obj_patch_ids][:, 0]   # (len)
            obj_cls_attention_score = torch.mean(obj_cls_attention_scores, dim=0).item()
            # obj_cls_attention_score = torch.sum(obj_cls_attention_scores.flatten()).item()
            obj_attentions_cls[obj_name].append(obj_cls_attention_score)
        
        # clustering using self attention
        # 1. get attention scores
        attention_scores = all_attentions[layer_id][0]  # the first sample: (num_heads, len, len), len = 1([cls]) + width * height
        attention_scores = attention_scores[:, 1:, 1:]  # remove cls token
        reduced_attention = torch.mean(attention_scores, dim=0)  # (len, len) (reduced over attention heads)
        # print("Image shape", image_size)
        # print("Attention shape", reduced_attention.shape)
        # print(reduced_attention[2], torch.sum(reduced_attention[2]))
        # import pdb; pdb.set_trace()
        
        # 2. For each object, compute the attention scores inside and outside it
        seq_len = reduced_attention.shape[0]
        for obj_name, obj_patch_ids in object_patch_ids.items():
            if obj_name not in objects_to_check:
                continue
            
            # get attention scores between the patches of the obj
            obj_patch_ids = object_patch_ids[obj_name]
            obj_self_attention_scores = reduced_attention[obj_patch_ids][:, obj_patch_ids]
            obj_self_attention_score = torch.mean(obj_self_attention_scores).item()
            obj_atentions_inside[obj_name].append(obj_self_attention_score)
            
            # get attention scores between the patches of the obj and the other objects
            for other_obj_name, other_obj_patch_ids in object_patch_ids.items():
                if other_obj_name == obj_name:
                    continue
                obj_other_attention_scores = reduced_attention[obj_patch_ids][:, other_obj_patch_ids]
                obj_other_attention_score = torch.mean(obj_other_attention_scores).item()
                obj_attentions_outside[obj_name][other_obj_name].append(obj_other_attention_score)
    
        print(f"image: {image_path}, layer: {layer_id-1} Finished!")
    # print("obj_atentions_inside: ", obj_atentions_inside)
    # print("obj_attentions_outside: ", obj_attentions_outside)
    
    # draw the attention scores for each object
    for obj_name in objects_to_check:
        save_dir = os.path.join(image_save_dir, f"attns")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"attentions-{obj_name}.pdf")
        
        plt.figure()
        layer_ids = [i + 1 for i in range(vit_layer_num)]
        if not colors:
            # randomly select a color for each obj
            colors = {name: mcolors.to_hex(plt.cm.viridis(i / len(obj_bboxes))) for i, name in enumerate(obj_bboxes.keys())}
        # plt.plot(layer_ids, obj_attentions_cls[obj_name], color=colors[obj_name], linestyle="-.", label=f"attention: {obj_name} -> [cls]")
        plt.plot(layer_ids, obj_atentions_inside[obj_name], color=colors[obj_name], label=f"attention: {obj_name} -> {obj_name}")
        for key, val in obj_attentions_outside[obj_name].items():
            plt.plot(layer_ids, val, color=colors[key], linestyle="--", label=f"attention: {obj_name} -> {key}")
        
        plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
        plt.ylabel("Attention", color=mcolors.to_rgba('black', 1))
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        legend = plt.legend(framealpha=0.6, fontsize=8)
        for text in legend.get_texts():
            text.set_color(mcolors.to_rgba('black', 1))
        
        print(f"save_path: {save_path}")
        plt.savefig(save_path, bbox_inches='tight')

def check_similarity(
    device, 
    model_name, 
    image_path,
    object_names=None,
    colors=None,
):    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None

        # path
        save_dir = root_dir / "figures/check_obj_similarity"
        new_save_dir = os.path.join(save_dir, model_name)
        image_name = os.path.basename(image_path).split(".")[0]
        image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}")
        os.makedirs(image_save_dir, exist_ok=True)
            
        ## forward ViT
        vit_layer_num = 0
        if any(name in model_name for name in ["llava"]):
            replace_llava1_5_processor_forward_return_image_size()
            vit_layer_num = model.config.vision_config.num_hidden_layers
        elif any(name in model_name for name in ["intern"]):
            vit_layer_num = model.config.vision_config.num_hidden_layers
            
        ## Preparation for inference
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert('RGB')
        inputs, image_size = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs.to(model.device)
    
    # get attention scores
    with torch.no_grad():
        all_hidden_states = vit(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
        ).hidden_states  # (1 + layer_num) * (batch_size, seq_len, dim)
    
    # get objects info and filter the objects
    obj_bboxes = get_scene_info(image_path)
    draw_bbox(image_path, obj_bboxes, image_save_dir)
    if object_names:
        obj_bboxes = {name: bbox for name, bbox in obj_bboxes.items() if name in object_names}
    
    # get obj patches
    if True:
        # 1. get resized bbox
        # height, width = image_size  # after patching
        height, width = inputs["pixel_values"][0].shape[1:]  # after resize, before patching
        original_image = Image.open(image_path).convert('RGB')
        original_width, original_height = original_image.size  # square image
        scale = width / original_width
        print(f"original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
        # import pdb; pdb.set_trace()
        resized_bboxes = {}
        for obj_name, obj_bbox in obj_bboxes.items():
            x1, y1, x2, y2 = obj_bbox  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)  # 
            resized_bboxes[obj_name] = (x1, y1, x2, y2)
        print(f"image_size: {width}, {height}")
        
        # 2. the resized image is split into patches with self.patch_size as size. Now let's get the obj patches
        # get the coordinates of the patches in the whole image
        object_patch_ids = {}
        patch_size = model.config.vision_config.patch_size
        for obj_name, obj_bbox in resized_bboxes.items():
            obj_patch_ids = []
            x1, y1, x2, y2 = obj_bbox
            for i in range(int(width // patch_size)):
                for j in range(int(height // patch_size)):
                    x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                    if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                    # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                        obj_patch_ids.append(j * int(width // patch_size) + i)
            object_patch_ids[obj_name] = obj_patch_ids
    
    # check if the patches are correct
    # 1. check resized bboxes
    if True:
        height, width = inputs["pixel_values"][0].shape[1:]  # after resize, before patching
        # draw a blank image
        seg_map = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(seg_map)
        for obj_name, obj_bbox in resized_bboxes.items():
            # draw bbox
            x1, y1, x2, y2 = obj_bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # draw text
            text = obj_name
            left, top, right, bottom = draw.textbbox((0, 0), text)
            text_width, text_height = right - left, top - bottom
            text_x = x1 + (x2 - x1 - text_width) / 2
            text_y = y1 + (y2 - y1 - text_height) / 2
            draw.text((text_x, text_y), text, fill="red")
            # draw coordinates
            draw.text((x1, y1), f"({x1}, {y1})", fill="red")
            draw.text((x2, y2), f"({x2}, {y2})", fill="red")
            
        # save the image
        save_path = os.path.join(image_save_dir, f"resized_bboxes.png")
        seg_map.save(save_path)
        
        # 2. check the bbox patches
        height, width = image_size  #  after patching
        for obj_name, obj_patch_ids in object_patch_ids.items():
            token_names = []
            all_names = ["black", obj_name]
            for i in range(height * width):
                name = "black"
                if i in obj_patch_ids:
                    name = obj_name
                token_names.append(name)
                
            color_map = {}
            for i, label in enumerate(all_names):
                color_map[label] = mcolors.to_hex(plt.cm.viridis(i / 2))
                
            # create a blank image
            scale = model.config.vision_config.patch_size
            original_width = width * scale
            original_height = height * scale
            new_token_names = np.array([[None for _ in range(original_width)] for _ in range(original_height)])
            for j in range(len(token_names)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                x_range = slice(x * scale, (x + 1) * scale)
                y_range = slice(y * scale, (y + 1) * scale)
                for row in range(y_range.start, y_range.stop):
                    for col in range(x_range.start, x_range.stop):
                        new_token_names[row][col] = token_names[j]
            token_names = new_token_names.flatten()
            seg_map = Image.new("RGB", (original_width, original_height), (0, 0, 0))

            for j in range(len(token_names)):
                x = j % original_width
                y = j // original_width
                x, y = int(x), int(y)
                color = color_map[token_names[j]]
                rgb_color = mcolors.to_rgb(color)
                int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
                seg_map.putpixel((x, y), int_color)
                
            # 5. save the segmentation map
            save_path = os.path.join(image_save_dir, f"patched_img_bbox_{obj_name}.png")
            seg_map.save(save_path)
    
    # compute similarity scores
    import pdb; pdb.set_trace()
    obj_sims_inside = {name: [] for name in obj_bboxes.keys()}
    obj_sims_outside = {name: [] for name in obj_bboxes.keys()}
    for layer_id in range(vit_layer_num):
        # 1. get hidden states
        hidden_states = all_hidden_states[layer_id][0]  # the first sample: (len, dim)
        
        # 2. compute sim matrix
        hidden_states = hidden_states[1:, :]  # remove cls token
        hidden_states = torch.nn.functional.normalize(hidden_states, dim=-1)  # (len, dim)
        sim_matrix = torch.matmul(hidden_states, hidden_states.T)  # (len, len)
        sim_matrix = sim_matrix.float().cpu().numpy()
        rows, cols = np.triu_indices_from(sim_matrix, k=1)
        # sims = sim_matrix[rows, cols]
        # print(sims)
        # print("rows: ", rows)
        # print("cols: ", cols)
        # print("number of pairs: ", len(rows))
        # print("total sims: ", sim_matrix.shape)
        # import pdb; pdb.set_trace()

        # 2. For each object, compute the sims inside and outside it
        seq_len = hidden_states.shape[0]
        for obj_name, obj_patch_ids in object_patch_ids.items():
            
            # get the average sim between the patches of the obj
            # import pdb; pdb.set_trace()
            inside_sims = []
            for i in range(seq_len):
                for j in range(seq_len):
                    if i in obj_patch_ids and j in obj_patch_ids:
                        inside_sims.append(sim_matrix[i][j])
           
            # get sims between the patches of the obj and the rest of the image
            outside_sims = []
            for i in range(seq_len):
                for j in range(seq_len):
                    if i in obj_patch_ids and j not in obj_patch_ids:
                        outside_sims.append(sim_matrix[i][j])
            obj_sims_inside[obj_name].append(np.mean(inside_sims))
            obj_sims_outside[obj_name].append(np.mean(outside_sims))

        print(f"image: {image_path}, layer: {layer_id-1} Finished!")
    # print("obj_atentions_inside: ", obj_atentions_inside)
    # print("obj_attentions_outside: ", obj_attentions_outside)
    
    # draw the attention scores for each object
    plt.figure()
    layer_ids = [i for i in range(vit_layer_num)]
    if not colors:
        # randomly select a color for each obj
        colors = {name: mcolors.to_hex(plt.cm.viridis(i / len(obj_bboxes))) for i, name in enumerate(obj_bboxes.keys())}
    for key, val in obj_sims_inside.items():
        plt.plot(layer_ids, val, color=colors[key], label=key)
    for key, val in obj_sims_outside.items():
        plt.plot(layer_ids, val, color=colors[key], linestyle="--", label=f"{key} outside")
    # for key, val in obj_attentions_cls.items():
    #     plt.plot(layer_ids, val, color=colors[key], linestyle="-.", label=f"{key} [cls]")
    
    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel("Token Probability", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 1))
    
    save_path = os.path.join(image_save_dir, f"{image_name}-sims.pdf")
    print(f"save_path: {save_path}")
    plt.savefig(save_path, bbox_inches='tight')

def check_similarity_each_obj(
    device, 
    model_name, 
    image_path,
    objects_to_check=None,
    colors=None,
):  
    """
    More granular analysis than check_similarity. 
    """
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None

        # path
        save_dir = root_dir / "figures/check_obj_similarity"
        new_save_dir = os.path.join(save_dir, model_name)
        image_name = os.path.basename(image_path).split(".")[0]
        image_save_dir = os.path.join(new_save_dir, f"seg_img_{image_name}")
        os.makedirs(image_save_dir, exist_ok=True)
            
        ## forward ViT
        vit_layer_num = 0
        if any(name in model_name for name in ["llava"]):
            replace_llava1_5_processor_forward_return_image_size()
            vit_layer_num = model.config.vision_config.num_hidden_layers
        elif any(name in model_name for name in ["intern"]):
            vit_layer_num = model.config.vision_config.num_hidden_layers
            
        ## Preparation for inference
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path).convert('RGB')
        inputs, image_size = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs.to(model.device)
    
    # get attention scores
    with torch.no_grad():
        all_hidden_states = vit(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
        ).hidden_states  # (1 + layer_num) * (batch_size, seq_len, dim)
    
    # get objects info and filter the objects
    obj_bboxes = get_scene_info(image_path)
    draw_bbox(image_path, obj_bboxes, image_save_dir)
    
    # get obj patches
    if True:
        # 1. get resized bbox
        # height, width = image_size  # after patching
        height, width = inputs["pixel_values"][0].shape[1:]  # after resize, before patching
        original_image = Image.open(image_path).convert('RGB')
        original_width, original_height = original_image.size  # square image
        scale = width / original_width
        print(f"original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
        # import pdb; pdb.set_trace()
        resized_bboxes = {}
        for obj_name, obj_bbox in obj_bboxes.items():
            x1, y1, x2, y2 = obj_bbox  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)  # 
            resized_bboxes[obj_name] = (x1, y1, x2, y2)
        print(f"image_size: {width}, {height}")
        
        # 2. the resized image is split into patches with self.patch_size as size. Now let's get the obj patches
        # get the coordinates of the patches in the whole image
        object_patch_ids = {}
        patch_size = model.config.vision_config.patch_size
        for obj_name, obj_bbox in resized_bboxes.items():
            obj_patch_ids = []
            x1, y1, x2, y2 = obj_bbox
            for i in range(int(width // patch_size)):
                for j in range(int(height // patch_size)):
                    x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                    if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                    # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                        obj_patch_ids.append(j * int(width // patch_size) + i)
            object_patch_ids[obj_name] = obj_patch_ids
    
    # check if the patches are correct
    # 1. check resized bboxes
    if False:
        height, width = inputs["pixel_values"][0].shape[1:]  # after resize, before patching
        # draw a blank image
        seg_map = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(seg_map)
        for obj_name, obj_bbox in resized_bboxes.items():
            # draw bbox
            x1, y1, x2, y2 = obj_bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # draw text
            text = obj_name
            left, top, right, bottom = draw.textbbox((0, 0), text)
            text_width, text_height = right - left, top - bottom
            text_x = x1 + (x2 - x1 - text_width) / 2
            text_y = y1 + (y2 - y1 - text_height) / 2
            draw.text((text_x, text_y), text, fill="red")
            # draw coordinates
            draw.text((x1, y1), f"({x1}, {y1})", fill="red")
            draw.text((x2, y2), f"({x2}, {y2})", fill="red")
            
        # save the image
        save_path = os.path.join(image_save_dir, f"resized_bboxes.png")
        seg_map.save(save_path)
        
        # 2. check the bbox patches
        height, width = image_size  #  after patching
        for obj_name, obj_patch_ids in object_patch_ids.items():
            token_names = []
            all_names = ["black", obj_name]
            for i in range(height * width):
                name = "black"
                if i in obj_patch_ids:
                    name = obj_name
                token_names.append(name)
                
            color_map = {}
            for i, label in enumerate(all_names):
                color_map[label] = mcolors.to_hex(plt.cm.viridis(i / 2))
                
            # create a blank image
            scale = model.config.vision_config.patch_size
            original_width = width * scale
            original_height = height * scale
            new_token_names = np.array([[None for _ in range(original_width)] for _ in range(original_height)])
            for j in range(len(token_names)):
                x = j % width
                y = j // width
                x, y = int(x), int(y)
                x_range = slice(x * scale, (x + 1) * scale)
                y_range = slice(y * scale, (y + 1) * scale)
                for row in range(y_range.start, y_range.stop):
                    for col in range(x_range.start, x_range.stop):
                        new_token_names[row][col] = token_names[j]
            token_names = new_token_names.flatten()
            seg_map = Image.new("RGB", (original_width, original_height), (0, 0, 0))

            for j in range(len(token_names)):
                x = j % original_width
                y = j // original_width
                x, y = int(x), int(y)
                color = color_map[token_names[j]]
                rgb_color = mcolors.to_rgb(color)
                int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
                seg_map.putpixel((x, y), int_color)
                
            # 5. save the segmentation map
            check_dir = os.path.join(image_save_dir, "check")
            os.makedirs(check_dir, exist_ok=True)
            save_path = os.path.join(check_dir, f"patched_img_bbox_{obj_name}.png")
            seg_map.save(save_path)
    
    # compute similarity scores
    # import pdb; pdb.set_trace()
    object_inside_sims = {key: [] for key in objects_to_check}
    object_outside_sims = {}
    for key in objects_to_check:
        object_outside_sims[key] = {name: [] for name in obj_bboxes.keys() if name != key}
    for layer_id in range(vit_layer_num):
        # 1. get hidden states
        hidden_states = all_hidden_states[layer_id][0]  # the first sample: (len, dim)
        
        # 2. compute sim matrix
        hidden_states = hidden_states[1:, :]  # remove cls token
        hidden_states = torch.nn.functional.normalize(hidden_states, dim=-1)  # (len, dim)
        sim_matrix = torch.matmul(hidden_states, hidden_states.T)  # (len, len)
        sim_matrix = sim_matrix.float().cpu().numpy()
        # rows, cols = np.triu_indices_from(sim_matrix, k=1)

        # 2. For each object to check, compute the sims inside and outside it
        seq_len = hidden_states.shape[0]
        for obj_name, obj_patch_ids in object_patch_ids.items():
            if obj_name not in objects_to_check:
                continue
            
            # get the average sim between the patches of the obj
            # import pdb; pdb.set_trace()
            inside_sims = []
            for i in range(seq_len):
                for j in range(seq_len):
                    if i in obj_patch_ids and j in obj_patch_ids:
                        inside_sims.append(sim_matrix[i][j])
            # for (i, j) in zip(rows, cols):
            #     if i in obj_patch_ids and j in obj_patch_ids:
            #         inside_sims.append(sim_matrix[i][j])
            object_inside_sims[obj_name].append(np.mean(inside_sims))
           
            # get sims between the patches of the obj and the rest of the image
            for other_obj_name, other_obj_patch_ids in object_patch_ids.items():
                if other_obj_name == obj_name:
                    continue
                outside_sims_other = []
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i in obj_patch_ids and j in other_obj_patch_ids:
                            outside_sims_other.append(sim_matrix[i][j])
                # for (i, j) in zip(rows, cols):
                #     if i in obj_patch_ids and j in other_obj_patch_ids:
                #         outside_sims_other.append(sim_matrix[i][j])
                object_outside_sims[obj_name][other_obj_name].append(np.mean(outside_sims_other))

        print(f"image: {image_path}, object: {obj_name}, layer: {layer_id-1} Finished!")
    
    for obj_name in objects_to_check:
        pass
        # print(f"obj_name: {obj_name}, object_inside_sims: {object_inside_sims[obj_name]}")
        # print(f"obj_name: {obj_name}, object_outside_sims: {object_outside_sims[obj_name]}")
        # print("--------------------------------------------------")    

    # draw the attention scores for each object
    for obj_name in objects_to_check:
        save_dir = os.path.join(image_save_dir, f"sims")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sims_{obj_name}.pdf")
        
        plt.figure()
        layer_ids = [i + 1 for i in range(vit_layer_num)]
        if not colors:
            colors = {name: mcolors.to_hex(plt.cm.viridis(i / len(obj_bboxes))) for i, name in enumerate(obj_bboxes.keys())}
        
        plt.plot(layer_ids, object_inside_sims[obj_name], color=colors[obj_name], label=f"similarity: <{obj_name},{obj_name}>")
        plt.annotate(obj_name, xy=(layer_ids[-1], object_inside_sims[obj_name][-1]), xytext=(layer_ids[-1], object_inside_sims[obj_name][-1]), fontsize=7, ha='right', va='center')
        for key, val in object_outside_sims[obj_name].items():
            plt.plot(layer_ids, val, color=colors[key], linestyle="--", label=f"similarity: <{obj_name},{key}>")
            # mark the key at the end of the curve
            plt.annotate(f"{key}", xy=(layer_ids[-1], val[-1]), xytext=(layer_ids[-1], val[-1]), fontsize=7, ha='left', va='center')
        
        plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
        plt.ylabel("Similarities", color=mcolors.to_rgba('black', 1))
        
        # set y lim
        # plt.ylim(0, 1)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        legend = plt.legend(framealpha=0.6, fontsize=8)
        for text in legend.get_texts():
            text.set_color(mcolors.to_rgba('black', 1))
        
        print(f"save_path: {save_path}")
        plt.savefig(save_path, bbox_inches='tight')
    
    # draw the differences in attention scores
    for obj_name in objects_to_check:
        save_dir = os.path.join(image_save_dir, f"sims")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sims_{obj_name}_diff.pdf")
        
        plt.figure()
        layer_ids = [i + 1 for i in range(vit_layer_num)]
        if not colors:
            colors = {name: mcolors.to_hex(plt.cm.viridis(i / len(obj_bboxes))) for i, name in enumerate(obj_bboxes.keys())}
        
        for key, val in object_outside_sims[obj_name].items():
            diff = np.array(object_inside_sims[obj_name]) - np.array(val)
            plt.plot(layer_ids, diff, color=colors[key], linestyle="--", label=f"sim-diff: <{obj_name},{obj_name}> - <{key},{obj_name}>")
            # mark the key at the end of the curve
            plt.annotate(f"{key}", xy=(layer_ids[-1], diff[-1]), xytext=(layer_ids[-1], diff[-1]), fontsize=7, ha='left', va='center')
        
        plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
        plt.ylabel("Differences in Similarities", color=mcolors.to_rgba('black', 1))
        
        # set y lim
        # plt.ylim(0, 1)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        legend = plt.legend(framealpha=0.6, fontsize=8)
        for text in legend.get_texts():
            text.set_color(mcolors.to_rgba('black', 1))
        
        print(f"save_path: {save_path}")
        plt.savefig(save_path, bbox_inches='tight')

def test_token_truncation(
    model_name="llava1_5_7b",
    vision_decoder_path=None,
    dataset_name="GQA",
    device="cuda:1",
    batch_size=1,
    data_num=1000,
    random=False,
    tag="0",
    method="method_1"
):
    """
    Truncate the image tokens.
    """
    official_dataset_name = DATASET_NAME_TO_OFFICIAL.get(dataset_name, dataset_name)
    exp_save_dir = root_dir / "eval" / "results" / official_dataset_name /f"test_llm_image_token_truncation-{method}"
    save_dir = exp_save_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    functions_llava = {
        "method_1": replace_llava_1_5_token_truncation_by_logit_lens,
        "method_2": replace_llava_1_5_logit_lens_adaptive,
        "method_3": replace_llava_1_5_token_truncation_by_logit_lens_turnback,
        "method_4": replace_llava_1_5_token_truncation_by_logit_lens_runlength,
        "method_5": replace_llava_1_5_token_truncation_by_logit_lens_runlength_adaptive,
    }
    functions_qwen2 = {
        "method_5": replace_qwen2_vl_token_truncation_by_logit_lens_runlength_adaptive,
    }
    functions_qwen2_5 = {
        "method_5": replace_qwen2_5_vl_token_truncation_by_logit_lens_runlength_adaptive,
    }
    if "llava1_5" in model_name:
        monkey_patch_func = functions_llava[method]
        monkey_patch_func()
    elif "qwen2_5" in model_name:
        monkey_patch_func = functions_qwen2_5[method]
        monkey_patch_func()
    elif "qwen2" in model_name:
        monkey_patch_func = functions_qwen2[method]
        monkey_patch_func()
    else:
        pass
    
    model, processor, tokenizer = load_model(model_name, device, use_flash_attention=True)
    if "intern" in model_name:
        replace_intern2_5_vl_token_truncation_by_logit_lens_runlength_adaptive(model)
    
    if vision_decoder_path is not None:
        if "llava" in model_name:
            vision_decoder = model.language_model.model.vision_decoder
        elif "qwen" in model_name:
            vision_decoder = model.model.vision_decoder
        checkpoint = torch.load(vision_decoder_path, map_location=device)
        model_state_dict = checkpoint["model_state_dict"]
        vision_decoder.load_state_dict(model_state_dict)
        vision_decoder.to(device)
        vision_decoder.eval()

    # generate
    save_path = save_dir / f"truncation_res_{dataset_name}_{model_name}_{tag}.jsonl"
    truncation_ratio_dir = root_dir / "eval" / "results" / "share" / f"test_llm_image_token_truncation-method_5_1" / model_name
    truncation_ratio_dir.mkdir(parents=True, exist_ok=True)
    truncation_ratio_path = truncation_ratio_dir / f"truncation_ratio.jsonl"
    save_path = str(save_path)
    truncation_ratio_path = str(truncation_ratio_path)
    _ = generate_batch_responses(
        model,
        processor,
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_new_tokens=10,
        save_path=save_path,
        data_num=data_num,
        random=random
    )
    # calculate task performance
    ratios = []
    try:
        with jsonlines.open(truncation_ratio_path, "r") as f:
            for line in f:
                ratios.append(line["truncation_ratio"])
        avg_ratio = sum(ratios) / len(ratios)
    except:
        avg_ratio = None
    print(f"avg_ratio: {avg_ratio}")
    
    if dataset_name == "mme":
        res_dir = save_path.split('/')[:-1]
        res_dir = '/'.join(res_dir)
        two_total_scores = evaluate_mme(res_dir)
        shutil.rmtree(res_dir)
        
        return two_total_scores, avg_ratio
    
    if dataset_name in ["vqa"]:
        anno_path = data_dir / "VQA_v2/v2_mscoco_val2014_annotations.json"
        qn_path = data_dir / "VQA_v2/v2_OpenEnded_mscoco_val2014_questions.json"
        eval_vqa = evaluate_vqa(anno_path, qn_path, save_path)
        print(f"Evaluation result ({model_name}): {eval_vqa}")
        return eval_vqa, avg_ratio
    
    with jsonlines.open(save_path, "r") as f:
        if dataset_name in ["coco"]:
            eval_chair = evaluate_chair(save_path)
            print(f"Evaluation result ({model_name}): {eval_chair}")
            return eval_chair
        elif dataset_name in ["pope_random", "pope_popular", "pope_adversarial"]:
            labels = []
            preds = []
            for sample in f:
                pred = sample["pred"]
                if "llava" in model_name:
                    pred = pred.lower()
                preds.append(pred)
                labels.append(sample["answer"])
            acc = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, average='macro')
            recall = recall_score(labels, preds, average='macro')
            f1 = f1_score(labels, preds, average='macro')
            print(f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}")
        else:
            accs = []
            for sample in f:
                pred = sample["pred"]
                if dataset_name in ["mmb", "sqa", "whatsup_a", "whatsup_b", "cocoqa_1", "cocoqa_2", "gqa_1", "gqa_2", "whatsup_a_left_right", "whatsup_a_on_under", "whatsup_b_behind_in_front_of", "whatsup_b_left_right", "whatsup_a_on", "whatsup_a_under", "whatsup_b_behind", "whatsup_b_in_front_of"]:
                    if any(name in model_name for name in ["llava", "intern"]):
                        if len(pred) > 1:
                            if pred[1] == ':' or all(char == ' ' for char in pred[1:]):
                                pred = pred[0]
                            else:
                                pred = None   # "D: A bottle"
                        elif len(pred) == 1:
                            pred = pred[0]  # "D"
                        else:
                            pred = None
                    accs.append(pred == sample["answer"])
                elif dataset_name == "vsr":
                    if "True" in pred:
                        pred = 1
                    elif "False" in pred:
                        pred = 0
                    else:
                        pred = None
                    accs.append(pred == sample["answer"])
                elif dataset_name in ["GQA", "gqa_no_spatial"]:
                    if "llava" in model_name:
                        pred = pred.lower()
                    accs.append(pred == sample["answer"])
                elif dataset_name in ["textvqa"]:
                    if "llava" in model_name:
                        pred = pred.lower()
                    res = [pred == ans for ans in sample["answers"]]
                    acc = min(1, sum(res) / 3)
                    accs.append(acc)
            acc = sum(accs) / len(accs)
            print(f"acc: {acc}")
    
    print(f"avg_ratio: {avg_ratio}")
    if os.path.exists(truncation_ratio_path):
        with jsonlines.open(truncation_ratio_path, "w") as f:
            pass
    
    return acc, avg_ratio

def check_entity_tokens_equivalence(all_image_tokens, all_entity_names, all_entity_tokens):
    # bicycle, bike, bicycle
    for image_token in all_image_tokens:
        image_token = image_token.lower()
        for entity_name in all_entity_names:
            if image_token == entity_name:
                return True
        for entity_tokens in all_entity_tokens:
            if image_token == entity_tokens[0]:
                return True
    return False

def check_vit_hallucination(
    model_name,
    dataset_name,
    device, 
    batch_size=8,
    llm_layer_id=25,
):
    # path
    save_dir = root_dir / "figures/vit_halucination_layerwise"
    new_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(new_save_dir, exist_ok=True)
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        discard_vit_layers_func = None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
            # raise ValueError("Unsupported model type.")
            vision_process_func = None
            discard_vit_layers_func = replace_llava1_5_test_directions_vit_discard_layers
            vit_layer_num = model.config.vision_config.num_hidden_layers
            replace_llava1_5_return_image_mask()
            replace_llava1_5_processor_forward_return_image_size()
        elif "qwen" in model_name:
            vision_process_func = process_vision_info_qwen
            if "qwen2_5" in model_name:
                replace_qwen2_5_vl_return_image_mask()
            elif "qwen2" in model_name:
                pass
        elif "intern" in model_name:
            vision_process_func = load_image_intern
        else:
            pass

    # load data
    dataloader = load_data(dataset_name, model_name, processor, data_num=None, random=False, batch_size=batch_size)
    
    # load synonyms
    synonym_file = root_dir / "eval/CHAIR/utils/synonyms.txt"
    synonyms = open(synonym_file).readlines()
    synonyms = [s.strip().split(', ') for s in synonyms]
    synonyms = [list(set(item)) for item in synonyms]
    synonyms_tokens = []
    for item in synonyms:
        category_tokens = []
        for word in item:
            word_tokens = tokenizer.tokenize(word)
            word_tokens = [token.strip("▁") for token in word_tokens]
            category_tokens.append(word_tokens)
        synonyms_tokens.append(category_tokens)
    
    all_results = []
    for layer_id in range(vit_layer_num):
        # monkey patch
        layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
        discard_vit_layers_func(layer_ids_to_delete)
                
        # Start inference
        positive_samples_cnt = 0
        negative_samples_cnt = 0
        preds_strict = []
        preds_loose = []
        labels = []
        for batch in tqdm(dataloader):
            inputs, samples = batch
            
            ## Preparation for inference
            image_paths = [sample["image"] for sample in samples]
            model_inputs = get_model_inputs(
                model_name, 
                processor, 
                vision_process_func,
                image_paths,
                prompts=["describe the image and tell me what is the main object in the image"] * batch_size
            )
            
            if "llava" in model_name:
                inputs, image_size = model_inputs
            elif "qwen" in model_name:
                inputs = model_inputs
            else:
                pass
            inputs.to(model.device)
            
            # forward
            with torch.no_grad():
                outputs, image_mask = model(
                    **inputs,
                    return_dict=True,
                    output_hidden_states=True,
                )
            if "llava" in model_name:
                activations = llm.lm_head(outputs.hidden_states[llm_layer_id + 1])
            elif "qwen" in model_name:
                activations = model.lm_head(outputs.hidden_states[llm_layer_id + 1])
            else:
                pass
            
            # start processing
            for idx in range(len(samples)):
                
                # set save path
                qn = samples[idx]["question"]
                ans = samples[idx]["answer"]
                sample_id = samples[idx]["id"]
                pattern = r'Is there (a|an) ([a-zA-Z ]+) in the image\?'
                match = re.search(pattern, qn)
                entity_name = match.group(2)
                
                sample_dir = os.path.join(new_save_dir, f"{dataset_name}-cases-llm_layer_{llm_layer_id}", f"sample-id_{sample_id}-entity_{entity_name}-ans_{ans}")
                os.makedirs(sample_dir, exist_ok=True)
                shutil.copy(samples[idx]["image"], sample_dir)
                text_map_path = os.path.join(sample_dir, "text_map.svg")
                text_tokens_path = os.path.join(sample_dir, "text_tokens.jsonl")

                sample_image_mask = image_mask[idx].squeeze(-1)
                    
                # get text tokens
                # text_ids = torch.argmax(activations[0], dim=-1)  # (seq_len)
                text_vals_top_3, text_ids_top_3 = torch.topk(activations[idx], k=3, dim=-1)  # (seq_len, top_k)
                text_ids = text_ids_top_3[:, 0]
                text_ids_second = text_ids_top_3[:, 1]
                text_ids_third = text_ids_top_3[:, 2]
                
                all_tokens = [processor.tokenizer.decode(text_id) for text_id in text_ids]
                all_tokens_second = [processor.tokenizer.decode(text_id) for text_id in text_ids_second]
                all_tokens_third = [processor.tokenizer.decode(text_id) for text_id in text_ids_third]
                
                text_tokens = [all_tokens[i] for i in range(len(all_tokens)) if sample_image_mask[i]]
                text_tokens_second = [all_tokens_second[i] for i in range(len(all_tokens_second)) if sample_image_mask[i]]
                text_tokens_third = [all_tokens_third[i] for i in range(len(all_tokens_third)) if sample_image_mask[i]]
                
                assert len(all_tokens) == len(sample_image_mask)
                
                # save_text_tokens
                with jsonlines.open(text_tokens_path, "w") as f:
                    f.write({"probs_level": 1, "text_tokens": text_tokens})
                    f.write({"probs_level": 2, "text_tokens": text_tokens_second})
                    f.write({"probs_level": 3, "text_tokens": text_tokens_third})
                
                # check hallucination
                entity_category_id = None
                for category_id, names in enumerate(synonyms):
                    if entity_name in names:
                        entity_category_id = category_id
                        break
                all_entity_names = synonyms[entity_category_id]
                all_entity_tokens = synonyms_tokens[entity_category_id]
                
                strict_in = check_entity_tokens_equivalence(text_tokens, all_entity_names, all_entity_tokens)
                loose_in = check_entity_tokens_equivalence(text_tokens + text_tokens_second + text_tokens_third, all_entity_names, all_entity_tokens)
                
                ans = 1 if ans == "yes" else 0
                labels.append(ans)
                preds_strict.append(int(strict_in))
                preds_loose.append(int(loose_in))
                if ans == 1:
                    positive_samples_cnt += 1
                else:
                    negative_samples_cnt += 1
                
                # draw text token map
                if False:
                    for probs_id in range(3):
                        # create a blank image
                        if "llava" in model_name:
                            height, width = image_size  # after patching
                        elif "qwen" in model_name:
                            image_grid_thw = inputs["image_grid_thw"]
                            time, height, width = image_grid_thw[0].cpu().numpy()
                            # set to int32
                            height, width = int(height), int(width)
                            height = height // model.config.vision_config.spatial_merge_size
                            width = width // model.config.vision_config.spatial_merge_size
                        else:
                            pass
                        # print(f"image_size: {image_size}")
                        # import pdb; pdb.set_trace()
                        cell_width = 50
                        table_width, table_height = width * cell_width, height * cell_width
                        cell_width, cell_height = cell_width, cell_width

                        # Create SVG canvas
                        cell_size = cell_width 
                        svg_width = table_width
                        svg_height = table_height
                        dwg = svgwrite.Drawing(size=(f"{svg_width}px", f"{svg_height}px"))
                        
                        # Add white background
                        dwg.add(dwg.rect(
                            insert=(0, 0),
                            size=(f"{svg_width}px", f"{svg_height}px"),
                            fill="white"
                        ))
                        
                        # Draw grid
                        for x in range(0, svg_width + 1, cell_size):
                            dwg.add(dwg.line(start=(x, 0), end=(x, svg_height), stroke="black", stroke_width=1))
                        for y in range(0, svg_height + 1, cell_size):
                            dwg.add(dwg.line(start=(0, y), end=(svg_width, y), stroke="black", stroke_width=1))
                        
                        font_path = "/raid_sdd/lyy/font/SimHei.ttf"
                        try:
                            font = ImageFont.truetype(font_path, size=15)
                        except IOError:
                            font = None
                            
                        # Add text
                        text_tokens_to_draw = None
                        if probs_id == 0:
                            text_tokens_to_draw = text_tokens
                        elif probs_id == 1:
                            text_tokens_to_draw = text_tokens_second
                        elif probs_id == 2:
                            text_tokens_to_draw = text_tokens_third
                        else:
                            pass
                        for j, text in enumerate(text_tokens_to_draw):
                            x = j % width
                            y = j // width
                            x_pos = x * cell_size + cell_size / 2
                            y_pos = y * cell_size + cell_size / 2
                            
                            dwg.add(dwg.text(
                                text,
                                insert=(x_pos, y_pos),
                                fill="black",
                                font_family=font,  # Use your font
                                font_size="12px",
                                text_anchor="middle",
                                dominant_baseline="middle"
                            ))
                        
                        # Save SVG
                        dwg.saveas(text_map_path)
        
        # calculate accuracy, precision, recall, f1-score for strict and loose
        strict_acc = accuracy_score(labels, preds_strict)
        strict_precision = precision_score(labels, preds_strict)
        strict_recall = recall_score(labels, preds_strict)
        strict_f1 = f1_score(labels, preds_strict)
        loose_acc = accuracy_score(labels, preds_loose)
        loose_precision = precision_score(labels, preds_loose)
        loose_recall = recall_score(labels, preds_loose)
        loose_f1 = f1_score(labels, preds_loose)
        layer_res = {
            "layer_id": layer_id,
            "strict": {
                "acc": strict_acc,
                "precision": strict_precision,
                "recall": strict_recall,
                "f1": strict_f1,
            },
            "loose": {
                "acc": loose_acc,
                "precision": loose_precision,
                "recall": loose_recall,
                "f1": loose_f1,
            },
            "preds_strict": preds_strict,
            "preds_loose": preds_loose,
            "labels": labels,
        }
        all_results.append(layer_res)
    
    # save the results
    save_path = os.path.join(new_save_dir, f"vit_hallucination_results.jsonl")
    with jsonlines.open(save_path, mode='w') as f:
        for result in all_results:
            f.write(result)
    
    # plot the change in each metric with layers
    metrics = ["acc", "precision", "recall", "f1"]
    metric_names = ["accuracy", "precision", "recall", "F1-score"]
    for metric, metric_name in zip(metrics, metric_names):
        strict_values = [result["strict"][metric] for result in all_results]
        loose_values = [result["loose"][metric] for result in all_results]
        layer_ids = [result["layer_id"] + 1 for result in all_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(layer_ids, strict_values, label=f"{metric} (strict)", marker='o')
        plt.plot(layer_ids, loose_values, label=f"{metric} (loose)", marker='x')
        
        plt.xlabel("Layer ID")
        plt.ylabel(metric_name.capitalize())
        plt.xticks(layer_ids)
        plt.legend()
        
        save_fig_path = os.path.join(new_save_dir, f"{metric}_{dataset_name}.pdf")
        plt.savefig(save_fig_path)
        plt.close()

def plot_hallucination():
    data_path = root_dir / "figures/vit_halucination_layerwise/llava1_5_7b/vit_hallucination_results.jsonl"
    with jsonlines.open(data_path, "r") as f:
        all_results = [line for line in f]
    
    # plot the change in each metric with layers
    metrics = ["acc", "precision", "recall", "f1"]
    metric_names = ["accuracy", "precision", "recall", "F1-score"]
    for metric, metric_name in zip(metrics, metric_names):
        strict_values = [result["strict"][metric] for result in all_results]
        loose_values = [result["loose"][metric] for result in all_results]
        layer_ids = [result["layer_id"] + 1 for result in all_results]
        
        # preds_strict = [result["preds_strict"] for result in all_results]  # (n_layers, n_samples)
        # preds_loose = [result["preds_loose"] for result in all_results]
        # labels = all_results[0]["labels"]  # (n_samples)
        # strict_values = [f1_score(labels, preds_strict[i]) for i in range(len(all_results))]
        # loose_values = [f1_score(labels, preds_loose[i]) for i in range(len(all_results))]
        # # compute the standard error of the f1 scores in each layer (preds_strict[i] is the results in layer i (n_samples))
        # strict_values_std_error = [np.std([preds_strict[i][j] == labels[j] for j in range(len(labels))]) / np.sqrt(len(labels)) for i in range(len(all_results))]
        # loose_values_std_error = [np.std([preds_loose[i][j] == labels[j] for j in range(len(labels))]) / np.sqrt(len(labels)) for i in range(len(all_results))]

        plt.figure()
        plt.plot(layer_ids, strict_values, label=f"{metric} (strict)", marker='o', color='blue')
        plt.plot(layer_ids, loose_values, label=f"{metric} (loose)", marker='o', color='green')
        # plt.fill_between(layer_ids, np.array(strict_values) - np.array(strict_values_std_error), np.array(strict_values) + np.array(strict_values_std_error), color='blue', alpha=0.2)
        # plt.fill_between(layer_ids, np.array(loose_values) - np.array(loose_values_std_error), np.array(loose_values) + np.array(loose_values_std_error), color='orange', alpha=0.2)
        
        plt.xlabel("Layer ID")
        plt.ylabel(metric_name.capitalize())
        # plt.xticks(layer_ids)
        plt.legend()
        plt.tight_layout()
        
        dataset_name = "pope_random"
        new_save_dir = root_dir / "figures/vit_halucination_layerwise/llava1_5_7b"
        save_fig_path = os.path.join(new_save_dir, f"{metric}_{dataset_name}.pdf")
        plt.savefig(save_fig_path)
        plt.close()


if __name__ == "__main__":
    
    fig_dir = root_dir / "test_figs/gqa"
    # files_jpg = glob.glob(os.path.join(fig_dir, "*.jpg"))
    # files_png = glob.glob(os.path.join(fig_dir, "*.png"))
    # files = files_jpg + files_png
    # print(files)
    files = [
        "2354453.jpg",  # √ bear
        # "2338056.jpg",  # rice √
        # "2342218.jpg",  # night soccer x
        # "2369060.jpg",  # girl √
        # "2416755.jpg",  # dog √
        # "2332870.jpg",  # √ ski
        # "2339558.jpg",  # bird √
        # "2331146.jpg",  # zebra x
        # "2332040.jpg",  # cat √
        # "2332720.jpg",  # scooter √
        # "2333495.jpg",  # giraffe
    ]

    for file in files:
    
        # seg_with_activations(
        #     device="cuda:1",
        #     model_name="llava1_5_7b",  # "qwen2_5_vl"
        #     image_path=os.path.join(fig_dir, file),
        #     dim_reduction=False,
        #     standarization=False,
        #     normalization=True,
        # )
        
        # if info does not exist, use `write_labels=True, original_size=True, use_image_info=False`, else `write_labels=False, original_size=True, use_image_info=True`
        # seg_with_unembedding_tokens_qwen(
        #     device="cuda:2",
        #     model_name="qwen2_5_vl",
        #     image_path=os.path.join(fig_dir, file),
        #     original_size=True,
        #     unembed_llm_layer=-1
        # )
        
        break
        # if info does not exist, use `write_labels=True, original_size=True, use_image_info=False`, else `write_labels=False, original_size=True, use_image_info=True`
        seg_with_unembedding_tokens(
            device="cuda:3",
            model_name="llava1_5_7b",  #  llava1_5_7b
            image_path=os.path.join(fig_dir, file),
            original_size=True,
            unembed_llm_layer=24
        )
        
        # seg_with_self_attention(
        #     device="cuda:3",
        #     model_name="llava1_5_7b",
        #     image_path=os.path.join(fig_dir, file),
        #     original_size=False,
        # )
    
    # fig_dir = root_dir / "test_figs"
    # seg_with_unembedding_tokens(
    #     device="cuda:1",
    #     model_name="llava1_5_7b",  #  llava1_5_7b
    #     image_path=os.path.join(fig_dir, "teddy.jpg"),
    #     original_size=True,
    #     unembed_llm_layer=24
    # )
    # =====================================================================================
    # img_path = root_dir / "test_figs/gqa/2354453.jpg"
    # obj_bboxes = get_scene_info(img_path)
    # print(obj_bboxes.keys())
    # draw_bbox(img_path, obj_bboxes, save_dir=root_dir / "test_figs/bbox")
    
    # skis', 'ground', 'hat', 'clothes', 'child', 'clouds', 'ski
    # 'bear', 'log', 'rock', 'ear', 'eye', 'nose', 'water', 'paw', 'snow'
    # check_attention(
    #     device="cuda:0",
    #     model_name="llava1_5_7b",
    #     image_path=os.path.join(fig_dir, "2332870.jpg"),
    #     object_names=[],
    #     colors=[]
    # )
    # check_attention_each_obj(
    #     device="cuda:3",
    #     model_name="llava1_5_7b",
    #     image_path=os.path.join(fig_dir, "2354453.jpg"),
    #     objects_to_check=["bear", "log", "rock", "ear", "eye", "nose", "water", "paw", "snow"],
    #     colors=[]
    # )
    
    # 2354453: ["bear", "log", "rock", "ear", "eye", "nose", "water", "paw", "snow"]
    # 2332870: ['skis', 'ground', 'hat', 'clothes', 'child', 'clouds', 'ski']
    # check_similarity(
    #     device="cuda:1",
    #     model_name="llava1_5_7b",
    #     image_path=os.path.join(fig_dir, "2354453.jpg"),
    #     object_names=[],
    #     colors=[]
    # )
    # check_similarity_each_obj(
    #     device="cuda:2",
    #     model_name="llava1_5_7b",
    #     image_path=os.path.join(fig_dir, "2354453.jpg"),  # 2354453  2332870
    #     objects_to_check=["bear", "log", "rock", "ear", "eye", "nose", "water", "paw", "snow"],
    #     colors=[]
    # )  # 14040M
    
    # =====================================================================================
    # image_path = os.path.join(fig_dir, "2354453.jpg")
    # image_path = root_dir / "test_figs/teddy.jpg"
    
    # image_path = root_dir / "test_figs/pope/bad_case/llava_pope_adversarial_Is there a bus in the image?_no/1487.png"
    # image_path = root_dir / "test_figs/pope/bad_case/llava_pope_adversarial_Is there a dog in the image?_no/483.png"
    # image_path = root_dir / "test_figs/pope/bad_case/llava_pope_adversarial_Is there a fire hydrant in the image?_yes/2404.png"
    # image_path = root_dir / "test_figs/mme/code_0007.png"
    # image_path = root_dir / "test_figs/gqa/2332870.jpg"
    # check_emergence = False
    # select_unembedding_layer(device="cuda:3", model_name="qwen2_5_vl", image_path=image_path, check_emergence=check_emergence)  # "llava1_5_7b", "qwen2_5_vl"
    
    # select_unembedding_layer_batch(device="cuda:2", model_name="llava1_5_7b", batch_size=8, bi_dir_att=True)
    
    # check_vit_hallucination("llava1_5_7b", "pope_random", "cuda:1", batch_size=8, llm_layer_id=25)
    # plot_hallucination()
    
    # ------------------------------------- test_segmentation_map -------------------------------------
    # 2354453, 2332870, 2338056, 2339558, 2369060, 2416755
    # ["2354453", "2338056", "2339558", "2369060", "2416755", "2332040"]  2332720  2333495
    
    # fig_dir = root_dir / "test_figs/gqa"
    # image_ids = ["2354453"]  # bear
    # image_ids=["2354453", "2338056", "2339558", "2369060", "2416755", "2332040"]
    # image_ids = ["2332870"]
    # # fig_dir = root_dir / "test_figs/whatsup"
    # # image_ids = ["mug_left_of_plate"]
    # for image_id in image_ids:
        # seg_with_unembedding_tokens_svg(
        #     device="cuda:3",
        #     model_name="llava1_5_7b",
        #     image_path=os.path.join(fig_dir, f"{image_id}.jpg"),
        #     original_size=True,
        #     unembed_llm_layer=24,
        #     # delete_pos_embed=True,
        # )
        
    # analyze_semantic_segmentation(
    #     image_ids=["2354453", "2338056", "2339558", "2416755", "2332040"], 
    #     model_name="llava1_5_7b",
    #     delete_pos_embed=True,
    # )
    # "2354453", "2338056", "2339558", "2416755", "2332040"
    
    # ------------------------------------- test_token_truncation -------------------------------------
    # evaluate_mme(root_dir / "eval/MME/results/test_llm_image_token_truncation-method_5/qwen2_vl_7b")
    # model_name = "llava1_5_7b"  # "llava1_5_7b"  "qwen2_5_vl"
    # model_name = "llava1_5_7b"  # "llava1_5_7b"  "qwen2_5_vl"
    # if "llava" in model_name:
    #     vision_decoder_path = work_dir / "checkpoints_vision_decoder/llava1_5_7b-epoch2-bsz16-lr1e-4-alpha0.7-temp2.5-patience5/model_best.pt"
    # elif "qwen" in model_name:
    #     vision_decoder_path = work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr1e-4-alpha0.9-temp5.0-patience10/model_step_9000.pt"
    #     vision_decoder_path = work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr2e-4-warmup1000-alpha0.8-temp5.0-patience10/model_step_9000.pt"
    # else:
    #     pass
    # dataset_name = "textvqa"  # "GQA", "pope_random", "pope_popular", "pope_adversarial", "coco"
    # accs = []
    # ratios = []
    # for i in range(1):
    #     acc, avg_ratio = test_token_truncation(
    #         model_name=model_name,
    #         vision_decoder_path=vision_decoder_path,
    #         dataset_name=dataset_name,  # "GQA", "pope_random", "pope_popular", "pope_adversarial", "coco"
    #         device="cuda:3",
    #         batch_size=1,
    #         data_num=1000,
    #         random=True,
    #         tag=str(i),
    #         method="method_5"  # "method_1", "method_2", "method_3"
    #     )
    #     # GQA: data_num=1000, random=True
    #     # pope: data_num=None, random=False
    #     # coco: data_num=500, random=True
    #     accs.append(acc)
    #     ratios.append(avg_ratio)
    # try:
    #     avg_acc = sum(accs) / len(accs)
    # except:
    #     avg_acc = None
    # try:
    #     avg_ratio = sum(ratios) / len(ratios)
    # except:
    #     avg_ratio = None
    # print(f"model: {model_name}, dataset: {dataset_name}, avg_acc: {avg_acc}, avg_ratio: {avg_ratio}, accs: {accs}")
    

    # get bbox
    # image_path = work_dir / "test_figs/square/2354453.jpg"
    # obj_bboxes = get_scene_info(image_path)
    # image_save_dir = work_dir / "figures"
    # obj_names = ["rock"]
    # draw_bbox(image_path, obj_bboxes, image_save_dir, obj_names=obj_names)