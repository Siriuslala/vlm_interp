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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from qwen_vl_utils import process_vision_info as process_vision_info_qwen
from utils import load_image_intern
import clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import Counter

import sys
sys.path.append("/raid_sdd/lyy/Interpretability/lyy/mm")
from eval.data_utils import *
from patch.monkey_patch import *

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import svgwrite
from tqdm import tqdm
import glob
import json
import jsonlines
import os
import copy
import shutil
import string


MODEL_NAME_TO_PATH = {
    "qwen2_5_vl": "/raid_sdd/lyy/hf/Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "/raid_sdd/lyy/hf/Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_vl_7b": "/raid_sdd/lyy/hf/Qwen/Qwen2-VL-7B-Instruct",
    "qwen2_vl_2b": "/raid_sdd/lyy/hf/Qwen/Qwen2-VL-2B-Instruct",
    "llava1_5_7b": "/raid_sdd/lyy/hf/models--LLaVA-1.5-7B",
    "llava1_6_mistral_7b": "/raid_sdd/lyy/hf/models--LLaVA-NeXT-Mistral-7B",
    "internvl2_5_8b": "OpenGVLab/InternVL2_5-8B",
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
        root_dir="/raid_sdd/lyy/dataset/whatsup_vlms/data/whatsup_vlms_data"
        if dataset_name == "whatsup_a":
            dataset = Controlled_Images(image_preprocess=None, subset="A", root_dir=root_dir)    
        elif dataset_name == "whatsup_b":
            dataset = Controlled_Images(image_preprocess=None, subset="B", root_dir=root_dir)
        else:
            raise ValueError("Invalid dataset name.")
        data_collator = Controlled_Images_collate_function(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "cocoqa" in dataset_name:
        root_dir="/raid_sdd/lyy/dataset/whatsup_vlms/data/whatsup_vlms_data"
        if dataset_name == "cocoqa_1":
            dataset = COCO_QA(image_preprocess=None, subset='one', root_dir=root_dir)
        elif dataset_name == "cocoqa_2":
            dataset = COCO_QA(image_preprocess=None, subset='two', root_dir=root_dir)
        else:
            raise ValueError("Invalid dataset name.")
        data_collator = COCO_QA_collate_function(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "gqa" in dataset_name:
        if "no_spatial" in dataset_name:
            data_path = [
                "/raid_sdd/lyy/Interpretability/lyy/mm/utils/GQA_data/gqa_no_spatial.jsonl",
                "/raid_sdd/lyy/dataset/GQA/images/images"
            ]
            dataset = GQADataset(data_path, data_num=data_num, random_select=random)
            data_collator = GQACcollator(processor, vision_process_func=vision_process_func, model_name=model_name)
        elif "spatial" in dataset_name:
            data_path = [
                "/raid_sdd/lyy/Interpretability/lyy/mm/utils/GQA_data/gqa_spatial.jsonl",
                "/raid_sdd/lyy/dataset/GQA/images/images"
            ]
            dataset = GQADataset(data_path, data_num=data_num, random_select=random)
            data_collator = GQACcollator(processor, vision_process_func=vision_process_func, model_name=model_name)
        else:
            root_dir="/raid_sdd/lyy/dataset/whatsup_vlms/data/whatsup_vlms_data"
            if dataset_name == "gqa_1":
                dataset = VG_QA(image_preprocess=None, subset='one', root_dir=root_dir)
            elif dataset_name == "gqa_2":
                dataset = VG_QA(image_preprocess=None, subset='two', root_dir=root_dir)
            else:
                raise ValueError("Invalid dataset name.")
            data_collator = VG_QA_collate_function(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif dataset_name == "vsr":
        text_path = "/raid_sdd/lyy/dataset/visual-spatial-reasoning/data/splits/random/test.jsonl"
        image_path = "/raid_sdd/lyy/dataset/visual-spatial-reasoning/data/images/test"
        data_path = [text_path, image_path]
        dataset = VSRDataset(data_path=data_path, data_num=None, random_select=random)
        data_collator = VSRCollator(processor, vision_process_func=vision_process_func, model_name=model_name)
    elif "GQA" in dataset_name:
        data_path = [
            "/raid_sdd/lyy/dataset/GQA/questions/testdev_balanced_questions.json",
            "/raid_sdd/lyy/dataset/GQA/images/images"
        ]
        qa_mode = kwargs.get("QA_mode", True)
        data_collator = GQACcollator(processor, vision_process_func=vision_process_func, model_name=model_name, QA_mode=qa_mode)
        random=True
        dataset = GQADataset(data_path, data_num=data_num, random_select=random)
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
        processor = AutoProcessor.from_pretrained(model_dir, padding_side='left', use_fast=True)
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
):
    image_id = os.path.basename(image_path).split(".")[0]
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

def draw_all_bboxes(
    image_path, 
    bbox_path="/raid_sdd/lyy/dataset/GQA/objects", 
    link_path="/raid_sdd/lyy/dataset/GQA/objects/gqa_objects_info.json",
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
    train_scene_path = "/raid_sdd/lyy/dataset/GQA/scene/train_sceneGraphs.json"
    val_scene_path = "/raid_sdd/lyy/dataset/GQA/scene/val_sceneGraphs.json"
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
            # print(f"image: {samples['image_path']}, choices: {samples['caption_options']}, pred: {out}")
            print(f"pred: [{out}], gold_ans: {sample["answer"]}")
        
        preds.extend(outputs)  
        
    with jsonlines.open(save_path, "w") as f:
        for pred, sample in zip(preds, all_samples):
            sample.update({"pred": pred})
            f.write(sample)
               
    return preds

def test_llm_image_bidirectional_attention(
    model_name="qwen2_5_vl",
    dataset_name="GQA",
    device="cuda:1",
    batch_size=8,
    include_vision_tokens=False,  # whether or not to delete <|vision_start|> and <|vision_end|>
    data_num=1000,
    random=False,
    tag="0"
):
    """
    Change the attention at LLM stage at image positions to bidirectional attention.
    """
    exp_save_dir = "/raid_sdd/lyy/Interpretability/lyy/mm/eval/GQA/results/test_llm_image_bidirectional_attention"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device, use_flash_attention=False)
    
    if "llava1_5" in model_name:
        pass
        replace_llava1_5_llm_image_bidirectional_attention(include_vision_tokens=include_vision_tokens)
    elif "qwen2_5" in model_name:
        replace_qwen2_5_vl_llm_image_bidirectional_attention(include_vision_tokens=include_vision_tokens)
    elif "qwen2" in model_name:
        replace_qwen2_vl_llm_image_bidirectional_attention(include_vision_tokens=include_vision_tokens)
    elif "intern" in model_name:
        replace_intern2_5_vl_llm_image_bidirectional_attention(model)
    else:
        pass
    
    # generate
    save_path = os.path.join(save_dir, f"llm_bidirectional_image_attention-{model_name}_{tag}.jsonl")
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
    with jsonlines.open(save_path, "r") as f:
        accs = []
        for sample in f:
            pred = sample["pred"]
            if dataset_name in ["whatsup_a", "whatsup_b", "cocoqa_1", "cocoqa_2", "gqa_1", "gqa_2", "whatsup_a_left_right", "whatsup_a_on_under", "whatsup_b_behind_in_front_of", "whatsup_b_left_right", "whatsup_a_on", "whatsup_a_under", "whatsup_b_behind", "whatsup_b_in_front_of"]:
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
            elif dataset_name in ["GQA", "gqa_no_spatial", "gqa_spatial"]:
                if "llava" in model_name:
                    pred = pred.lower()
                accs.append(pred == sample["answer"])
        acc = sum(accs) / len(accs)
            
    print(f"acc: {acc}")
    return acc
    
def test_llm_image_no_attention(
    model_name="qwen2_5_vl",
    dataset_name="GQA",
    device="cuda:1",
    batch_size=8,
    include_vision_tokens=False,  # whether or not to delete <|vision_start|> and <|vision_end|>
    data_num=1000,
    random=False,
    tag="0"
):
    """
    Block the attention at LLM stage at image positions.
    """
    exp_save_dir = "/raid_sdd/lyy/Interpretability/lyy/mm/eval/GQA/results/test_llm_image_no_attention"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device, use_flash_attention=False)
    
    if "llava1_5" in model_name:
        pass
        replace_llava1_5_llm_image_no_attention(include_vision_tokens=include_vision_tokens)
    elif "qwen2_5" in model_name:
        replace_qwen2_5_vl_llm_image_no_attention(include_vision_tokens=include_vision_tokens)
    elif "qwen2" in model_name:
        replace_qwen2_vl_llm_image_no_attention(include_vision_tokens=include_vision_tokens)
    elif "intern" in model_name:
        replace_intern2_5_vl_llm_image_no_attention(model)
    else:
        pass
    
    # generate
    save_path = os.path.join(save_dir, f"llm_no_image_attention-{model_name}_{tag}.jsonl")
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
    with jsonlines.open(save_path, "r") as f:
        accs = []
        for sample in f:
            pred = sample["pred"]
            if dataset_name in ["whatsup_a", "whatsup_b", "cocoqa_1", "cocoqa_2", "gqa_1", "gqa_2", "whatsup_a_left_right", "whatsup_a_on_under", "whatsup_b_behind_in_front_of", "whatsup_b_left_right", "whatsup_a_on", "whatsup_a_under", "whatsup_b_behind", "whatsup_b_in_front_of"]:
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
            elif dataset_name in ["GQA", "gqa_no_spatial", "gqa_spatial"]:
                if "llava" in model_name:
                    pred = pred.lower()
                accs.append(pred == sample["answer"])
        acc = sum(accs) / len(accs)
            
    print(f"acc: {acc}")
    return acc

def test_vit_attention_pattern_dimension_group():
    """
    Focus on one attention layer, and calculate the attention pattern from satellite to the nucleus dimension group-wise.
    Check whether the four directions result in different attention patterns, and what the difference is between low and high dimensions.
    """

def test_vit_attention_distribution_change():
    """
    Test the change in the attention distribution over ViT layers, and check if the attention distribution will stop to change at some layer.
    """


if __name__ == "__main__":
    
    # ------------------------------------- test_llm_image_bidirectional_attention
    # model_name = "qwen2_5_vl"  # qwen2_5_vl, qwen2_5_vl_3b, qwen2_vl_2b, llava1_5_7b, internvl2_5_8b
    # accs = []
    # for i in range(10):
    #     acc = test_llm_image_bidirectional_attention(
    #         model_name=model_name,  # "qwen2_5_vl", llava1_5_7b, internvl2_5_8b
    #         dataset_name="GQA",
    #         device="cuda:0",
    #         batch_size=8,
    #         include_vision_tokens=False,  # whether or not to include <|vision_start|> and <|vision_end|>
    #         random=True,
    #         tag=str(i),
    #     )
    #     accs.append(acc)
    # avg_acc = sum(accs) / len(accs)
    # print(f"model: {model_name}, avg_acc: {avg_acc}, accs: {accs}")
    
    # GQA (train)
    # qwen2_vl_2b: 0.695, 0.708, 0.711, 0.685, 0.698, 0.667, 0.659, 0.653, 0.706, 0.712
    # qwen2_vl_7b: 0.7, 0.708, 0.689, 0.702, 0.745, 0.729, 0.721, 0.739, 0.717, 0.715
    # qwen2_5_vl_3b: 0.679, 0.661, 0.682, 0.698, 0.675, 0.664, 0.685, 0.657, 0.693, 0.677
    # qwen2_5_vl_7b: 0.674, 0.667, 0.664, 0.68, 0.674, 0.668, 0.648, 0.666, 0.641, 0.649
    # llava1_5_7b: 0.644, 0.678, 0.656, 0.663, 0.677, 0.689, 0.629, 0.68, 0.662, 0.656
    # internvl2_5_8b: 0.75, 0.745, 0.782, 0.757, 0.738, 0.747, 0.752, 0.744, 0.742, 0.741
    # Whatsup_B
    # qwen2_vl_2b: 0.5563725490196079, 0.5245098039215687, 0.5637254901960784, 0.553921568627451, 0.5465686274509803, 0.5343137254901961, 0.5441176470588235, 0.5318627450980392, 0.5588235294117647, 0.5490196078431373
    # qwen2_vl_7b: 0.8725490196078431, 0.8872549019607843, 0.9068627450980392, 0.8676470588235294, 0.8651960784313726, 0.875, 0.8651960784313726, 0.8897058823529411, 0.8897058823529411, 0.8897058823529411
    # qwen2_5_vl_3b: 
    # qwen2_5_vl_7b: 
    # llava1_5_7b: 
    # internvl2_5_8b: 
    # gqa_no_spatial
    # qwen2_vl_2b: (0.581) 0.578, 0.586, 0.596, 0.574, 0.574, 0.576, 0.574, 0.578, 0.578, 0.599
    # qwen2_vl_7b: 0.618, 0.569, 0.615, 0.604, 0.606, 0.58, 0.59, 0.59, 0.594, 0.572
    # gqa_spatial
    # qwen2_vl_2b: 
    # qwen2_vl_7b: 
    
    # ------------------------------------- test_llm_image_no_attention
    model_name = "qwen2_5_vl"  # qwen2_5_vl, qwen2_5_vl_3b, qwen2_vl_2b, llava1_5_7b, internvl2_5_8b
    accs = []
    for i in range(10):
        acc = test_llm_image_no_attention(
            model_name=model_name,  # "qwen2_5_vl", llava1_5_7b, internvl2_5_8b
            dataset_name="GQA",
            device="cuda:1",
            batch_size=8,
            include_vision_tokens=False,  # whether or not to include <|vision_start|> and <|vision_end|>
            random=True,
            tag=str(i),
        )
        accs.append(acc)
    avg_acc = sum(accs) / len(accs)
    print(f"model: {model_name}, avg_acc: {avg_acc}, accs: {accs}")
    # GQA (train)
    # qwen2_vl_2b: 
    # qwen2_vl_7b: 
    # qwen2_5_vl_3b: 
    # qwen2_5_vl_7b: 
    # llava1_5_7b: 
    # internvl2_5_8b: 