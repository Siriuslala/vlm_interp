"""
Test pos embed. 
"""
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoModel,
    AutoTokenizer, 
    AutoProcessor
)
from peft import PeftModel, PeftModelForCausalLM

from qwen_vl_utils import process_vision_info as process_vision_info_qwen
from modelscope import snapshot_download
import clip
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
from utils import load_image_intern
from model.qwen2_vl_rope_scaling import Qwen2VLForConditionalGeneration_rope_scaling, Qwen2_5_VLForConditionalGeneration_rope_scaling

from typing import List, Dict, Union
from tqdm import tqdm, trange
import jsonlines
import json
import glob
import numpy as np
import shutil
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

def get_direction_options(dataset_name):
    options = ["left", "right", "on", "under", "in_front_of", "behind"]
    ret = []
    for option in options:
        if option in dataset_name:
            if option == "in_front_of":
                ret.append("in front of")
            else:
                ret.append(option)
    return ret

def load_model(model_name, model_path, device):
    model, processor, tokenizer = None, None, None
    
    if "qwen" in model_name:
        if model_path is None or "lora" in model_path:
            model_dir = MODEL_NAME_TO_PATH[model_name]
        else:
            model_dir = model_path
        model_dir_official = MODEL_NAME_TO_PATH[model_name]
        
        if model_path is None or "normal" in model_path:
            model_class = Qwen2_5_VLForConditionalGeneration if "qwen2_5" in model_name else Qwen2VLForConditionalGeneration
        else:
            model_class = Qwen2_5_VLForConditionalGeneration_rope_scaling if "qwen2_5" in model_name else Qwen2VLForConditionalGeneration_rope_scaling
        model = model_class.from_pretrained(
            model_dir, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        if model_path is not None and "lora" in model_path:
            model = PeftModel.from_pretrained(model, model_id=model_path, device_map={"": device}).merge_and_unload()
        model.to(device)
        
        if model_path is not None and "poly" in model_path:
            # load poly scaling params
            pattern = r"poly_alpha_(\d+)-poly_p_(\d+)"
            match = re.search(pattern, model_path)
            if match:
                poly_alpha = match.group(1)
                poly_p = match.group(2)
            model.visual.rotary_pos_emb.poly_alpha = float(poly_alpha)
            model.visual.rotary_pos_emb.poly_p = float(poly_p)

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_dir_official, min_pixels=min_pixels, max_pixels=max_pixels, padding_side='left'
        )  # left padding: <|endoftext|> 151644
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side='left',
            use_fast=True,
        )
    elif "ViT" in model_name:
        model, processor = clip.load(model_name, device=device)
        tokenizer = None
    elif "llava" in model_name:
        model_dir = MODEL_NAME_TO_PATH[model_name]
        MODEL_CLASS = LlavaForConditionalGeneration if "llava1_5" in model_name else LlavaNextForConditionalGeneration
        model = MODEL_CLASS.from_pretrained(
            model_dir, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
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
            attn_implementation="flash_attention_2",
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
                
        # collect results
        for out, sample in zip(outputs, samples):
            
            if "coco" in dataset_name and "qa" not in dataset_name:
                print("----"*20)
                print(f"[caption_pred]: {out}")
                save_dir = save_path.rsplit("/", 1)[0]
                image_id = str(sample["image_id"])
                sample_save_dir = os.path.join(save_dir, "cases", image_id)
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
                
                # if "pope" in dataset_name and not out.lower() == sample["answer"]:
                #     bad_case_dir = root_dir / "test_figs/pope/bad_case"
                #     data_type = dataset_name.split("_")[-1]  # e.g., "random", "popular", "adversarial"
                #     image_path = os.path.join(fdata_dir / "POPE/images/{data_type}", sample["image"])
                #     sample_dir = os.path.join(bad_case_dir, f"{model_name}_{dataset_name}_{sample["question"]}_{sample["answer"]}")
                #     os.makedirs(sample_dir, exist_ok=True)
                #     shutil.copy(image_path, sample_dir)
                #     print(f"Bad case saved: {sample_dir}")
        
        preds.extend(outputs)  
        
    with jsonlines.open(save_path, "w") as f:
        for pred, sample in zip(preds, all_samples):
            if "coco" in dataset_name and "qa" not in dataset_name:
                sample.update({"caption": pred})
            elif dataset_name in ["vqa"]:
                sample.update({"answer": pred})
            elif "mme" in dataset_name:
                continue
            else:
                sample.update({"pred": pred})
            f.write(sample)
               
    return preds

def test_normal(
    model_name="qwen2_5_vl",
    model_path=None,
    dataset_name="GQA",
    device="cuda:1",
    tag="2",
    data_num=1000, 
    batch_size=8,
    random=False,  # 是否随机从测试集里采样
    ):
    model, processor, tokenizer = load_model(model_name, model_path, device)
    
    official_dataset_name = DATASET_NAME_TO_OFFICIAL.get(dataset_name, dataset_name)
    save_dir = root_dir / "eval" / "results" / official_dataset_name / "test_normal"
    save_dir = save_dir / f"{model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"normal-{model_name}_{tag}.jsonl"
    save_path = str(save_path)

    # import pdb; pdb.set_trace()
    results = generate_batch_responses(
        model, 
        processor, 
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_new_tokens=10, 
        save_path=save_path,
        data_num=data_num, 
        random=random,
    )
    
    if dataset_name == "mme":
        res_dir = save_path.split('/')[:-1]
        res_dir = '/'.join(res_dir)
        evaluate_mme(res_dir)
        return 1

    if dataset_name in ["coco"]:
        eval_chair = evaluate_chair(save_path)
        print(f"Evaluation result ({model_name}): {eval_chair}")
        return eval_chair
    
    if dataset_name in ["vqa"]:
        anno_path = data_dir / "VQA_v2/v2_mscoco_val2014_annotations.json"
        anno_path = str(anno_path)
        qn_path = data_dir / "VQA_v2/v2_OpenEnded_mscoco_val2014_questions.json"
        qn_path = str(qn_path)
        eval_vqa = evaluate_vqa(anno_path, qn_path, save_path)
        print(f"Evaluation result ({model_name}): {eval_vqa}")
        return eval_vqa
    
    with jsonlines.open(save_path, "r") as f:
        if dataset_name in ["pope_random", "pope_popular", "pope_adversarial"]:
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
                elif dataset_name in ["GQA", "gqa_no_spatial", "gqa_spatial"]:
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
            print(f"acc: {acc}")  # 0.612, 0.607, 0.624, 0.617, 0.629, 0.614
    
            return acc
    
def test_delete_pos_embed_qwen(
    model_name="qwen2_5_vl",
    dataset_name="gqa",
    device="cuda:1",
    region="vit",  # the region to delete pos embed (vit / llm / vit_and_llm)
    layer_by_layer=False,  # 逐层删除，否则从 i 层以后全删 delete layer by layer, or delete pos embed in layer i and after
    layerwise=True,
    data_num=1000, 
    random=False,  # 是否随机从测试集里采样
    ):
    
    exp_save_dir = root_dir / "eval/GQA/results/test_delete_pos_embed"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device)
    
    def generate_and_record(model, processor, dataset_name, settings, save_path, final_path):
        # generate
        _ = generate_batch_responses(
            model, 
            processor, 
            dataset_name=dataset_name, 
            max_new_tokens=10, 
            save_path=save_path,
            data_num=data_num,
            random=random,
        )
        # calculate task performance
        acc = []
        with jsonlines.open(save_path, "r") as f:
            accs = []
            for sample in f:
                accs.append(sample["pred"] == sample["answer"])
        acc = sum(accs) / len(accs)
        if final_path:
            with jsonlines.open(final_path, "a") as f:
                f.write({
                    "settings": settings,
                    "acc": acc
                })
        return acc
    
    if layerwise:  # check each layer
        tag_layer_by_layer = "layer_by_layer" if layer_by_layer else "and_all"
        final_path = os.path.join(save_dir, f"del_pos_embed-{model_name}.jsonl")
        if layer_by_layer:
            if region == "vit":        
                for i in trange(model.config.vision_config.depth):
                    settings = {"region": region, "layer_id": i, "layer_by_layer": tag_layer_by_layer}
                    save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-layer_{i}-{tag_layer_by_layer}.jsonl")
                    # patching
                    layer_ids = [i,]
                    replace_qwen2_5_vl_delete_vit_pos_embed(layer_ids_to_delete=layer_ids)
                    # generate
                    acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
            elif region == "llm":
                for i in trange(model.config.num_hidden_layers):
                    settings = {"region": region, "layer_id": i, "layer_by_layer": tag_layer_by_layer}
                    save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-layer_{i}-{tag_layer_by_layer}.jsonl")
                    # patching
                    layer_ids = [i,]
                    replace_qwen2_5_vl_delete_llm_pos_embed(layer_ids_to_delete=layer_ids)
                    # generate
                    acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
            else:
                raise ValueError("Invalid region name.")
        else:
            if region == "vit":        
                for i in trange(model.config.vision_config.depth):
                    settings = {"region": region, "layer_id": i, "layer_by_layer": tag_layer_by_layer}
                    save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-layer_{i}-{tag_layer_by_layer}.jsonl")
                    # patching
                    layer_ids = [i for i in range(i, model.config.vision_config.depth)]
                    if "qwen2_5" in model_name:
                        replace_qwen2_5_vl_delete_vit_pos_embed(layer_ids_to_delete=layer_ids)
                    else:
                        replace_qwen2_vl_delete_vit_pos_embed(layer_ids_to_delete=layer_ids)
                    # generate
                    acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
            elif region == "llm":
                for i in trange(model.config.num_hidden_layers):
                    settings = {"region": region, "layer_id": i, "layer_by_layer": tag_layer_by_layer}
                    save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-layer_{i}-{tag_layer_by_layer}.jsonl")
                    # patching
                    layer_ids = [i for i in range(i, model.config.num_hidden_layers)]
                    replace_qwen2_5_vl_delete_llm_pos_embed(layer_ids_to_delete=layer_ids)
                    # generate
                    acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
            elif region == "vit_and_llm":
                # suppose "region='llm'"" is already done
                for i in trange(model.config.vision_config.depth):
                    settings = {"region": region, "layer_id": i, "layer_by_layer": tag_layer_by_layer}
                    save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-layer_{i}-{tag_layer_by_layer}.jsonl")
                    # patching
                    layer_ids = [i for i in range(i, model.config.vision_config.depth)]
                    replace_qwen2_5_vl_delete_vit_pos_embed(layer_ids_to_delete=layer_ids)
                    llm_layer_ids = [i for i in range(model.config.num_hidden_layers)]
                    replace_qwen2_5_vl_delete_llm_pos_embed(layer_ids_to_delete=llm_layer_ids)
                    # generate
                    acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
            else:
                raise ValueError("Invalid region name.")
    else:
        if region == "vit":        
            save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-all.jsonl")
            # patching
            layer_ids = [i for i in range(0, model.config.vision_config.depth)]
            if "qwen2_5" in model_name:
                replace_qwen2_5_vl_delete_vit_pos_embed(layer_ids_to_delete=layer_ids)
            else:
                replace_qwen2_vl_delete_vit_pos_embed(layer_ids_to_delete=layer_ids)
            # generate
            acc = generate_and_record(model, processor, dataset_name, settings=None, save_path=save_path, final_path=None)
        elif region == "llm":
            save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-all.jsonl")
            # patching
            layer_ids = [i for i in range(0, model.config.num_hidden_layers)]
            replace_qwen2_5_vl_delete_llm_pos_embed(layer_ids_to_delete=layer_ids)
            # generate
            acc = generate_and_record(model, processor, dataset_name, settings=None, save_path=save_path, final_path=None)
        elif region == "vit_and_llm":
            pass
        
    print(f"acc: {acc}")   
    return acc

def test_delete_pos_embed(
    model_name="llava1_5_7b",
    dataset_name="GQA",
    device="cuda:1",
    region="vit",  # 删除的区域 (vit / llm / vit_and_llm)
    batch_size=8,
    data_num=1000, 
    random=False,  # 是否随机从测试集里采样
    ):
    
    exp_save_dir = root_dir / "eval/GQA/results/test_delete_pos_embed"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device)
    
    
    def generate_and_record(model, processor, dataset_name, settings, save_path, final_path):
        # generate
        _ = generate_batch_responses(
            model, 
            processor, 
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_new_tokens=10, 
            save_path=save_path,
            data_num=data_num,
            random=random,
        )
        # calculate task performance
        acc = []
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
            
        with jsonlines.open(final_path, "a") as f:
            f.write({
                "settings": settings,
                "acc": acc
            })
        return acc
    
    final_path = os.path.join(save_dir, f"del_pos_embed-{model_name}.jsonl")
    
    if region == "vit":        
        settings = {"region": region}
        save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}.jsonl")
        # patching
        if "llava1_5" in model_name or "llava1_6" in model_name:
            replace_llava1_5_vl_delete_vit_pos_embed()
        elif "intern" in model_name:
            replace_intern2_5_delete_vit_pos_embed(model)
        # generate
        acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
    elif region == "llm":
        layer_ids = model.config.text_config.num_hidden_layers
        for layer_id in range(layer_ids):
            settings = {"region": region, "layer_id": layer_id}
            save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-in_{region}-layer_{layer_id}.jsonl")
            # patching
            layer_ids = [i for i in range(layer_id, model.config.num_hidden_layers)]
            if "llava1_5" in model_name or "llava1_6" in model_name:
                # replace_llava1_5_vl_delete_llm_pos_embed(layer_ids_to_delete=layer_ids)
                pass
            elif "intern" in model_name:
                # replace_intern2_5_delete_llm_pos_embed(layer_ids_to_delete=layer_ids)
                pass
            # generate
            acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
    else:
        raise ValueError("Invalid region name.")
    print(f"acc: {acc}")
    
    return acc

def test_delete_image_pos_embed_qwen(
    model_name="qwen2_5_vl",
    dataset_name="gqa",
    device="cuda:1",
    layer_by_layer=False,  # 逐层删除，否则从 i 层以后全删
    data_num=1000, 
    random=False,  # 是否随机从测试集里采样
    tag="0"
):
    
    exp_save_dir = root_dir / f"eval/GQA/results/test_delete_image_pos_embed_{tag}"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor = load_model(model_name, device)
    
    def generate_and_record(model, processor, dataset_name, settings, save_path, final_path):
        # generate
        _ = generate_batch_responses(
            model, 
            processor, 
            dataset_name=dataset_name, 
            max_new_tokens=10, 
            save_path=save_path,
            data_num=data_num,
            random=random,
        )
        # calculate task performance
        acc = []
        with jsonlines.open(save_path, "r") as f:
            accs = []
            for sample in f:
                accs.append(sample["pred"] == sample["answer"])
            acc = sum(accs) / len(accs)
        with jsonlines.open(final_path, "a") as f:
            f.write({
                "settings": settings,
                "acc": acc
            })
    
    tag_layer_by_layer = "layer_by_layer" if layer_by_layer else "and_all"
    final_path = os.path.join(save_dir, f"del_image_pos_embed-{model_name}.jsonl")
    
    if layer_by_layer:
        for i in trange(model.config.num_hidden_layers):
            settings = {"layer_id": i, "layer_by_layer": tag_layer_by_layer}
            save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-layer_{i}-{tag_layer_by_layer}.jsonl")
            # patching
            layer_ids = [i,]
            replace_qwen2_5_vl_delete_llm_image_pos_embed(layer_ids_to_delete=layer_ids)
            # generate
            generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
    else:
        for i in trange(model.config.num_hidden_layers):
            settings = { "layer_id": i, "layer_by_layer": tag_layer_by_layer}
            save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-layer_{i}-{tag_layer_by_layer}.jsonl")
            # patching
            layer_ids = [i for i in range(i, model.config.num_hidden_layers)]
            replace_qwen2_5_vl_delete_llm_image_pos_embed(layer_ids_to_delete=layer_ids)
            # generate
            generate_and_record(model, processor, dataset_name, settings, save_path, final_path)

def test_delete_image_pos_embed(
    model_name="llava1_5_7b",
    dataset_name="GQA",
    device="cuda:1",
    batch_size=8,
    data_num=1000, 
    random=False,
    layer_by_layer=False,  # the pos embed in LLaVA's LLM is RoPE
    tag="0"
    ):
    
    exp_save_dir = root_dir / f"eval/GQA/results/test_delete_image_pos_embed_{tag}"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device)
    
    
    def generate_and_record(model, processor, dataset_name, settings, save_path, final_path):
        # generate
        _ = generate_batch_responses(
            model, 
            processor, 
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_new_tokens=10, 
            save_path=save_path,
            data_num=data_num,
            random=random,
        )
        # calculate task performance
        acc = []
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
                elif dataset_name in ["GQA", "gqa_no_spatial"]:
                    if "llava" in model_name:
                        pred = pred.lower()
                    accs.append(pred == sample["answer"])
            acc = sum(accs) / len(accs)
        
        with jsonlines.open(final_path, "a") as f:
            f.write({
                "settings": settings,
                "acc": acc
            })
        return acc
    
    tag_layer_by_layer = "layer_by_layer" if layer_by_layer else "and_all"
    final_path = os.path.join(save_dir, f"del_image_pos_embed-{model_name}.jsonl")
    
    layer_num = model.config.text_config.num_hidden_layers
    if layer_by_layer:
        for i in trange(layer_num):
            settings = {"layer_id": i, "layer_by_layer": tag_layer_by_layer}
            save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-layer_{i}-{tag_layer_by_layer}.jsonl")
            # patching
            layer_ids = [i,]
            if "llava1_5" in model_name or "llava1_6" in model_name:
                replace_llava1_5_vl_delete_llm_image_pos_embed(layer_ids)
            elif "intern" in model_name:
                pass
            # generate
            generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
    else:
        for i in trange(layer_num):
            settings = { "layer_id": i, "layer_by_layer": tag_layer_by_layer}
            save_path = os.path.join(save_dir, f"del_pos_embed-{model_name}-layer_{i}-{tag_layer_by_layer}.jsonl")
            # patching
            layer_ids = [i for i in range(i, layer_num)]
            if "llava1_5" in model_name or "llava1_6" in model_name:
                replace_llava1_5_vl_delete_llm_image_pos_embed(layer_ids)
            elif "intern" in model_name:
                pass
            # generate
            generate_and_record(model, processor, dataset_name, settings, save_path, final_path)

    plot_delete_image_pos_embed(model_name, layer_by_layer="layer_and_after", normal_acc=0.600)
  
def test_shuffle_image_tokens(
    model_name="qwen2_5_vl",
    dataset_name="GQA",
    device="cuda:1",
    batch_size=8,
    delete_vision_token=False,  # whether or not to delete <|vision_start|> and <|vision_end|>
    data_num=1000,
    random=False,
    tag="0"
):
    if delete_vision_token:
        exp_save_dir = root_dir / "eval/GQA/results/test_shuffle_image_tokens_delete_vision_token"
    else:
        exp_save_dir = root_dir / "eval/GQA/results/test_shuffle_image_tokens"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device)
    
    if "llava1_5" in model_name:
        replace_llava1_5_vl_shuffle_image_token_orders(delete_vision_token=delete_vision_token)
    elif "qwen2_5" in model_name:
        replace_qwen2_5_vl_shuffle_image_token_orders(delete_vision_token=delete_vision_token)
    elif "intern" in model_name:
        pass
        replace_intern2_5_vl_shuffle_image_token_orders(model)
    else:
        pass
    
    # generate
    save_path = os.path.join(save_dir, f"shuffle_image_tokens-{model_name}_{tag}.jsonl")
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
    acc = []
    with jsonlines.open(save_path, "r") as f:
        accs = []
        for sample in f:
            pred = sample["pred"]
            if "llava" in model_name:
                pred = pred.lower()
            accs.append(pred == sample["answer"])
    acc = sum(accs) / len(accs)
    print(f"acc: {acc}")
    return acc

def test_shuffle_image_pos_ids(
        model_name="qwen2_5_vl",
        dataset_name="gqa",
        device="cuda:1",
        tag="0",
        data_num=1000,
        random=False,
    ):
    
    exp_save_dir = root_dir / "eval/GQA/results/test_shuffle_image_pos_ids"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device)
    
    if "llava1_5" in model_name:
        replace_llava1_5_vl_shuffle_llm_image_tokens_pos_ids()
    elif "qwen2_5" in model_name:
        replace_qwen2_5_vl_shuffle_llm_image_tokens_pos_ids()
    else:
        pass
    
    # generate
    save_path = os.path.join(save_dir, f"shuffle_image_pos_ids-{model_name}_{tag}.jsonl")
    _ = generate_batch_responses(
        model, 
        processor, 
        dataset_name=dataset_name, 
        max_new_tokens=10, 
        save_path=save_path,
        data_num=data_num,
        random=random
    )
    # calculate task performance
    acc = []
    with jsonlines.open(save_path, "r") as f:
        accs = []
        for sample in f:
            pred = sample["pred"]
            if "llava" in model_name:
                pred = pred.lower()
            accs.append(pred == sample["answer"])
    acc = sum(accs) / len(accs)
    print(f"acc: {acc}")
    return acc

def test_add_pos_embed(
    model_name="llava1_5_7b",
    dataset_name="GQA",
    device="cuda:1",
    region="vit",
    batch_size=8,
    data_num=1000, 
    random=False,
    ):
    
    exp_save_dir = root_dir / "eval/GQA/results/test_add_pos_embed"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    model, processor, tokenizer = load_model(model_name, device)
    
    
    def generate_and_record(model, processor, dataset_name, settings, save_path, final_path):
        # generate
        _ = generate_batch_responses(
            model, 
            processor, 
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_new_tokens=10, 
            save_path=save_path,
            data_num=data_num,
            random=random,
        )
        # calculate task performance
        acc = []
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
                elif dataset_name in ["GQA", "gqa_no_spatial"]:
                    if "llava" in model_name:
                        pred = pred.lower()
                    accs.append(pred == sample["answer"])
            acc = sum(accs) / len(accs)
            
        with jsonlines.open(final_path, "a") as f:
            f.write({
                "settings": settings,
                "acc": acc
            })
        return acc
    
    final_path = os.path.join(save_dir, f"add_pos_embed-{model_name}.jsonl")
    
    settings = {"region": region}
    save_path = os.path.join(save_dir, f"add_pos_embed-{model_name}-in_{region}.jsonl")
    # patching
    if "llava1_5" in model_name or "llava1_6" in model_name:
        replace_llava1_5_vl_add_vit_pos_embed()
    elif "intern" in model_name:
        pass
    # generate
    acc = generate_and_record(model, processor, dataset_name, settings, save_path, final_path)
    
    print(f"acc: {acc}")
    
    return acc

def plot_delete_pos_embed(
        final_path=root_dir / "eval/GQA/results/test_delete_pos_embed/qwen2_5_vl/del_pos_embed-qwen2_5_vl.jsonl",
        region="vit", 
        layer_by_layer="and_all"
    ):
    
    # plot: delete only in ViT, layer and after
    layer_ids = []
    accs = []
    with jsonlines.open(final_path, "r") as f:
        for data in f:
            if data["settings"]["region"] == region and data["settings"]["layer_by_layer"] == layer_by_layer:
                layer_ids.append(data["settings"]["layer_id"])
                accs.append(data["acc"])
    layer_ids = [layer_id + 1 for layer_id in layer_ids]
    
    figure_dir = root_dir / "figures"
    tag = "layer_and_after" if layer_by_layer == "and_all" else "layer_by_layer"
    model_name = ""
    if "qwen" in final_path:
        model_name = "qwen2_5_vl"
    elif "llava" in final_path:
        model_name = "llava1_6_mistral_7b"
    elif "intern" in final_path:
        model_name = "internvl2_5_8b"
    else:
        pass
    save_path = os.path.join(figure_dir, f"delete_pos_embed-{model_name}-in_{region}-{tag}.pdf")

    plt.figure()
    y_line = 0.612
    plt.axhline(y=y_line, color='black', linestyle='--', alpha=0.55, label='Original')
    # ax1.text(0, y_line, f' {y_line}', color=color, verticalalignment='bottom', alpha=0.7)

    # plt.plot(layer_ids,
    #          accs,
    #          color="blue",
    #          label='Accuracy',
    #          alpha=0.55, 
    #          marker='o',
    #          markersize=10, 
    #         #  markerfacecolor=(0, 0, 1, 0.7),
    #          markeredgecolor='blue',
    #          markeredgewidth=1.5)
    
    plt.plot(
        layer_ids,
        accs,
        color="blue",
        label='Delete positional embedding',
        alpha=0.55, 
        marker='o',
        markersize=5, 
        #  markerfacecolor=(0, 0, 1, 0.7),
        markeredgecolor='blue',
        markeredgewidth=1.5
    )
    
    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 0.7))
    plt.ylabel("Accuracy", color=mcolors.to_rgba('black', 0.7))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))
    
    plt.savefig(save_path, bbox_inches='tight')

def plot_delete_pos_embed_with_std_error(
    final_dir=root_dir / "eval/GQA/results/test_delete_pos_embed/qwen2_5_vl",
    region="vit", 
    layer_by_layer="and_all"
):
    
    # plot: delete only in ViT, layer and after
    layer_ids = list(range(32)) if region == "vit" else list(range(28))
    all_accs = []
    # del_pos_embed-qwen2_5_vl-in_vit-layer_5-and_all.jsonl
    for layer_id in layer_ids:
        accs = []
        file_path = os.path.join(final_dir, f"del_pos_embed-qwen2_5_vl-in_{region}-layer_{layer_id}-{layer_by_layer}.jsonl")
        with jsonlines.open(file_path, "r") as f:
            accs = []
            for line in f:
                if line["answer"] == line["pred"]:
                    accs.append(1)
                else:
                    accs.append(0)
        all_accs.append(accs)
    layer_ids = [layer_id + 1 for layer_id in layer_ids]
        
    figure_dir = root_dir / "figures"
    tag = "layer_and_after" if layer_by_layer == "and_all" else "layer_by_layer"
    
    save_path = os.path.join(figure_dir, f"delete_pos_embed-qwen2_5_vl-in_{region}-{tag}-new.pdf")

    plt.figure()
    y_line = 0.612
    plt.axhline(y=y_line, color='black', linestyle='--', alpha=0.55, label='Original')
    
    mean_acc = np.mean(all_accs, axis=1)
    std_err = np.std(all_accs, axis=1) / np.sqrt(len(all_accs[0]))
    plt.plot(
        layer_ids,
        mean_acc,
        color="blue",
        label='Delete positional embedding',
        alpha=0.55, 
        marker='o',
        markersize=5, 
        #  markerfacecolor=(0, 0, 1, 0.7),
        markeredgecolor='blue',
        markeredgewidth=1.5
    )
    plt.fill_between(layer_ids, mean_acc - std_err, mean_acc + std_err, color='blue', alpha=0.2)
    
    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 0.7))
    plt.ylabel("Accuracy", color=mcolors.to_rgba('black', 0.7))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))
    
    plt.savefig(save_path, bbox_inches='tight')
 
def plot_delete_image_pos_embed(model_name, layer_by_layer="and_all", normal_acc=0.612):
    final_dir = root_dir / f"eval/GQA/results/test_delete_image_pos_embed_0/{model_name}"
    
    # plot: delete only in ViT, layer and after
    layer_ids = list(range(32)) if "llava" in model_name else list(range(28))
    
    all_accs = []
    for layer_id in layer_ids:
        accs = []
        file_path = os.path.join(final_dir, f"del_pos_embed-{model_name}-layer_{layer_id}-{layer_by_layer}.jsonl")
        with jsonlines.open(file_path, "r") as f:
            accs = []
            for line in f:
                pred = line["pred"]
                if "llava" in model_name:
                    pred = pred.lower()
                if line["answer"] == pred:
                    accs.append(1)
                else:
                    accs.append(0)
        all_accs.append(accs)        
            
    layer_ids = [layer_id + 1 for layer_id in layer_ids]
            
    figure_dir = root_dir / "figures"
    tag = "layer_and_after" if layer_by_layer == "and_all" else "layer_by_layer"
    save_path = os.path.join(figure_dir, f"delete_image_pos_embed-{model_name}-{tag}_new.pdf")

    plt.figure()
    y_line = normal_acc
    plt.axhline(y=y_line, color='black', linestyle='--', alpha=0.55, label='Original')
    
    mean_acc = np.mean(all_accs, axis=1)
    std_err = np.std(all_accs, axis=1) / np.sqrt(len(all_accs[0]))
    
    plt.plot(
        layer_ids,
        mean_acc,
        color="blue",
        label='Delete positional embedding in image tokens',
        alpha=0.55, 
        marker='o',
        markersize=5, 
        #  markerfacecolor=(0, 0, 1, 0.7),
        markeredgecolor='blue',
        markeredgewidth=1.5
    )
    plt.fill_between(layer_ids, mean_acc - std_err, mean_acc + std_err, color='blue', alpha=0.2)
    
    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 0.7))
    plt.ylabel("Accuracy", color=mcolors.to_rgba('black', 0.7))
    
    # set y limit
    plt.ylim(0, 0.7)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    ax.spines["left"].set_color((0.3, 0.3, 0.3))
    ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 0.7))
    
    plt.savefig(save_path, bbox_inches='tight')
    

if __name__ == "__main__":
    
    # anno_path = data_dir / "VQA_v2/v2_mscoco_val2014_annotations.json"
    # qn_path = data_dir / "VQA_v2/v2_OpenEnded_mscoco_val2014_questions.json"
    # save_path = root_dir / "eval/vqa/results/test_normal/llava1_5_7b/normal-llava1_5_7b_0.jsonl"
    # eval_vqa = evaluate_vqa(anno_path, qn_path, save_path)
    # print(f"eval_vqa: {eval_vqa}")
    # pass

    ckpt_path = str(work_dir / "checkpoints_rope_scaling/normal/lora-qwen2_vl_7b-data_gqa_spatial_60000-bsz_32-lr_3e-6-rank_16-alpha_32/checkpoint-1875")
    # ckpt_path = str(work_dir / "checkpoints_rope_scaling/rope_scaling/lora-qwen2_vl_7b-data_gqa_spatial_60000-bsz_32-lr_3e-6-rank_16-alpha_32-scaling_type_poly-poly_alpha_49-poly_p_8/checkpoint-1875")
    model_name = "qwen2_vl_7b"  # qwen2_5_vl, llava1_5_7b, internvl2_5_8b
    accs = []
    for i in range(1):
        acc = test_normal(
            model_name=model_name,  # qwen2_5_vl qwen2_5_vl_3b qwen2_5_vl_3b qwen2_vl_2b, llava1_5_7b, internvl2_5_8b
            model_path=None,  # None or ckpt path
            dataset_name="mmb",  # "GQA", "whatsup_b", "gqa_no_spatial", "whatsup_b_left_right", "whatsup_b_behind_in_front_of"
            device="cuda:7",
            tag=str(i),
            data_num=1000,
            batch_size=8,
            random=True
        )  # 0.612
        # coco: data_num=500, random=True
        accs.append(acc)
    try:
        avg_acc = sum(accs) / len(accs)
    except:
        avg_acc = 0.0
    print(f"accs ({model_name}): avg_acc: {avg_acc}, {accs}")
    
    
    # ------------------------------------- delete_pos_embed
    # accs = []
    # layerwise=False
    # model_name = "qwen2_5_vl"
    # if layerwise:
    #     test_delete_pos_embed_qwen(
    #         model_name=model_name,  # qwen2_5_vl, llava1_5_7b, llava1_6_mistral_7b, internvl2_5_8b
    #         dataset_name="GQA",
    #         device="cuda:2",
    #         region="vit",
    #         layer_by_layer=False,
    #         layerwise=layerwise,
    #     )
    # else:
    #     for i in range(5):
    #         acc = test_delete_pos_embed_qwen(
    #             model_name=model_name,  # qwen2_vl_2b, qwen2_5_vl, llava1_5_7b, llava1_6_mistral_7b, internvl2_5_8b
    #             dataset_name="gqa_spatial",  # "GQA", "whatsup_b"
    #             device="cuda:1",
    #             region="vit",
    #             layer_by_layer=False,
    #             layerwise=layerwise,
    #         )
    #         accs.append(acc)
    #     print(f"accs ({model_name}): {accs}")
    
    # accs = []
    # model_name = "internvl2_5_8b"  # llava1_5_7b, internvl2_5_8b
    # for i in range(5):
    #     acc = test_delete_pos_embed(
    #         model_name=model_name,  # llava1_5_7b, internvl2_5_8b
    #         dataset_name="GQA",
    #         device="cuda:3",
    #         region="vit",
    #         batch_size=2,
    #         random=True
    #     )
    #     accs.append(acc)
    # print(f"accs: {accs}")
    

    # ------------------------------------- add_pos_embed
    # accs = []
    # for i in range(5):
    #     acc = test_add_pos_embed(
    #         model_name="llava1_5_7b",  # llava1_5_7b, internvl2_5_8b
    #         dataset_name="whatsup_b",
    #         device="cuda:2",
    #         region="vit",
    #         batch_size=8,
    #         random=True
    #     )
    #     accs.append(acc)
    # print(f"accs: {accs}")

    
    # ------------------------------------- delete_image_pos_embed
    # test_delete_image_pos_embed(
    #     model_name="qwen2_5_vl",
    #     dataset_name="gqa",
    #     device="cuda:1",
    #     layer_by_layer=False,
    # )
    
    # test_delete_image_pos_embed(
    #     model_name="llava1_5_7b",  # llava1_5_7b, internvl2_5_8b
    #     dataset_name="GQA",
    #     device="cuda:2",
    #     batch_size=8,
    #     random=True,
    #     layer_by_layer=False,
    #     tag="1"
    # )

    # ------------------------------------- shuffle_image_tokens
    # accs = []
    # for i in range(5):
    #     acc = test_shuffle_image_tokens(
    #         model_name="internvl2_5_8b",  # "qwen2_5_vl", llava1_5_7b, internvl2_5_8b
    #         dataset_name="GQA",
    #         device="cuda:1",
    #         batch_size=2,
    #         delete_vision_token=False,
    #         tag=str(i),
    #     )
    #     accs.append(acc)
    # print(f"accs: {accs}")
    
    
    # ------------------------------------- shuffle_image_pos_ids
    # accs = []
    # for i in range(5):
    #     acc = test_shuffle_image_pos_ids(
    #     model_name="llava1_5_7b",  # "qwen2_5_vl", llava1_5_7b, internvl2_5_8b
    #     dataset_name="GQA",
    #     device="cuda:2",
    #     tag=str(i),
    #     random=True,
    # )
    #     accs.append(acc)
    # print(f"accs: {accs}")
    
    
    # plot_delete_pos_embed(region="vit", layer_by_layer="and_all")
    # plot_delete_pos_embed_with_std_error(region="vit", layer_by_layer="and_all")
    # plot_delete_image_pos_embed(layer_by_layer="and_all")    
    
    # path_name = root_dir / "eval/GQA/results/test_shuffle_image_tokens_delete_vision_token/qwen2_5_vl/shuffle_image_tokens-qwen2_5_vl_"
    # for i in range(5):
    #     path = path_name
    #     path += str(i) + ".jsonl"
    #     with jsonlines.open(path, "r") as f:
    #         accs = []
    #         for sample in f:
    #             accs.append(sample["pred"] == sample["answer"])
    #         acc = sum(accs) / len(accs)
    #         print(f"acc: {acc}")

    # plot_delete_image_pos_embed("llava1_5_7b", layer_by_layer="and_all", normal_acc=0.606)
    
    # plot_delete_image_pos_embed("qwen2_5_vl", layer_by_layer="and_all", normal_acc=0.612)