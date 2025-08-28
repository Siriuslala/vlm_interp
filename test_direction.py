"""
Test directions. 
"""
import torch
from torch.utils.data import DataLoader
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
from qwen_vl_utils import process_vision_info as process_vision_info_qwen
from utils import load_image_intern, load_image_intern_return_wh
import clip
from utils.intern_vl import get_conv_template

from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

import sys
sys.path.append(root_dir)

from eval.data_utils import *
from patch.monkey_patch import *

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
import glob
import json
import jsonlines
import os
import copy
from collections import defaultdict
import random


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
            "num_patches_list": num_patches_list,  # include the thumbnail
            "block_wh_list": block_wh_list
            # "question_length": question_length,
        }
    else:
        raise ValueError("Model name not supported.")
    
    return inputs

def get_intern_inputs_embeds(model, tokenizer, model_inputs, vit_embeds=None):
    
    # in batch_chat()
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    
    num_patches_list = model_inputs["num_patches_list"]
    questions = model_inputs["questions"]
    pixel_values = model_inputs["pixel_values"]
    queries = []
    for idx, num_patches in enumerate(num_patches_list):
        question = questions[idx]
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
            
        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
        queries.append(query)
    
    model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    
    # in generate()
    if vit_embeds is None:
        vit_embeds = model.extract_feature(pixel_values)  # (n_tiles, n_patches, hidden_size)
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == model.img_context_token_id)
    assert selected.sum() != 0
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)
    
    return input_embeds, attention_mask

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
        with torch.no_grad():
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
        
        for out, samples in zip(outputs, samples):
            # print(f"image: {samples['image_path']}, choices: {samples['caption_options']}, pred: {out}")
            print(f"pred: [{out}]")
        
        preds.extend(outputs)  
        
    with jsonlines.open(save_path, "w") as f:
        for pred, sample in zip(preds, all_samples):
            sample.update({"pred": pred})
            f.write(sample)
               
    return preds

def draw_bbox(
    image_path,
    bboxes=None,
    save_dir="",
    save_format="png",
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
    save_path = os.path.join(save_dir, f"bboxes_{image_name}_objects.{save_format}")
    image.save(save_path)
    print(f"save_path: {save_path}")
    
def test_vit_direction_clip(device, model_name, dataset_name):
    
    model, preprocess, tokenizer = load_model(model_name, device)
    print(type(model))
   
    dataloader = load_data(dataset_name, model, processor=None, data_num=None)
    
    sims_feature = []
    sims_clip = []
    for idx, batch in enumerate(tqdm(dataloader)):
        
        _, samples = batch
        batchsize = len(samples)
        image_list = [preprocess(Image.open(sample["image_path"])).unsqueeze(0).to(device) for sample in samples]
        images = torch.cat(image_list)  # 形状: [batch_size, 3, 224, 224]
        texts = [clip.tokenize(sample["caption_options"]).to(device) for sample in samples]
            
        with torch.inference_mode():
            # Get image embeddings
            image_features = model.encode_image(images)  # 形状: [batch_size, embed_dim]
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Get text representations
            text_features_list = []
            for text_group in texts:
                text_features = model.encode_text(text_group)  # 形状: [4, embed_dim]
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features_list.append(text_features)

            # Compute the cosine similarities between the image and text embeddings
            # ABCD, the first option is correct
            for i in range(batchsize):
                similarity = (image_features[i] @ text_features_list[i].T).softmax(dim=-1).tolist()
                sims_feature.append(similarity)
            
            for image, text_group in zip(image_list, texts):
                logits_per_image, logits_per_text = model(image, text_group)
                probs = logits_per_image.softmax(dim=-1).tolist()[0]
                sims_clip.append(probs)
            # print(sims_clip)
            # import pdb; pdb.set_trace()
    
    sims_feature_correct = np.mean([sim[0] for sim in sims_feature])
    sims_feature_false_average = np.mean([np.mean(sim[1:]) for sim in sims_feature])
    sims_clip_correct = np.mean([sim[0] for sim in sims_clip])
    sims_clip_false_average = np.mean([np.mean(sim[1:]) for sim in sims_clip])
    print("sims_feature_correct", sims_feature_correct)
    print("sims_feature_false_average", sims_feature_false_average)
    print("sims_clip_correct", sims_clip_correct)
    print("sims_clip_false_average", sims_clip_false_average)
    
    accs_feature = []
    accs_clip = []
    for i in range(len(sims_feature)):
        if sims_feature[i][0] == max(sims_feature[i]):
            accs_feature.append(1)
        else:
            accs_feature.append(0)
    for i in range(len(sims_clip)):
        if sims_clip[i][0] == max(sims_clip[i]):
            accs_clip.append(1)
        else:
            accs_clip.append(0)
    accs_feature = np.mean(accs_feature)
    accs_clip = np.mean(accs_clip)
    print("accs_feature", accs_feature)
    print("accs_clip", accs_clip)
    
    # whatsup_a, ViT-L/14@336px
    # sims_feature_correct 0.25017323213465076
    # sims_feature_false_average 0.24994185703252655
    # sims_clip_correct 0.2610593181030423
    # sims_clip_false_average 0.24631380567363664
    # accs_feature 0.27450980392156865
    # accs_clip 0.27205882352941174
    
    # whatsup_b, ViT-L/14@336px
    # sims_feature_correct 0.2504553933745449
    # sims_feature_false_average 0.24984444923771237
    # sims_clip_correct 0.2799668659284277
    # sims_clip_false_average 0.24001014965637604
    # accs_feature 0.32281553398058255
    # accs_clip 0.28640776699029125
    
def test_vit_direction_vlm_old(device, model_name, dataset_name):
    """ 
    Trying to compute the sim between ViT representation and text representation.
    """
    
    model, processor, tokenizer = load_model(model_name, device)
    vit, llm = model.visual, model.model
    
    replace_qwen2_5_vl_test_directions_processor_return_indices()
    dataloader = load_data(dataset_name, model, processor)
    
    for idx, batch in enumerate(tqdm(dataloader)):
        
        model_inputs, samples = batch
        batchsize = len(samples)
        inputs, image_indices = model_inputs
        inputs.to(model.device)
        # print(image_indices)
            
        with torch.inference_mode():
            # Get image embeddings
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            pixel_values = pixel_values.type(vit.dtype)
            # print(pixel_values.shape)  #  before 14*14 patching, after resize
            image_embeds = vit(pixel_values, grid_thw=image_grid_thw)
            # print(image_embeds.shape)  # after 2*2 downsampling (merger)
            image_embeds = [image_embeds[indice] for indice in image_indices]
            # for image_embed in image_embeds:
            #     print(image_embed.shape)  # (image_len, hidden_size)

            # Get text representations
            option_embeds = []
            all_options = []
            for i in range(batchsize):
                all_options.extend(samples[i]["caption_options"])
            options_input_ids = tokenizer(all_options, return_tensors="pt", padding=True).input_ids.to(model.device)
            options_attention_mask = tokenizer(all_options, return_tensors="pt", padding=True).attention_mask.to(model.device)
            # for ids in options_input_ids:
            #     tokens = tokenizer.convert_ids_to_tokens(ids)
            #     print(tokens)
            # import pdb; pdb.set_trace()
            # option_embeds = llm.embed_tokens(options_input_ids.to(model.device))  # text embeddings
            option_embeds = llm(options_input_ids, options_attention_mask, return_dict=True).last_hidden_state  # text hidden states
            option_embeds = option_embeds.reshape(batchsize, 4, -1, model.config.hidden_size)
            options_attention_mask = options_attention_mask.reshape(batchsize, 4, -1)
            # print(option_embeds.shape)  # (bsz, num_options, seq_len, hidden_size)
            
            # Compute the cosine similarities between the image and text embeddings
            # ABCD, the first option is correct
            correct_option_sim = []
            average_false_option_sim = []
            for i in range(batchsize):
                image_embed_sample = image_embeds[i]  # (image_len, hidden_size)
                option_embed_sample = option_embeds[i]  # (num_options, seq_len, hidden_size)
                option_attn_mask_sample = options_attention_mask[i]  # (num_options, seq_len)
                
                # compute the similarity
                # image_embed_sample = image_embed_sample[-1].unsqueeze(0)  # (1, hidden_size)
                # option_embed_sample = option_embed_sample[:, -1, :]  # (num_options, hidden_size)
                # sims = torch.cosine_similarity(image_embed_sample, option_embed_sample, dim=1)
                
                # # 计算注意力分数 [4, m, n] 
                attn_scores = torch.einsum("md,bnd->bmn", image_embed_sample, option_embed_sample)
                # Mask padding [4, m, n]
                mask = option_attn_mask_sample.unsqueeze(1).expand(-1, image_embed_sample.size(0), -1)  # [4, m, n]
                attn_scores = attn_scores.masked_fill(mask == 0, -float('inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)

                # 加权文本表示 [4, m, hidden_dim]
                attended_text = torch.einsum("bmn,bnd->bmd", attn_weights, option_embed_sample)

                # 计算相似度 [4]
                sims = F.cosine_similarity(
                    image_embed_sample.unsqueeze(0).expand(4, -1, -1),  # [4, m, hidden_dim]
                    attended_text,                               # [4, m, hidden_dim]
                    dim=-1
                ).mean(dim=1)  # [4]
                
                print("sims", sims)
                correct_option_sim.append(sims[0].item())
                average_false_option_sim.append(torch.mean(sims[1:]).item())
                print("correct_option_sim", correct_option_sim)
                print("average_false_option_sim", average_false_option_sim)
                # import pdb; pdb.set_trace()
            print("correct_option_sim", correct_option_sim)
            print("average_false_option_sim", average_false_option_sim)
            import pdb; pdb.set_trace()

def test_vit_direction_vlm(device, model_name, dataset_name, tag='', batch_size=8, shuffle_image_tokens=False):
    """
    Directly test the performance of the VLM using ViT representations of different layers. 
    """
    
    def generate_and_record(model, processor, dataset_name, batch_size, save_path, final_path, settings):
        # generate
        _ = generate_batch_responses(
            model, 
            processor, 
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_new_tokens=5, 
            save_path=save_path,
        )
        # calculate task performance
        accs = []
        with jsonlines.open(save_path, "r") as f:
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
                    accs.append(sample["pred"] == sample["answer"])
            acc = sum(accs) / len(accs)
            
        with jsonlines.open(final_path, "a") as f:
            f.write({
                "settings": settings,
                "acc": acc
            })

        return acc
    
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    
    if any(name in model_name for name in ["qwen"]):
        vit_layer_num = model.config.vision_config.depth
        monkey_patch_func = replace_qwen2_5_vl_test_directions_vit_discard_layers
        if shuffle_image_tokens:
            replace_qwen2_5_vl_shuffle_image_token_orders(delete_vision_token=False)
    elif any(name in model_name for name in ["llava"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
        monkey_patch_func = replace_llava1_5_test_directions_vit_discard_layers
    elif any(name in model_name for name in ["intern"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
        monkey_patch_func = replace_intern2_5_vl_test_directions_vit_discard_layers
    
    # set save path
    theme = "discard_vit_layers_and_shuffle_image_tokens" if shuffle_image_tokens else "test_vit_directions_vlm"
    if any(name in dataset_name for name in ["whatsup", "coco", "gqa"]):
        exp_save_dir = root_dir / f"eval/WhatsUp/results/{theme}"
    elif dataset_name == "vsr":
        exp_save_dir = root_dir / f"eval/VSR/results/{theme}"
    elif any(name in dataset_name for name in ["GQA"]):
        exp_save_dir = root_dir / f"eval/GQA/results/{theme}"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    sub_save_dir = os.path.join(save_dir, "res")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sub_save_dir, exist_ok=True)
    
    acc_layers = []
    for layer_id in range(vit_layer_num):
        # discard layers
        layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
        if "intern" in model_name:
            monkey_patch_func(model, layer_ids_to_delete)
        else:
            monkey_patch_func(layer_ids_to_delete)
        
        # test
        acc = generate_and_record(
            model,
            processor,
            dataset_name=dataset_name,
            batch_size=batch_size,
            save_path=os.path.join(sub_save_dir, f"layer_{layer_id}_{dataset_name}_{tag}.jsonl"),
            final_path=os.path.join(save_dir, f"final_{dataset_name}_{tag}.jsonl"),
            settings={"layer_id": layer_id, "model_name": model_name, "dataset_name": dataset_name}
        )
        acc_layers.append(acc)
    
    print("acc_layers", acc_layers)

def test_vit_direction_left_vs_on(device, model_name, dataset_name, tag="0", batch_size=8):
    
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    
    if any(name in model_name for name in ["qwen"]):
        vit_layer_num = model.config.vision_config.depth
        monkey_patch_func = replace_qwen2_5_vl_test_directions_vit_discard_layers
    elif any(name in model_name for name in ["llava"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
        monkey_patch_func = replace_llava1_5_test_directions_vit_discard_layers
    elif any(name in model_name for name in ["intern"]):
        vit_layer_num = model.config.vision_config.num_hidden_layers
        monkey_patch_func = replace_intern2_5_vl_test_directions_vit_discard_layers
        
    # set save path
    exp_save_dir = root_dir / "eval/WhatsUp/results/test_vit_directions_left_vs_on"
    save_dir = os.path.join(exp_save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    # get token ids for: on, On, under, Under, left, Left, right, Right
    token_ids = {}
    if any(word in dataset_name for word in ["in_front_of", "behind"]):
        if "llava" in model_name:
            token_ids = {"in": ["in", "In", "▁In"], "behind": ["behind", "Behind", "▁Be"], "left": ["left", "Left", "▁Left"], "right": ["right", "Right", "▁Right"]}
        elif "qwen" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["Ġbehind", "Behind"], "left": ["left", "Left"], "right": ["right", "Right"]}
        elif "intern" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["beh", "Beh"], "left": ["left", "Left"], "right": ["right", "Right"]}
        else:
            pass
    else:
        token_ids = {"on": ["on", "On", "▁On"], "under": ["under", "Under", "▁Under"], "left": ["left", "Left", "▁Left"], "right": ["right", "Right", "▁Right"]}
    
    for key, value in token_ids.items():
        token_ids[key] = [tokenizer.convert_tokens_to_ids(token) for token in value]
        
    # print("token_ids", token_ids)
    # # sentences = ["A is in front of B", "A is behind B", "A is on the left of B", "A is on the right of B"]
    # sentences = ["In front of B", "Behind B", "Left of B", "Right of B."]
    # for sentence in sentences:
    #     tokens = tokenizer.tokenize(sentence)
    #     token_ids = tokenizer.convert_tokens_to_ids(tokens)
    #     print(tokens, token_ids)
    # import pdb; pdb.set_trace()
        
    if any(word in dataset_name for word in ["in_front_of", "behind"]):
        direction_ids = {
            "in": token_ids["in"],
            "behind": token_ids["behind"],
            "left": token_ids["left"],
            "right": token_ids["right"]
        }
    else:
        direction_ids = {
            "on": token_ids["on"],
            "under": token_ids["under"],
            "left": token_ids["left"],
            "right": token_ids["right"]
        }
    
    # test
    if any(word in dataset_name for word in ["in_front_of", "behind"]):
        probs_all_layers = {"in": [], "behind": [], "left": [], "right": []}
    else:
        probs_all_layers = {"on": [], "under": [], "left": [], "right": []}
    ori_model_name = model_name 
    for layer_id in range(vit_layer_num):
        # discard layers
        layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
        if "intern" in model_name:
            monkey_patch_func(model, layer_ids_to_delete)
        else:
            monkey_patch_func(layer_ids_to_delete)
        
        # forward    
        if isinstance(model, Qwen2_5_VLForConditionalGeneration):
            model_name = "qwen2_5_vl"
        elif isinstance(model, LlavaForConditionalGeneration) or isinstance(model, LlavaNextForConditionalGeneration):
            model_name = "llava"
        elif "Intern" in str(type(model)):
            model_name = "intern"
        dataloader = load_data(dataset_name, model_name, processor, data_num=None, random=False, collator=Controlled_Images_no_options_collate_function, batch_size=batch_size)
        
        if any(word in dataset_name for word in ["in_front_of", "behind"]):
            probs_layer = {"in": [], "behind": [], "left": [], "right": []}
        else:
            probs_layer = {"on": [], "under": [], "left": [], "right": []}
        all_samples = []
        for idx, batch in enumerate(tqdm(dataloader)):
            inputs, samples = batch
            all_samples.extend(samples)
            
            if "intern" in model_name:
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(model.device)
            else:
                inputs.to(model.device)
            
            outputs = []
            with torch.no_grad():
                if "intern" in model_name:
                    input_embeds, attention_mask = get_intern_inputs_embeds(
                        model, tokenizer, inputs, vit_embeds=None
                    )
                    outputs = model.language_model(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        return_dict=True, 
                        output_hidden_states=True
                    )
                else:
                    outputs = model(
                        **inputs, 
                        return_dict=True, 
                        output_hidden_states=True
                    )
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)  # (bsz, vocab_size)
                for token, token_ids in direction_ids.items():
                    probs_layer[token].extend(torch.sum(probs[:, token_ids], dim=-1).tolist())  # (bsz)
                # our_tokens = [tokenizer.tokenize(token) for token in direction_ids.keys()]
                pred = torch.argmax(probs, dim=-1)  # (bsz)
                pred_token = tokenizer.convert_ids_to_tokens(pred)
                print("pred_token", pred_token, "ans", [sample["answer"] for sample in samples])
            # break
            
        for token in probs_all_layers.keys():
            probs_all_layers[token].append(np.mean(probs_layer[token]))  # mean over dataset
        final_path = os.path.join(save_dir, f"final_{dataset_name}_{tag}.jsonl")
        with jsonlines.open(final_path, "a") as f:
            data = {
                "settings": {
                    "layer_id": layer_id, 
                    "model_name": model_name, 
                    "dataset_name": dataset_name
                }, 
                "probs": probs_layer
            }
            f.write(data)
        print(f"layer {layer_id} finished")
        
    plot_vit_direction_left_vs_on_logits(ori_model_name)

def get_relation_representations(model_name, dataset_name, device, before_connector=False, add_repr=False, tsne=True, objects=["bowl", "candle"], color="viridis", delete_pos_embed=False, draw=True, use_thumbnail=False):
    """
    A rather long function that performs the following tasks:
    1. For each sampe, get the representations of the two objects in four images corresponding to the four relations.
    2. (Optional) Cluster the representations of the direction vectors.
    """
    # set save path
    save_dir = root_dir / "eval/WhatsUp/results"
    if delete_pos_embed:
        exp_dir = os.path.join(save_dir, "test_vit_directions_relation_delete_pos_embed", f"{model_name}")
    else:
        if use_thumbnail:
            exp_dir = os.path.join(save_dir, "test_vit_directions_relation", f"{model_name}_thumbnail")
        else:
            exp_dir = os.path.join(save_dir, "test_vit_directions_relation", f"{model_name}")
    if add_repr:
        exp_dir += "_add_repr"
    os.makedirs(exp_dir, exist_ok=True)
    
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    model_dir = MODEL_NAME_TO_PATH[model_name]
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    
    vit, llm = None, None
    if "llava" in model_name:
        vit, llm = model.vision_tower, model.language_model
        vision_process_func = None
        if delete_pos_embed:
            replace_llava1_5_vl_delete_vit_pos_embed()
    elif "qwen" in model_name:
        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
        if "qwen2_5" in model_name:
            replace_qwen2_5_vl_test_directions_processor_return_indices()
            if delete_pos_embed:
                layer_ids_to_delete = [i for i in range(0, model.config.vision_config.depth)]
                replace_qwen2_5_vl_delete_vit_pos_embed(layer_ids_to_delete)
        else:
            replace_qwen2_vl_test_directions_processor_return_indices()
            if delete_pos_embed:
                layer_ids_to_delete = [i for i in range(0, model.config.vision_config.depth)]
                replace_qwen2_vl_delete_vit_pos_embed(layer_ids_to_delete)
    elif "intern" in model_name:
        vision_process_func = load_image_intern_return_wh
        if delete_pos_embed:
            replace_intern2_5_delete_vit_pos_embed(model)
    else:
        pass
    
    # load batch of images
    whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
    whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
    dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
    image_files = os.listdir(dir_name)
    image_names = [file_name.split('.')[0] for file_name in image_files]
    image_pairs = []
    for i, img_name in enumerate(image_names):
        obj_satellite = img_name.split('_')[0]
        obj_nucleus = img_name.split('_')[-1]
        image_pairs.append((obj_satellite, obj_nucleus))
    image_pairs = list(set(image_pairs))
    
    bboxes = {}
    bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
    with jsonlines.open(bboxes_path, "r") as f:
        for sample in f:
            bboxes[sample["image"]] = sample["bbox"]

    # forward
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    silhouette_score_list = []
    davies_bouldin_score_list = []
    calinski_harabasz_score_list = []
    adjusted_rand_score_list = []
    orthogonality_dist_list = []
    cos_matrix_list = []
        
    for pair_id, image_pair in enumerate(image_pairs):
        obj_satellite, obj_nucleus = image_pair
        if objects and not ((obj_satellite == objects[0]) and (obj_nucleus == objects[1])):
            continue
                
        # get four images for a pair
        sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
        sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
        sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
        
        # get inputs
        model_inputs = get_model_inputs(
            model_name=model_name,
            processor=processor,
            vision_process_func=vision_process_func,
            image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
            prompts=["describe the image and tell me what is the main object in the image" for _ in range(4)]
        )
        if "llava" in model_name:
            inputs = model_inputs
        if "qwen" in model_name:
            inputs, image_indices = model_inputs
        elif "intern" in model_name:
            inputs = model_inputs
            image_indices = []
            st = 0
            n_patches_per_tile = int(model_config.vision_config.image_size // (model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)) ) ** 2
            for idx in range(len(inputs["num_patches_list"])):  # for each image
                if use_thumbnail:  # thumbnail tile is at the end of the tile list
                    num_tiles = inputs["num_patches_list"][idx]
                    image_indices.append(slice(st + (num_tiles - 1) * n_patches_per_tile, st + num_tiles * n_patches_per_tile))
                    st += num_tiles * n_patches_per_tile
                else:
                    num_tiles = inputs["num_patches_list"][idx]  # include the thumbnail
                    image_indices.append(slice(st, st + num_tiles * n_patches_per_tile))
                    st += num_tiles * n_patches_per_tile
            # print("n_patches_list", inputs["num_patches_list"])
            # print("image_indices", image_indices)

        # get resized bboxes
        sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
        resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
        for idx, image_path in enumerate(sample_image_paths):
            # draw bboxes
            # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
            
            # 1. get original sizes
            image = Image.open(image_path).convert('RGB')
            original_width, original_height = image.size
            image.close()
            # 2. get current size (after resize)
            if "llava" in model_name:
                height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
            elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                image_grid_thw = inputs["image_grid_thw"]  # after resize & patching, before merging
                _, height, width = image_grid_thw[idx].cpu().numpy()
                height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                width *= model.config.vision_config.spatial_patch_size
            elif "intern" in model_name:
                if use_thumbnail:
                    height, width = 448, 448
                else:
                    block_image_width, block_image_height = inputs["block_wh_list"][idx]
                    height = block_image_height * model_config.vision_config.image_size
                    width = block_image_width * model_config.vision_config.image_size
            # 3. get resized bboxes
            scale_width = width / original_width
            scale_height = height / original_height
            print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale_width}, {scale_height}")
            # import pdb; pdb.set_trace()
            x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
            resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
            x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
            resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
            
        # get object patches
        object_patch_ids = [{"satellite": None, "nucleus": None} for _ in range(4)]
        if "llava" in model_name:
            patch_size = model_config.vision_config.patch_size
        elif "qwen" in model_name:
            patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
        elif "intern" in model_name:
            patch_size = model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)
        for idx, image_path in enumerate(sample_image_paths):
            for obj_name, obj_bbox in resized_bboxes[idx].items():
                obj_patch_ids = []
                x1, y1, x2, y2 = obj_bbox
                for i in range(int(width // patch_size)):
                    for j in range(int(height // patch_size)):
                        x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                        if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                        # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                            obj_patch_ids.append(j * int(width // patch_size) + i)
                object_patch_ids[idx][obj_name] = obj_patch_ids

        # check bbox patches
        if False:
            for idx, image_name in enumerate(sample_image_names):
                check_img_save_path = os.path.join(pair_check_dir, f"patches_{image_name}.pdf")
                
                # get patched image size
                if "llava" in model_name:
                    patch_size = model_config.vision_config.patch_size
                    resized_height, resized_width = inputs["pixel_values"][idx].shape[1:]  #  after resize, before patching
                    grid_height, grid_width = height // patch_size, width // patch_size
                elif "qwen" in model_name:
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, grid_height, grid_width = image_grid_thw[idx].cpu().numpy()
                    resized_height = grid_height * model.config.vision_config.spatial_patch_size
                    resized_width = grid_width * model.config.vision_config.spatial_patch_size
                    grid_height /= model.config.vision_config.spatial_merge_size
                    grid_width /= model.config.vision_config.spatial_merge_size
                    grid_height, grid_width = int(grid_height), int(grid_width)
                elif "intern" in model_name:
                    patch_size = model_config.vision_config.patch_size
                    block_image_width, block_image_height = inputs["block_wh_list"][idx]
                    resized_height = block_image_height * model_config.vision_config.image_size
                    resized_width = block_image_width * model_config.vision_config.image_size
                    grid_height = block_image_height * model_config.vision_config.image_size // patch_size
                    grid_width = block_image_width * model_config.vision_config.image_size // patch_size
                    grid_height /= model_config.downsample_ratio
                    grid_width /= model_config.downsample_ratio
                    grid_height, grid_width = int(grid_height), int(grid_width)
                else:
                    pass
                
                # get colors in the patched image
                token_colors = ["black" for _ in range(grid_height * grid_width)]
                for color_id, color in enumerate(token_colors):
                    if color_id in object_patch_ids[idx]["satellite"]:
                        token_colors[color_id] = "blue"
                    elif color_id in object_patch_ids[idx]["nucleus"]:
                        token_colors[color_id] = "yellow"
                    
                # get colors in the original image
                if "llava" in model_name:
                    back_scale = model.config.vision_config.patch_size
                elif "qwen" in model_name:
                    back_scale = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
                elif "intern" in model_name:
                    back_scale = int(model_config.vision_config.patch_size * model_config.downsample_ratio)
                
                new_token_colors = np.array([[None for _ in range(resized_width)] for _ in range(resized_height)])
                for j in range(len(token_colors)):
                    x = j % grid_width
                    y = j // grid_width
                    x, y = int(x), int(y)
                    x_range = slice(x * back_scale, (x + 1) * back_scale)
                    y_range = slice(y * back_scale, (y + 1) * back_scale)
                    for row in range(y_range.start, y_range.stop):
                        for col in range(x_range.start, x_range.stop):
                            new_token_colors[row][col] = token_colors[j]
                token_colors = new_token_colors.flatten()
                
                # create a blank image
                seg_map = Image.new("RGB", (resized_width, resized_height), (0, 0, 0))

                # fill color
                for j in range(len(token_colors)):
                    x = j % resized_width
                    y = j // resized_width
                    x, y = int(x), int(y)
                    rgb_color = mcolors.to_rgb(token_colors[j])
                    int_color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
                    seg_map.putpixel((x, y), int_color)
                    
                # 5. save the segmentation map
                seg_map.save(check_img_save_path)
            continue
        # import pdb; pdb.set_trace()
        
        # forward as a batch     
        # get vit output (before llm or before connector)
        image_reprs = None  # llava: (bsz, len(24*24), dim)  qwen2.5-vl: (len, dim)
        with torch.no_grad():
            if "intern" in model_name:
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(device)
            else:
                inputs.to(device)
            if before_connector:
                if "llava" in model_name:
                    image_reprs = vit(
                        pixel_values=inputs["pixel_values"],
                        output_hidden_states=True,
                    ).last_hidden_state  # (batch_size, seq_len, dim)
                elif "qwen" in model_name:
                    image_reprs = None
                else:
                    pass
            else:
                if "llava" in model_name:
                    image_reprs = model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        vision_feature_layer=model.config.vision_feature_layer,
                        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    )
                elif "qwen" in model_name:
                    inputs["pixel_values"] = inputs["pixel_values"].type(vit.dtype)
                    image_reprs = vit(
                        inputs["pixel_values"], 
                        grid_thw=inputs["image_grid_thw"], 
                    )
                    # original_image size: [1280, 960], image_size: [1148, 840]
                    # print("pixel_values", inputs["pixel_values"].shape)  # [19680, 1176]
                    # print("image_reprs", image_reprs.shape)  # [4920, 3584]  4920 = 4 * (1148 / 28) * (840 / 28)
                    # import pdb; pdb.set_trace()
                elif "intern" in model_name:
                    image_reprs = model.extract_feature(
                        pixel_values=inputs["pixel_values"],
                    ) # (n_tiles, num_patches (24*24), dim)
                    image_reprs = image_reprs.reshape(-1, image_reprs.shape[-1])  # (n_tiles * num_patches, dim)
                else:
                    pass
        
        # get the relation representation
        # sample_relation_repr = {}
        object_reprs = {"satellite": [], "nucleus": []}
        for idx, image_path in enumerate(sample_image_paths):
            # 1. get the pooled representation for each object
            if "llava" in model_name:
                img_repr = image_reprs[idx]
            elif "qwen" in model_name:
                img_repr = image_reprs[image_indices[idx]]
            elif "intern" in model_name:
                img_repr = image_reprs[image_indices[idx]]
                # print("img_repr", img_repr.shape)  # (num_patches, dim)
                # import pdb; pdb.set_trace()
            else:
                pass
            satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].float().cpu().tolist()  # .mean(dim=0).unsqueeze(0).cpu().numpy()
            nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].float().cpu().tolist()
            object_reprs["satellite"].append(satellite_repr)
            object_reprs["nucleus"].append(nucleus_repr)
        
        # save
        directions_left = []
        directions_right = []
        directions_behind = []
        directions_in_front_of = []
        relation_reprs = [directions_left, directions_right, directions_behind, directions_in_front_of]
        relation_reprs_pooled = []
        for i in range(4):
            # get object reprs
            satellite_repr = torch.tensor(object_reprs["satellite"][i]).to(device)  # shape: (len, dim) .mean(dim=0).unsqueeze(0)
            nucleus_repr = torch.tensor(object_reprs["nucleus"][i]).to(device)  # shape: (len, dim) .mean(dim=0).unsqueeze(0)
            # randomly sample patches from satellite and nucleus
            s_len = satellite_repr.shape[0]
            n_len = nucleus_repr.shape[0]
            print("s_len", s_len, "n_len", n_len)
            s_scale, n_scale = 2, 2
            if s_len <= 20:
                s_scale = 1
            if n_len <= 20:
                n_scale = 1
            for _ in range(100):
                satellite_patch_ids = random.sample(range(s_len), s_len // s_scale)
                nucleus_patch_ids = random.sample(range(n_len), n_len // n_scale)
                satellite_repr_ = satellite_repr[satellite_patch_ids, :].mean(dim=0).unsqueeze(0)
                nucleus_repr_ = nucleus_repr[nucleus_patch_ids, :].mean(dim=0).unsqueeze(0)
                if add_repr:
                    relation_reprs[i].append(satellite_repr_ + nucleus_repr_)
                else:
                    relation_reprs[i].append(satellite_repr_ - nucleus_repr_)
            pooled_satellite_repr = satellite_repr.mean(dim=0).unsqueeze(0)
            pooled_nucleus_repr = nucleus_repr.mean(dim=0).unsqueeze(0)
            if add_repr:
                relation_reprs_pooled.append(pooled_satellite_repr + pooled_nucleus_repr)
            else:
                relation_reprs_pooled.append(pooled_satellite_repr - pooled_nucleus_repr)
                
        # prepare data for dimensionality reduction & clustering
        # First, check the orthogonality of the relation vectors
        relation_reprs_pooled = torch.cat(relation_reprs_pooled, dim=0).cpu().numpy()  # (4, 3584)
        if tsne:
            tsne = TSNE(n_components=2, random_state=41, perplexity=3, n_iter=1000)
            reduced_relation_reprs = tsne.fit_transform(relation_reprs_pooled)
        else:
            n_components_95 = 3
            pca = PCA(n_components=n_components_95)
            reduced_relation_reprs = pca.fit_transform(relation_reprs_pooled)
            
        norm = np.linalg.norm(reduced_relation_reprs, axis=1, keepdims=True)
        normalized_relation_reprs = reduced_relation_reprs / norm
        cos_matrix = np.dot(normalized_relation_reprs, normalized_relation_reprs.T)
        cos_matrix_list.append(cos_matrix)
        orthogonality_dist = []
        for i in range(4):
            for j in range(0, i):
                cos_sim = cos_matrix[i, j].item()
                if (i, j) in [(1, 0), (3, 2)]:
                    orthogonality_dist.append(cos_sim - (-1))
                elif (i, j) in [(2, 0), (2, 1), (3, 0), (3, 1)]:
                    orthogonality_dist.append(cos_sim - 0)
                else:
                    pass
        orthogonality_dist = np.sum(orthogonality_dist)
        orthogonality_dist_list.append(orthogonality_dist)
        
        # Next, check the relation representations
        relation_reprs = [torch.cat(relation_reprs[i], dim=0) for i in range(4)]
        print("relation_reprs", relation_reprs[0].shape)  # (100, 3584)
        all_reprs = torch.cat(relation_reprs, dim=0)
        print("all_reprs", all_reprs.shape)  # (400, 3584)
        # set label (left: 0, right: 1, behind: 2, in front of: 3)
        all_reprs = all_reprs.cpu().numpy()
        scaler = StandardScaler()
        all_reprs = scaler.fit_transform(all_reprs)
        minmax_scaler = MinMaxScaler()
        all_reprs = minmax_scaler.fit_transform(all_reprs)
        true_labels = []
        for i in range(4):
            true_labels.extend([i] * len(directions_left))
        
        if True:
            # perform t-SNE for data of all four directions
            if tsne:
                print("t-SNE start")
                tsne = TSNE(n_components=2, random_state=41, perplexity=30, n_iter=1000)
                tsne_results = tsne.fit_transform(all_reprs)
                print("t-SNE finished")
            else:
                # perform PCA for data of all four directions
                print("PCA start")
                pca = PCA(n_components=2)
                pca_results = pca.fit_transform(all_reprs)
                print("PCA finished")
            
            # perform k-means clustering
            print("k-means start")
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=14, n_init='auto')
            results_for_clustering = tsne_results if tsne else pca_results
            kmeans.fit(results_for_clustering)
            labels = kmeans.labels_
            silhouette = silhouette_score(all_reprs, labels)
            db_index = davies_bouldin_score(all_reprs, labels)
            ch_index = calinski_harabasz_score(all_reprs, labels)
            ari = adjusted_rand_score(true_labels, labels)
            silhouette_score_list.append(silhouette)
            davies_bouldin_score_list.append(db_index)
            if ch_index < 10000:
                calinski_harabasz_score_list.append(ch_index)
            adjusted_rand_score_list.append(ari)
            print(f"Silhouette Coefficient: {silhouette}")
            print(f"Davies-Bouldin Index: {db_index}")
            print(f"Calinski-Harabasz Index: {ch_index}")
            print(f"Adjusted Rand Index: {ari}")
            
            cluster_labels = kmeans.labels_
            print(f"k-means for sample {pair_id} finished \n{'----' * 20}")
            
            # visualize the clustering results
            if draw:
                exp_sample_dir = os.path.join(exp_dir, f"cluster", f"{obj_satellite}_{obj_nucleus}")
                os.makedirs(exp_sample_dir, exist_ok=True)
                
                if tsne:
                    plt.figure()
                    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_labels, cmap=color, s=20, alpha=1)
                    plt.xlabel('t-SNE Dimension 1')
                    plt.ylabel('t-SNE Dimension 2')
                    cbar = plt.colorbar(scatter, ticks=range(4))
                    cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                    save_path = os.path.join(exp_sample_dir, f"relation_repr_tsne_normalized.pdf")
                    print("save_path", save_path)
                    plt.savefig(save_path, bbox_inches='tight')
                else:
                    plt.figure()
                    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=true_labels, cmap=color, s=20, alpha=1)
                    plt.xlabel('PCA Dimension 1')
                    plt.ylabel('PCA Dimension 2')
                    cbar = plt.colorbar(scatter, ticks=range(4))
                    cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                    save_path = os.path.join(exp_sample_dir, f"relation_repr_pca_normalized.pdf")
                    print("save_path", save_path)
                    plt.savefig(save_path, bbox_inches='tight')
                print("save_path", save_path)
                plt.savefig(save_path, bbox_inches='tight')
                
                # visualize the k-means clustering results
                plt.figure()
                scatter = plt.scatter(results_for_clustering[:, 0], results_for_clustering[:, 1], c=cluster_labels, cmap=color, s=20, alpha=1)
                dim_name = "t-SNE" if tsne else "PCA"
                plt.xlabel(f'{dim_name} Dimension 1')
                plt.ylabel(f'{dim_name} Dimension 2')
                cbar = plt.colorbar(scatter, ticks=range(n_clusters))
                cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                save_path = os.path.join(exp_sample_dir, f"relation_repr_kmeans.pdf")
                save_path = ''.join(save_path.split(".")[:-1]) + "_normalized.pdf"
                save_path = ''.join(save_path.split(".")[:-1]) + f"_after_{dim_name}.pdf"
                print("save_path", save_path)
                plt.savefig(save_path, bbox_inches='tight')
    
    avg_silhouette_score = np.mean(silhouette_score_list)
    avg_davies_bouldin_score = np.mean(davies_bouldin_score_list)
    avg_calinski_harabasz_score = np.mean(calinski_harabasz_score_list)
    avg_adjusted_rand_score = np.mean(adjusted_rand_score_list)
    avg_orthogonality_dist = np.mean(orthogonality_dist_list)
    visual_geometry_score_list = [1 / (1 + score) for score in orthogonality_dist_list]
    visual_geometry_score = np.mean(visual_geometry_score_list)
    cos_matrix = np.mean(cos_matrix_list, axis=0)
    print("----" * 20)
    print(f"model_name: {model_name}, dataset_name: {dataset_name}, before_connector: {before_connector}, delete_pos_embed: {delete_pos_embed}")
    print(f"avg_silhouette_score: {avg_silhouette_score}")
    print(f"avg_davies_bouldin_score: {avg_davies_bouldin_score}")
    print(f"avg_calinski_harabasz_score: {avg_calinski_harabasz_score}")
    print(f"avg_adjusted_rand_score: {avg_adjusted_rand_score}")
    print(f"avg_orthogonality_dist: {avg_orthogonality_dist}")
    print(f"visual_geometry_score: {visual_geometry_score}")
    print(f"cos_matrix: {cos_matrix}")
    
    cluster_res_path = os.path.join(exp_dir, f"cluster_res.jsonl")
    silhouette_score_list_converted = [float(score) for score in silhouette_score_list]
    davies_bouldin_score_list_converted = [float(score) for score in davies_bouldin_score_list]
    calinski_harabasz_score_list_converted = [float(score) for score in calinski_harabasz_score_list]
    adjusted_rand_score_list_converted = [float(score) for score in adjusted_rand_score_list]
    cos_matrix_list_converted = [cos_matrix.tolist() for cos_matrix in cos_matrix_list]
    with jsonlines.open(cluster_res_path, "w") as f:
        f.write({
            "silhouette_score": silhouette_score_list_converted,
            "davies_bouldin_score": davies_bouldin_score_list_converted,
            "calinski_harabasz_score": calinski_harabasz_score_list_converted,
            "adjusted_rand_score": adjusted_rand_score_list_converted,
            "orthogonality_dist": orthogonality_dist_list,
            "visual_geometry_score": visual_geometry_score_list,
            "cos_matrix": cos_matrix_list_converted,
        })            

def get_relation_representations_layerwise(model_name, dataset_name, device, before_connector=False, objects=["bowl", "candle"], tsne=True, color="viridis", delete_pos_embed=False, use_thumbnail=False):
    """
    A rather rather long function that performs the following tasks:
    For each ViT layer:
        1. For each sampe, get the representations of the two objects in four images corresponding to the four relations.
        2. (Optional) Cluster the representations of the direction vectors.
    """
    # set save path
    save_dir = root_dir / "eval/WhatsUp/results"
    if delete_pos_embed:
        exp_dir = os.path.join(save_dir, "test_vit_directions_relation_layerwise_delete_pos_embed", f"{model_name}")
    else:
        if use_thumbnail:
            exp_dir = os.path.join(save_dir, "test_vit_directions_relation_layerwise", f"{model_name}_thumbnail")
        else:
            exp_dir = os.path.join(save_dir, "test_vit_directions_relation_layerwise", f"{model_name}")
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        model_dir = MODEL_NAME_TO_PATH[model_name]
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
            vision_process_func = None
            vit_layer_num = model.config.vision_config.num_hidden_layers
            monkey_patch_func = replace_llava1_5_test_directions_vit_discard_layers
        elif "qwen" in model_name:
            vit, llm = model.visual, model.model
            vision_process_func = process_vision_info_qwen
            vit_layer_num = model.config.vision_config.depth
            if "qwen2_5" in model_name:
                replace_qwen2_5_vl_test_directions_processor_return_indices()
                if delete_pos_embed:
                    monkey_patch_func = replace_qwen2_5_vl_test_directions_vit_discard_layers_and_delete_pos_embed
                else:
                    monkey_patch_func = replace_qwen2_5_vl_test_directions_vit_discard_layers
            else:
                replace_qwen2_vl_test_directions_processor_return_indices()
                monkey_patch_func = replace_qwen2_vl_test_directions_vit_discard_layers
        elif "intern" in model_name:
            vision_process_func = load_image_intern_return_wh
            vit_layer_num = model.config.vision_config.num_hidden_layers
            monkey_patch_func = replace_intern2_5_vl_test_directions_vit_discard_layers
        else:
            pass
    
    # load batch of images
    if True:
        whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
        whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
        dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
        image_files = os.listdir(dir_name)
        image_names = [file_name.split('.')[0] for file_name in image_files]
        image_pairs = []
        for i, img_name in enumerate(image_names):
            obj_satellite = img_name.split('_')[0]
            obj_nucleus = img_name.split('_')[-1]
            image_pairs.append((obj_satellite, obj_nucleus))
        image_pairs = list(set(image_pairs))
        
        bboxes = {}
        bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
        with jsonlines.open(bboxes_path, "r") as f:
            for sample in f:
                bboxes[sample["image"]] = sample["bbox"]

    # forward
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    silhouette_score_list = []
    davies_bouldin_score_list = []
    calinski_harabasz_score_list = []
    adjusted_rand_score_list = []
    orthogonality_dist_list = []
    visual_geometry_score_list = []
            
    for layer_id in range(vit_layer_num):
        
        layer_save_dir = os.path.join(exp_dir, f"layer_{layer_id}")
        os.makedirs(layer_save_dir, exist_ok=True)
        
        # discard layers
        layer_ids_to_delete = [i for i in range(layer_id+1, vit_layer_num)]
        if "intern" in model_name:
            monkey_patch_func(model, layer_ids_to_delete)
        else:
            monkey_patch_func(layer_ids_to_delete)
        
        layer_silhouette_score_list = []
        layer_davies_bouldin_score_list = []
        layer_calinski_harabasz_score_list = []
        layer_adjusted_rand_score_list = []
        layer_orthogonality_dist_list = []
            
        for pair_id, image_pair in enumerate(image_pairs):
            
            if objects is None and pair_id > 20:
                break
            
            obj_satellite, obj_nucleus = image_pair
            if objects and not ((obj_satellite == objects[0]) and (obj_nucleus == objects[1])):
                continue
            
            # get four images for a pair
            sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
            sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
            sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
            
            # get inputs
            model_inputs = get_model_inputs(
                model_name=model_name,
                processor=processor,
                vision_process_func=vision_process_func,
                image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
                prompts=["describe the image and tell me what is the main object in the image" for _ in range(4)]
            )
            if "llava" in model_name:
                inputs = model_inputs
            if "qwen" in model_name:
                inputs, image_indices = model_inputs
            elif "intern" in model_name:
                inputs = model_inputs
                image_indices = []
                st = 0
                n_patches_per_tile = int(model_config.vision_config.image_size // (model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)) ) ** 2
                for idx in range(len(inputs["num_patches_list"])):
                    if use_thumbnail:  # thumbnail tile is at the end of the tile list
                        num_tiles = inputs["num_patches_list"][idx]
                        image_indices.append(slice(st + (num_tiles - 1) * n_patches_per_tile, st + num_tiles * n_patches_per_tile))
                        st += num_tiles * n_patches_per_tile
                    else:
                        num_tiles = inputs["num_patches_list"][idx]  # include the thumbnail
                        image_indices.append(slice(st, st + num_tiles * n_patches_per_tile))
                        st += num_tiles * n_patches_per_tile
                # print("n_patches_list", inputs["num_patches_list"])
                # print("image_indices", image_indices)

            # get object info (resized bboxes and patch ids)
            if True:
                # get resized bboxes
                sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
                resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
                for idx, image_path in enumerate(sample_image_paths):
                    # draw bboxes
                    # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                    
                    # 1. get original sizes
                    image = Image.open(image_path).convert('RGB')
                    original_width, original_height = image.size
                    image.close()
                    # 2. get current size (after resize)
                    if "llava" in model_name:
                        height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                    elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                        image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                        _, height, width = image_grid_thw[idx].cpu().numpy()
                        height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                        width *= model.config.vision_config.spatial_patch_size
                    elif "intern" in model_name:
                        if use_thumbnail:
                            height, width = 448, 448
                        else:
                            block_image_width, block_image_height = inputs["block_wh_list"][idx]
                            height = block_image_height * model_config.vision_config.image_size
                            width = block_image_width * model_config.vision_config.image_size
                    # 3. get resized bboxes
                    scale_width = width / original_width
                    scale_height = height / original_height
                    print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale_width}, {scale_height}")
                    # import pdb; pdb.set_trace()
                    x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                    x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                    resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                    x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                    x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                    resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                    
                # get object patches
                object_patch_ids = [{"satellite": None, "nucleus": None} for _ in range(4)]
                if "llava" in model_name:
                    patch_size = model_config.vision_config.patch_size
                elif "qwen" in model_name:
                    patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
                elif "intern" in model_name:
                    patch_size = model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)
                for idx, image_path in enumerate(sample_image_paths):
                    for obj_name, obj_bbox in resized_bboxes[idx].items():
                        obj_patch_ids = []
                        x1, y1, x2, y2 = obj_bbox
                        for i in range(int(width // patch_size)):
                            for j in range(int(height // patch_size)):
                                x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                                if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                                # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                    obj_patch_ids.append(j * int(width // patch_size) + i)
                        object_patch_ids[idx][obj_name] = obj_patch_ids
            
            # forward as a batch     
            # get vit output (before llm or before connector)
            image_reprs = None  # llava: (bsz, len(24*24), dim)  qwen2.5-vl: (len, dim)
            with torch.no_grad():
                if "intern" in model_name:
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(device)
                else:
                    inputs.to(device)
                if before_connector:
                    if "llava" in model_name:
                        image_reprs = vit(
                            pixel_values=inputs["pixel_values"],
                            output_hidden_states=True,
                        ).last_hidden_state  # (batch_size, seq_len, dim)
                    elif "qwen" in model_name:
                        image_reprs = None
                    else:
                        pass
                else:
                    if "llava" in model_name:
                        image_reprs = model.get_image_features(
                            pixel_values=inputs["pixel_values"],
                            vision_feature_layer=model.config.vision_feature_layer,
                            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                        )
                    elif "qwen" in model_name:
                        inputs["pixel_values"] = inputs["pixel_values"].type(vit.dtype)
                        image_reprs = vit(
                            inputs["pixel_values"], 
                            grid_thw=inputs["image_grid_thw"], 
                        )
                        # original_image size: [1280, 960], image_size: [1148, 840]
                        # print("pixel_values", inputs["pixel_values"].shape)  # [19680, 1176]
                        # print("image_reprs", image_reprs.shape)  # [4920, 3584]  4920 = 4 * (1148 / 28) * (840 / 28)
                        # import pdb; pdb.set_trace()
                    elif "intern" in model_name:
                        image_reprs = model.extract_feature(
                            pixel_values=inputs["pixel_values"],
                        )
                        image_reprs = image_reprs.reshape(-1, image_reprs.shape[-1])  # (n_tiles * num_patches, len(24*24), dim)
                    else:
                        pass
            
            # get the relation representation
            # sample_relation_repr = {}
            object_reprs = {"satellite": [], "nucleus": []}
            for idx, image_path in enumerate(sample_image_paths):
                if "llava" in model_name:
                    img_repr = image_reprs[idx]
                elif "qwen" in model_name:
                    img_repr = image_reprs[image_indices[idx]]
                elif "intern" in model_name:
                    img_repr = image_reprs[image_indices[idx]]
                    # print("img_repr", img_repr.shape)  # (num_patches, dim)
                    # import pdb; pdb.set_trace()
                else:
                    pass
                satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].float().cpu().tolist()  # .mean(dim=0).unsqueeze(0).cpu().numpy()
                nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].float().cpu().tolist()
                object_reprs["satellite"].append(satellite_repr)
                object_reprs["nucleus"].append(nucleus_repr)
            
            # cluster
            directions_left = []
            directions_right = []
            directions_behind = []
            directions_in_front_of = []
            relation_reprs = [directions_left, directions_right, directions_behind, directions_in_front_of]
            relation_reprs_pooled = []
            for i in range(4):
                # get object reprs
                satellite_repr = torch.tensor(object_reprs["satellite"][i]).to(device)  # shape: (len, dim) .mean(dim=0).unsqueeze(0)
                nucleus_repr = torch.tensor(object_reprs["nucleus"][i]).to(device)  # shape: (len, dim) .mean(dim=0).unsqueeze(0)
                # randomly sample patches from satellite and nucleus
                s_len = satellite_repr.shape[0]
                n_len = nucleus_repr.shape[0]
                print("s_len", s_len, "n_len", n_len)
                s_scale, n_scale = 2, 2
                if s_len <= 20:
                    s_scale = 1
                if n_len <= 20:
                    n_scale = 1
                for _ in range(100):
                    satellite_patch_ids = random.sample(range(s_len), s_len // s_scale)
                    nucleus_patch_ids = random.sample(range(n_len), n_len // n_scale)
                    satellite_repr_ = satellite_repr[satellite_patch_ids, :].mean(dim=0).unsqueeze(0)
                    nucleus_repr_ = nucleus_repr[nucleus_patch_ids, :].mean(dim=0).unsqueeze(0)
                    relation_reprs[i].append(satellite_repr_ - nucleus_repr_)
                pooled_satellite_repr = satellite_repr.mean(dim=0).unsqueeze(0)
                pooled_nucleus_repr = nucleus_repr.mean(dim=0).unsqueeze(0)
                relation_reprs_pooled.append(pooled_satellite_repr - pooled_nucleus_repr)
                        
            # prepare data for dimensionality reduction & clustering
            # First, check orthogonality
            if True:
                relation_reprs_pooled = torch.cat(relation_reprs_pooled, dim=0).cpu().numpy()  # (4, 3584)
                if tsne:
                    tsne = TSNE(n_components=2, random_state=41, perplexity=3, n_iter=1000)
                    reduced_relation_reprs = tsne.fit_transform(relation_reprs_pooled)
                else:
                    n_components_95 = 3
                    pca = PCA(n_components=n_components_95)
                    reduced_relation_reprs = pca.fit_transform(relation_reprs_pooled)
                    
                norm = np.linalg.norm(reduced_relation_reprs, axis=1, keepdims=True)
                normalized_relation_reprs = reduced_relation_reprs / norm
                cos_matrix = np.dot(normalized_relation_reprs, normalized_relation_reprs.T)
                
                # visualize
                plt.figure()
                colors = [mcolors.to_rgba("yellowgreen", 1), (1, 1, 1), mcolors.to_rgba("darkgray", 1)]  # (yellowgreen, gold)
                n_bins = 100
                cmap_name = 'my_list'
                cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
                plt.imshow(cos_matrix, interpolation='nearest', cmap=cm, vmin=-1, vmax=1)
                plt.colorbar()
                plt.xticks(range(4), ["left", "right", "behind", "in front of"])
                plt.yticks(range(4), ["left", "right", "behind", "in front of"])
                plt.xlabel("Direction vectors")
                plt.ylabel("Directions vectors")
                cluster_name = "t-SNE" if tsne else "PCA"
                exp_sample_dir = os.path.join(layer_save_dir, f"cluster_{cluster_name}", f"{obj_satellite}_{obj_nucleus}")
                os.makedirs(exp_sample_dir, exist_ok=True)
                save_path = os.path.join(exp_sample_dir, f"orthogonality.pdf")
                print("save_path", save_path)
                plt.savefig(save_path, bbox_inches='tight')   
                
                orthogonality_dist = []
                for i in range(4):
                    for j in range(0, i):
                        cos_sim = cos_matrix[i, j].item()
                        if (i, j) in [(1, 0), (3, 2)]:
                            orthogonality_dist.append(cos_sim - (-1))
                        elif (i, j) in [(2, 0), (2, 1), (3, 0), (3, 1)]:
                            orthogonality_dist.append(cos_sim - 0)
                        else:
                            pass
                orthogonality_dist = np.sum(orthogonality_dist)
                layer_orthogonality_dist_list.append(orthogonality_dist)
            
            # Next, check the relation representations
            relation_reprs = [torch.cat(relation_reprs[i], dim=0) for i in range(4)]
            print("relation_reprs", relation_reprs[0].shape)  # (100, 3584)
            all_reprs = torch.cat(relation_reprs, dim=0)
            print("all_reprs", all_reprs.shape)  # (400, 3584)
            
            # set label (left: 0, right: 1, behind: 2, in front of: 3)
            all_reprs = all_reprs.cpu().numpy()
            scaler = StandardScaler()
            all_reprs = scaler.fit_transform(all_reprs)
            minmax_scaler = MinMaxScaler()
            all_reprs = minmax_scaler.fit_transform(all_reprs)
            true_labels = []
            for i in range(4):
                true_labels.extend([i] * len(directions_left))
            
            if True:
                # perform t-SNE for data of all four directions
                if tsne:
                    print("t-SNE start")
                    tsne = TSNE(n_components=2, random_state=41, perplexity=30, n_iter=1000)
                    tsne_results = tsne.fit_transform(all_reprs)
                    print("t-SNE finished")
                else:
                    # perform PCA for data of all four directions
                    print("PCA start")
                    pca = PCA(n_components=2)
                    pca_results = pca.fit_transform(all_reprs)
                    print("PCA finished")
                
                # perform k-means clustering
                print("k-means start")
                n_clusters = 4
                kmeans = KMeans(n_clusters=n_clusters, random_state=14, n_init='auto')
                results_for_clustering = tsne_results if tsne else pca_results
                kmeans.fit(results_for_clustering)
                labels = kmeans.labels_
                silhouette = silhouette_score(all_reprs, labels)
                db_index = davies_bouldin_score(all_reprs, labels)
                ch_index = calinski_harabasz_score(all_reprs, labels)
                ari = adjusted_rand_score(true_labels, labels)
                layer_silhouette_score_list.append(silhouette)
                layer_davies_bouldin_score_list.append(db_index)
                layer_calinski_harabasz_score_list.append(ch_index)
                layer_adjusted_rand_score_list.append(ari)
                
                cluster_labels = kmeans.labels_
                print(f"k-means for sample {pair_id} finished \n{'----' * 20}")
                
                # visualize the t-SNE results
                if True:
                    cluster_name = "t-SNE" if tsne else "PCA"
                    exp_sample_dir = os.path.join(layer_save_dir, f"cluster_{cluster_name}", f"{obj_satellite}_{obj_nucleus}")
                    os.makedirs(exp_sample_dir, exist_ok=True)
                    
                    if tsne:
                        plt.figure()
                        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_labels, cmap=color, s=20, alpha=1)
                        plt.xlabel('t-SNE Dimension 1')
                        plt.ylabel('t-SNE Dimension 2')
                        cbar = plt.colorbar(scatter, ticks=range(4))
                        cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                        save_path = os.path.join(exp_sample_dir, f"relation_repr_tsne_normalized.pdf")
                        print("save_path", save_path)
                        plt.savefig(save_path, bbox_inches='tight')
                    else:
                        plt.figure()
                        scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=true_labels, cmap=color, s=20, alpha=1)
                        plt.xlabel('PCA Dimension 1')
                        plt.ylabel('PCA Dimension 2')
                        cbar = plt.colorbar(scatter, ticks=range(4))
                        cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                        save_path = os.path.join(exp_sample_dir, f"relation_repr_pca_normalized.pdf")
                        print("save_path", save_path)
                        plt.savefig(save_path, bbox_inches='tight')
                    print("save_path", save_path)
                    plt.savefig(save_path, bbox_inches='tight')
                    
                    # visualize the k-means clustering results
                    plt.figure()
                    scatter = plt.scatter(results_for_clustering[:, 0], results_for_clustering[:, 1], c=cluster_labels, cmap=color, s=20, alpha=1)
                    plt.xlabel(f'{cluster_name} Dimension 1')
                    plt.ylabel(f'{cluster_name} Dimension 2')
                    cbar = plt.colorbar(scatter, ticks=range(n_clusters))
                    cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                    save_path = os.path.join(exp_sample_dir, f"relation_repr_kmeans.pdf")
                    save_path = ''.join(save_path.split(".")[:-1]) + "_normalized.pdf"
                    save_path = ''.join(save_path.split(".")[:-1]) + f"_after_{cluster_name}.pdf"
                    print("save_path", save_path)
                    plt.savefig(save_path, bbox_inches='tight')
                        
        print(f"layer {layer_id} finished \n{'++++' * 20}")
        if True:
            silhouette_score_list.append([float(score) for score in layer_silhouette_score_list])
            davies_bouldin_score_list.append([float(score) for score in layer_davies_bouldin_score_list])
            calinski_harabasz_score_list.append([float(score) for score in layer_calinski_harabasz_score_list])
            adjusted_rand_score_list.append([float(score) for score in layer_adjusted_rand_score_list])
            orthogonality_dist_list.append(layer_orthogonality_dist_list)
            visual_geometry_score_list.append([1 / (1 + float(score)) for score in layer_orthogonality_dist_list])
            
            cluster_res_path = os.path.join(layer_save_dir, f"cluster_res_{cluster_name}.jsonl")
            layer_silhouette_score_list_converted = [float(score) for score in layer_silhouette_score_list]
            layer_davies_bouldin_score_list_converted = [float(score) for score in layer_davies_bouldin_score_list]
            layer_calinski_harabasz_score_list_converted = [float(score) for score in layer_calinski_harabasz_score_list]
            layer_adjusted_rand_score_list_converted = [float(score) for score in layer_adjusted_rand_score_list]
            with jsonlines.open(cluster_res_path, "w") as f:
                f.write({
                    "silhouette_score": layer_silhouette_score_list_converted,
                    "davies_bouldin_score": layer_davies_bouldin_score_list_converted,
                    "calinski_harabasz_score": layer_calinski_harabasz_score_list_converted,
                    "adjusted_rand_score": layer_adjusted_rand_score_list_converted,
                    "orthogonality_dist": layer_orthogonality_dist_list,
                    "visual_geometry_score": visual_geometry_score_list[-1]
                })
    
    # save cluster res of all layers
    total_cluster_res_path = os.path.join(exp_dir, f"cluster_res_{cluster_name}.jsonl")
    with jsonlines.open(total_cluster_res_path, "w") as f:
        f.write({
            "silhouette_score": silhouette_score_list,
            "davies_bouldin_score": davies_bouldin_score_list,
            "calinski_harabasz_score": calinski_harabasz_score_list,
            "adjusted_rand_score": adjusted_rand_score_list,
            "orthogonality_dist": orthogonality_dist_list,
            "visual_geometry_score": visual_geometry_score_list
        })
    print(f"silhouette_score: {silhouette_score_list}")
    print(f"davies_bouldin_score: {davies_bouldin_score_list}")
    print(f"calinski_harabasz_score: {calinski_harabasz_score_list}")
    print(f"adjusted_rand_score: {adjusted_rand_score_list}")
    print(f"orthogonality_dist: {orthogonality_dist_list}")
    print(f"visual_geometry_score: {visual_geometry_score_list}")
    
    # plot
    cluster_data = [silhouette_score_list, davies_bouldin_score_list, calinski_harabasz_score_list, adjusted_rand_score_list, visual_geometry_score_list]
    metric_label = ["Silhouette Score", "Davies Bouldin Score", "Calinski Harabasz Score", "Adjusted Rand Score", "Visual Geometry Score"]
    colors = ["blue", "blue", "blue", "blue", "blue"]
    markers = ["o", "o", "o", "o", "o"]
    layer_ids = list(range(vit_layer_num))
    layer_ids = [i + 1 for i in layer_ids]
    for i in range(len(metric_label)):
        mean = np.mean(cluster_data[i], axis=1)
        std_error = np.std(cluster_data[i], axis=1) / np.sqrt(len(cluster_data[i][1]))
        plt.figure()
        plt.plot(
            layer_ids,
            mean,
            color=colors[i],
            label=f"{metric_label[i]}",
            alpha=1, 
            marker=markers[i],
            markersize=6, 
            markerfacecolor="none",
            markeredgewidth=1.5
        )
        plt.fill_between(
            layer_ids, 
            mean - std_error, 
            mean + std_error, 
            color=colors[i], 
            alpha=0.1, 
        )
    
        plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
        plt.ylabel(f"{metric_label[i]}", color=mcolors.to_rgba('black', 1))
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        # plt.legend()
        save_path = os.path.join(exp_dir, f"cluster_res_{cluster_name}_{metric_label[i]}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(save_path)
                  
def intervene_in_spatial_reasoning(model_name, dataset_name, device, ori_dir_id=0, curr_dir_id=1, positive=False):
    """
    Intervene in the spatial reasoning of the model by changing the direction information of the object. 
    Args:
        positive: whether the intervention will enhance the model's performance or not.
    """
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    labels = ["left", "right", "behind", "in front of"]
    
    # set save path
    save_dir = root_dir / "eval/WhatsUp/results"
    exp_dir = os.path.join(save_dir, "test_vit_intervene_spatial", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, f"result_{labels[ori_dir_id]}_to_{'_'.join(labels[curr_dir_id].split(' '))}.jsonl")
    if positive:
        save_path = ''.join(save_path.split(".")[:-1]) + "_positive.jsonl"
    
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    model_dir = MODEL_NAME_TO_PATH[model_name]
    model_config = AutoConfig.from_pretrained(model_dir)
    
    vit, llm = None, None
    if "llava" in model_name:
        vit, llm = model.vision_tower, model.language_model
        vision_process_func = None
        replace_llava1_5_receive_vit_output()
    elif "qwen" in model_name:
        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
        replace_qwen2_5_vl_test_directions_processor_return_indices()
        replace_qwen2_5_vl_receive_vit_output()
    elif "intern" in model_name:
        vision_process_func = load_image_intern
    else:
        pass
    
    # load data
    whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
    whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
    dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
    image_files = os.listdir(dir_name)
    image_names = [file_name.split('.')[0] for file_name in image_files]
    image_pairs = []
    for i, img_name in enumerate(image_names):
        obj_satellite = img_name.split('_')[0]
        obj_nucleus = img_name.split('_')[-1]
        image_pairs.append((obj_satellite, obj_nucleus))
    image_pairs = list(set(image_pairs))
    
    bboxes = {}
    bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
    with jsonlines.open(bboxes_path, "r") as f:
        for sample in f:
            bboxes[sample["image"]] = sample["bbox"]

    # prepare direction probs
    if True:
        token_ids = {}
        if "llava" in model_name:
            token_ids = {"in": ["in", "In", "▁In"], "behind": ["behind", "Behind", "▁Be"], "left": ["left", "Left", "▁Left"], "right": ["right", "Right", "▁Right"]}
        elif "qwen" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["Ġbehind", "Behind"], "left": ["left", "Left"], "right": ["right", "Right"]}
        elif "intern" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["beh", "Beh"], "left": ["left", "Left"], "right": ["right", "Right"]}
        else:
            pass
        
        for key, value in token_ids.items():
            token_ids[key] = [tokenizer.convert_tokens_to_ids(token) for token in value]
            
        direction_ids = {
            "in": token_ids["in"],
            "behind": token_ids["behind"],
            "left": token_ids["left"],
            "right": token_ids["right"]
        }
        
    # forward
    # if positive:
    #     intervene_weights = np.linspace(0.0, 10.0, 11).tolist()
    # else:
    #     intervene_weights = np.linspace(0.0, 1.0, 11).tolist()
    intervene_weights = np.linspace(0.0, 1.0, 11).tolist()
    intervene_weights = [round(weight, 1) for weight in intervene_weights]
    
    for pair_id, image_pair in enumerate(image_pairs):
        obj_satellite, obj_nucleus = image_pair
        
        probs_normal = {"in": None, "behind": None, "left": None, "right": None}
        probs_intervene = {key: {"in": None, "behind": None, "left": None, "right": None} for key in intervene_weights}
    
        # preparation
        if True:
            # get four images for a pair
            sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
            sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
            sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
            
            # get inputs
            instruction = f"Is the {obj_satellite} in front of/behind/to the left of/to the right of the {obj_nucleus}? Please choose the best answer from the four options: [In front of, Behind, Left, Right], and reply with only one word. \nYour answer is:"
            model_inputs = get_model_inputs(
                model_name=model_name,
                processor=processor,
                vision_process_func=vision_process_func,
                image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
                prompts=[instruction for _ in range(4)]
            )
            if "qwen" in model_name:
                inputs, image_indices = model_inputs
            else:
                inputs = model_inputs
            inputs.to(device)

            # get resized bboxes
            sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
            resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                # draw bboxes
                # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                
                # 1. get original sizes
                image = Image.open(image_path).convert('RGB')
                original_width, original_height = image.size
                image.close()
                # 2. get current size (after resize)
                if "llava" in model_name:
                    height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, height, width = image_grid_thw[idx].cpu().numpy()
                    height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                    width *= model.config.vision_config.spatial_patch_size
                # 3. get resized bboxes
                scale = width / original_width
                # print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
                # import pdb; pdb.set_trace()
                x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                
            # get object patches
            object_patch_ids = [{"satellite": None, "nucleus": None} for _ in range(4)]
            if "llava" in model_name:
                patch_size = model_config.vision_config.patch_size
            else:
                patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                            # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids[idx][obj_name] = obj_patch_ids

        # first get the normal output
        if False:
            with torch.no_grad():
                outputs = model(
                    **inputs, 
                    return_dict=True, 
                    output_hidden_states=True
                )
                logits = outputs.logits[0, -1, :]  # (vocab_size)
                probs = F.softmax(logits, dim=-1)  # (vocab_size)
                for token, token_ids in direction_ids.items():
                    probs_normal[token] = torch.sum(probs[token_ids], dim=-1).tolist()
                pred = torch.argmax(probs, dim=-1)
                # print(f"probs shape: {probs.shape}, pred: {pred}")
                pred_token = tokenizer.convert_ids_to_tokens([pred])
                print(f"Sample {pair_id}: pred_token (normal): {pred_token}, ans: left")
        
        # then intervene with the opposite direction
        with torch.no_grad():
            for alpha in intervene_weights:
                # forward ViT
                image_reprs = None  # llava: (bsz, len(24*24), dim)  qwen2.5-vl: (len, dim)
                if "llava" in model_name:
                    image_reprs = model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        vision_feature_layer=model.config.vision_feature_layer,
                        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    )
                elif "qwen" in model_name:
                    inputs["pixel_values"] = inputs["pixel_values"].type(vit.dtype)
                    image_reprs = vit(
                        inputs["pixel_values"], 
                        grid_thw=inputs["image_grid_thw"], 
                    )
                else:
                    pass
                
                # get the relation representation
                object_reprs = {"satellite": [], "nucleus": []}
                for idx, image_path in enumerate(sample_image_paths):
                    if "llava" in model_name:
                        img_repr = image_reprs[idx]
                    elif "qwen" in model_name:
                        img_repr = image_reprs[image_indices[idx]]
                    else:
                        pass
                    satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].mean(dim=0).unsqueeze(0)  # (1, dim)
                    nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].mean(dim=0).unsqueeze(0)
                    object_reprs["satellite"].append(satellite_repr)
                    object_reprs["nucleus"].append(nucleus_repr)
            
                # forward LLM
                # intervene in the decision of "left_of", using the repr of "right_of"
                single_inputs = copy.deepcopy(inputs)
                for key in single_inputs.keys():
                    single_inputs[key] = single_inputs[key][ori_dir_id].unsqueeze(0)
                
                if "llava" in model_name:
                    image_left_repr = image_reprs[ori_dir_id]
                elif "qwen" in model_name:
                    image_left_repr = image_reprs[image_indices[ori_dir_id]]
                else:
                    pass
                if positive:
                    # image_left_repr[object_patch_ids[ori_dir_id]["satellite"], :] += alpha * (object_reprs["satellite"][ori_dir_id] - object_reprs["satellite"][curr_dir_id]).expand(len(object_patch_ids[ori_dir_id]["satellite"]), -1)
                    # image_left_repr[object_patch_ids[ori_dir_id]["nucleus"], :] += alpha * (object_reprs["nucleus"][ori_dir_id] - object_reprs["nucleus"][curr_dir_id]).expand(len(object_patch_ids[ori_dir_id]["nucleus"]), -1)
                    image_left_repr[object_patch_ids[ori_dir_id]["satellite"], :] += alpha * (image_left_repr[object_patch_ids[ori_dir_id]["satellite"], :] - object_reprs["satellite"][curr_dir_id].expand(len(object_patch_ids[ori_dir_id]["satellite"]), -1))
                    image_left_repr[object_patch_ids[ori_dir_id]["nucleus"], :] += alpha * (image_left_repr[object_patch_ids[ori_dir_id]["nucleus"], :] - object_reprs["nucleus"][curr_dir_id].expand(len(object_patch_ids[ori_dir_id]["nucleus"]), -1))
                else:
                    image_left_repr[object_patch_ids[ori_dir_id]["satellite"], :] = image_left_repr[object_patch_ids[ori_dir_id]["satellite"], :] * (1 - alpha) + object_reprs["satellite"][curr_dir_id].expand(len(object_patch_ids[ori_dir_id]["satellite"]), -1) * alpha
                    image_left_repr[object_patch_ids[ori_dir_id]["nucleus"], :] = image_left_repr[object_patch_ids[ori_dir_id]["nucleus"], :] * (1 - alpha) + object_reprs["nucleus"][curr_dir_id].expand(len(object_patch_ids[ori_dir_id]["nucleus"]), -1) * alpha
                
                if "llava" in model_name:
                    outputs = model(
                        **single_inputs,
                        image_features=image_left_repr,
                        return_dict=True, 
                        output_hidden_states=True
                    )
                elif "qwen" in model_name:
                    outputs = model(
                        **single_inputs,
                        image_embeds=image_left_repr,
                        return_dict=True, 
                        output_hidden_states=True
                    )
    
                logits = outputs.logits[0, -1, :]  # (vocab_size)
                probs = F.softmax(logits, dim=-1)  # (vocab_size)
                for token, token_ids in direction_ids.items():
                    probs_intervene[alpha][token] = torch.sum(probs[token_ids], dim=-1).tolist()
                pred = torch.argmax(probs, dim=-1)
                pred_token = tokenizer.convert_ids_to_tokens([pred])[0]
                print(f"Sample {pair_id}: pred_token (intervene alpha {alpha}): {pred_token}, ans: left")
            print(f"----"*50)

        # save
        with jsonlines.open(save_path, "a") as f:
            data = {
                "pair_id": pair_id,
                # "probs_normal": probs_normal,
                "probs_intervene": probs_intervene
            }
            f.write(data)
        
    # draw
    plot_vit_direction_intervene(model_name, dataset_name, ori_dir_id=ori_dir_id, curr_dir_id=curr_dir_id, positive=positive)

def check_orthogonality(model_name, device, objects=["bowl", "candle"]):
    exp_dir = root_dir / "eval/WhatsUp/results/test_vit_directions_relation"
    exp_dir = os.path.join(exp_dir, f"{model_name}")
    relation_path = os.path.join(exp_dir, f"relation_difference.jsonl")
    
    cos_matrices = []
    with jsonlines.open(relation_path, "r") as f:
        data = list(f)
        for sample_id, sample in tqdm(enumerate(data), desc="Reading files"):
            satellite, nucleus = sample["satellite"]["name"], sample["nucleus"]["name"]
            if objects and not ((satellite == objects[0]) and (nucleus == objects[1])):
                continue
            directions_left = None
            directions_right = None
            directions_behind = None
            directions_in_front_of = None
            relation_reprs = [directions_left, directions_right, directions_behind, directions_in_front_of]
            for i in range(4):
                # get object reprs
                satellite_repr = torch.tensor(sample["satellite"]["repr"][i]).to(device).mean(dim=0).unsqueeze(0)  # shape: (len, dim)
                nucleus_repr = torch.tensor(sample["nucleus"]["repr"][i]).to(device).mean(dim=0).unsqueeze(0)  # shape: (len, dim)
                relation_reprs[i] = satellite_repr - nucleus_repr
            relation_reprs = torch.cat(relation_reprs, dim=0).cpu().numpy()  # (4, dim)
            # dimensionality reduction
            if False:
                pca = PCA()
                pca.fit_transform(relation_reprs)
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
                plt.title('Cumulative Explained Variance by Number of Principal Components')
                plt.xlabel('Number of Principal Components')
                plt.ylabel('Cumulative Explained Variance Ratio')
                plt.grid(True)
                plt.axhline(y=0.95, color='r', linestyle='-')
                plt.text(1, 0.96, '95% cut-off threshold', color = 'red', fontsize=10)
                plt.savefig(os.path.join(exp_dir, f"pca_{sample_id}.pdf"), bbox_inches='tight')

                n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
            n_components_95 = 3
            pca = PCA(n_components=n_components_95)
            relation_reprs = pca.fit_transform(relation_reprs)
            print(f"After dimension redunction: {relation_reprs.shape}")
            
            # compute the cosine value of the angles between the four directions, and get a cosine value matrix using numpy
            norm = np.linalg.norm(relation_reprs, axis=1, keepdims=True)
            normalized_relation_reprs = relation_reprs / norm
            cos_matrix = np.dot(normalized_relation_reprs, normalized_relation_reprs.T)
            # relation_reprs = F.normalize(relation_reprs, p=2, dim=-1)
            # cos_matrix = torch.matmul(relation_reprs, relation_reprs.T)  # (4, 4)
            cos_matrices.append(cos_matrix)
            # print(f"cos_matrix: {cos_matrix}")
    # cos_matrix = torch.mean(torch.stack(cos_matrices), dim=0)
    cos_matrix = np.mean(cos_matrices, axis=0)
    # visualize
    plt.figure()
    
    # gray_to_white = ['#fccf8a', '#FFFFFF']  # 灰色到白色
    # white_to_coral = ['#FFFFFF', '#768e7c']  # 白色到珊瑚色
    # cmap1 = mcolors.LinearSegmentedColormap.from_list('gray_to_white', gray_to_white, N=128)
    # cmap2 = mcolors.LinearSegmentedColormap.from_list('white_to_coral', white_to_coral, N=128)
    # colors = np.vstack((cmap1(np.linspace(0, 1, 128)), cmap2(np.linspace(0, 1, 128))))
    # two_slope_cmap = mcolors.LinearSegmentedColormap.from_list('two_slope_gray_white_coral', colors)
    # norm = mcolors.TwoSlopeNorm(vmin=cos_matrix.min(), vmax = cos_matrix.max(), vcenter=0)
    # plt.imshow(cos_matrix, cmap=two_slope_cmap, aspect='auto', norm=norm)
    
    colors = [mcolors.to_rgba("yellowgreen", 1), (1, 1, 1), mcolors.to_rgba("darkgray", 1)]  # (yellowgreen, gold)
    n_bins = 100
    cmap_name = 'my_list'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    plt.imshow(cos_matrix, interpolation='nearest', cmap=cm, vmin=-1, vmax=1)
    
    plt.colorbar()
    plt.xticks(range(4), ["left", "right", "behind", "in front of"])
    plt.yticks(range(4), ["left", "right", "behind", "in front of"])
    plt.xlabel("Direction vectors")
    plt.ylabel("Directions vectors")
    save_path = os.path.join(exp_dir, f"orthogonality.pdf")
    print("save_path", save_path)
    plt.savefig(save_path, bbox_inches='tight')   

def get_relation_representations_one_object(model_name, dataset_name, device, before_connector=False, objects=["bowl", "candle"], tsne=False, color="viridis"):
    """
    A rather long function that performs the following tasks:
    1. For each sampe, get the representations of the two objects in four images corresponding to the four relations.
    2. (Optional) Cluster the representations of each one of the two objects.
    """
    # set save path
    save_dir = root_dir / "eval/WhatsUp/results"
    exp_dir = os.path.join(save_dir, "test_vit_directions_relation_one_object", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # load model
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        model_dir = MODEL_NAME_TO_PATH[model_name]
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
            vision_process_func = None
        elif "qwen" in model_name:
            vit, llm = model.visual, model.model
            vision_process_func = process_vision_info_qwen
            if "qwen2_5" in model_name:
                replace_qwen2_5_vl_test_directions_processor_return_indices()
            else:
                replace_qwen2_vl_test_directions_processor_return_indices()
        elif "intern" in model_name:
            vision_process_func = load_image_intern_return_wh
        else:
            pass
    
    # load batch of images
    whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
    whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
    dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
    image_files = os.listdir(dir_name)
    image_names = [file_name.split('.')[0] for file_name in image_files]
    image_pairs = []
    for i, img_name in enumerate(image_names):
        obj_satellite = img_name.split('_')[0]
        obj_nucleus = img_name.split('_')[-1]
        image_pairs.append((obj_satellite, obj_nucleus))
    image_pairs = list(set(image_pairs))
    
    bboxes = {}
    bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
    with jsonlines.open(bboxes_path, "r") as f:
        for sample in f:
            bboxes[sample["image"]] = sample["bbox"]

    # forward
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    silhouette_score_list = []
    davies_bouldin_score_list = []
    calinski_harabasz_score_list = []
    adjusted_rand_score_list = []
        
    for pair_id, image_pair in enumerate(image_pairs):
        
        # if objects is None and pair_id > 14:
        #     break
            
        obj_satellite, obj_nucleus = image_pair
        if objects and not ((obj_satellite == objects[0]) and (obj_nucleus == objects[1])):
            continue
        
        # get four images for a pair
        sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
        sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
        sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
        
        # get inputs
        model_inputs = get_model_inputs(
            model_name=model_name,
            processor=processor,
            vision_process_func=vision_process_func,
            image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
            prompts=["describe the image and tell me what is the main object in the image" for _ in range(4)]
        )
        if "llava" in model_name:
            inputs = model_inputs
        if "qwen" in model_name:
            inputs, image_indices = model_inputs
        elif "intern" in model_name:
            inputs = model_inputs
            image_indices = []
            st = 0
            n_patches_per_tile = int(model_config.vision_config.image_size // (model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)) ) ** 2
            for idx in range(len(inputs["num_patches_list"])):
                image_indices.append(slice(st, st + inputs["num_patches_list"][idx] * n_patches_per_tile))
                st += inputs["num_patches_list"][idx] * n_patches_per_tile
            # print("n_patches_list", inputs["num_patches_list"])
            # print("image_indices", image_indices)

        # get object info (resized bboxes and patch ids)
        if True:
            # get resized bboxes
            sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
            resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                # draw bboxes
                # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                
                # 1. get original sizes
                image = Image.open(image_path).convert('RGB')
                original_width, original_height = image.size
                image.close()
                # 2. get current size (after resize)
                if "llava" in model_name:
                    height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, height, width = image_grid_thw[idx].cpu().numpy()
                    height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                    width *= model.config.vision_config.spatial_patch_size
                elif "intern" in model_name:
                    block_image_width, block_image_height = inputs["block_wh_list"][idx]
                    height = block_image_height * model_config.vision_config.image_size
                    width = block_image_width * model_config.vision_config.image_size
                # 3. get resized bboxes
                scale_width = width / original_width
                scale_height = height / original_height
                print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale_width}, {scale_height}")
                # import pdb; pdb.set_trace()
                x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                
            # get object patches
            object_patch_ids = [{"satellite": None, "nucleus": None} for _ in range(4)]
            if "llava" in model_name:
                patch_size = model_config.vision_config.patch_size
            elif "qwen" in model_name:
                patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
            elif "intern" in model_name:
                patch_size = model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                            # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids[idx][obj_name] = obj_patch_ids
        
        # forward as a batch     
        # get vit output (before llm or before connector)
        image_reprs = None  # llava: (bsz, len(24*24), dim)  qwen2.5-vl: (len, dim)
        with torch.no_grad():
            if "intern" in model_name:
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(device)
            else:
                inputs.to(device)
            if before_connector:
                if "llava" in model_name:
                    image_reprs = vit(
                        pixel_values=inputs["pixel_values"],
                        output_hidden_states=True,
                    ).last_hidden_state  # (batch_size, seq_len, dim)
                elif "qwen" in model_name:
                    image_reprs = None
                else:
                    pass
            else:
                if "llava" in model_name:
                    image_reprs = model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        vision_feature_layer=model.config.vision_feature_layer,
                        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    )
                elif "qwen" in model_name:
                    inputs["pixel_values"] = inputs["pixel_values"].type(vit.dtype)
                    image_reprs = vit(
                        inputs["pixel_values"], 
                        grid_thw=inputs["image_grid_thw"], 
                    )
                    # original_image size: [1280, 960], image_size: [1148, 840]
                    # print("pixel_values", inputs["pixel_values"].shape)  # [19680, 1176]
                    # print("image_reprs", image_reprs.shape)  # [4920, 3584]  4920 = 4 * (1148 / 28) * (840 / 28)
                    # import pdb; pdb.set_trace()
                elif "intern" in model_name:
                    image_reprs = model.extract_feature(
                        pixel_values=inputs["pixel_values"],
                    )
                    image_reprs = image_reprs.reshape(-1, image_reprs.shape[-1])  # (n_tiles * num_patches, len(24*24), dim)
                else:
                    pass
        
        # get the image representation
        object_reprs = {"satellite": [], "nucleus": []}
        for idx, image_path in enumerate(sample_image_paths):
            # 1. get the pooled representation for each object
            if "llava" in model_name:
                img_repr = image_reprs[idx]
            elif "qwen" in model_name:
                img_repr = image_reprs[image_indices[idx]]
            elif "intern" in model_name:
                img_repr = image_reprs[image_indices[idx]]
            else:
                pass 
            satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].float().cpu().tolist()  # .mean(dim=0).unsqueeze(0).cpu().numpy()
            nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].float().cpu().tolist()
            object_reprs["satellite"].append(satellite_repr)
            object_reprs["nucleus"].append(nucleus_repr)
        
        # cluster
        # satellite
        obj_reprs_left = {"satellite": [], "nucleus": []}
        obj_reprs_right = {"satellite": [], "nucleus": []}
        obj_reprs_behind = {"satellite": [], "nucleus": []}
        obj_reprs_in_front_of = {"satellite": [], "nucleus": []}
        obj_reprs = [obj_reprs_left, obj_reprs_right, obj_reprs_behind, obj_reprs_in_front_of]
        for i in range(4):
            # get object reprs
            satellite_repr = torch.tensor(object_reprs["satellite"][i]).to(device)  # shape: (len, dim) .mean(dim=0).unsqueeze(0)
            nucleus_repr = torch.tensor(object_reprs["nucleus"][i]).to(device)  # shape: (len, dim) .mean(dim=0).unsqueeze(0)
            
            # randomly sample patches from satellite and nucleus
            s_len = satellite_repr.shape[0]
            n_len = nucleus_repr.shape[0]
            for _ in range(100):
                satellite_patch_ids = random.sample(range(s_len), s_len // 2)
                nucleus_patch_ids = random.sample(range(n_len), n_len // 2)
                satellite_repr_ = satellite_repr[satellite_patch_ids, :].mean(dim=0).unsqueeze(0)
                nucleus_repr_ = nucleus_repr[nucleus_patch_ids, :].mean(dim=0).unsqueeze(0)
                obj_reprs[i]["satellite"].append(satellite_repr_)
                obj_reprs[i]["nucleus"].append(nucleus_repr_)
              
        # prepare data for dimensionality reduction & clustering
        object_types = ["satellite", "nucleus"]
        for obj_type in object_types:
            object_reprs = [torch.cat(obj_reprs[i][obj_type], dim=0) for i in range(4)]
            all_reprs = torch.cat(object_reprs, dim=0)
            print("all_reprs", all_reprs.shape)  # (4*len(obj), 3584)
            # set label (left: 0, right: 1, behind: 2, in front of: 3)
            all_reprs = all_reprs.cpu().numpy()
            scaler = StandardScaler()
            all_reprs = scaler.fit_transform(all_reprs)
            minmax_scaler = MinMaxScaler()
            all_reprs = minmax_scaler.fit_transform(all_reprs)
            true_labels = []
            for i in range(4):
                true_labels.extend([i] * len(obj_reprs[i][obj_type]))
            
            if True:
                if tsne:
                    # perform t-SNE for data of all four directions
                    print("t-SNE start")
                    tsne = TSNE(n_components=2, random_state=41, perplexity=30, n_iter=1000)
                    tsne_results = tsne.fit_transform(all_reprs)
                    print("t-SNE finished")
                else:
                    # perform PCA for data of all four directions
                    print("PCA start")
                    pca = PCA(n_components=2)
                    pca_results = pca.fit_transform(all_reprs)
                    print("PCA finished")
                
                # perform k-means clustering
                print("k-means start")
                n_clusters = 4
                kmeans = KMeans(n_clusters=n_clusters, random_state=14, n_init='auto')
                results_for_clustering = tsne_results if tsne else pca_results
                kmeans.fit(results_for_clustering)
                labels = kmeans.labels_
                silhouette = silhouette_score(all_reprs, labels)
                db_index = davies_bouldin_score(all_reprs, labels)
                ch_index = calinski_harabasz_score(all_reprs, labels)
                ari = adjusted_rand_score(true_labels, labels)
                silhouette_score_list.append(silhouette)
                davies_bouldin_score_list.append(db_index)
                calinski_harabasz_score_list.append(ch_index)
                adjusted_rand_score_list.append(ari)
                print(f"Silhouette Coefficient: {silhouette}")
                print(f"Davies-Bouldin Index: {db_index}")
                print(f"Calinski-Harabasz Index: {ch_index}")
                print(f"Adjusted Rand Index: {ari}")
                
                cluster_labels = kmeans.labels_
                print(f"k-means for sample {pair_id} finished \n{'----' * 20}")
                
                # visualize the t-SNE results
                if True:
                    exp_sample_dir = os.path.join(exp_dir, f"cluster", f"{obj_satellite}_{obj_nucleus}")
                    os.makedirs(exp_sample_dir, exist_ok=True)
                    
                    if tsne:
                        plt.figure()
                        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_labels, cmap=color, s=20, alpha=1)
                        plt.xlabel('t-SNE Dimension 1')
                        plt.ylabel('t-SNE Dimension 2')
                        cbar = plt.colorbar(scatter, ticks=range(4))
                        cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                        save_path = os.path.join(exp_sample_dir, f"object_repr_tsne_normalized_{obj_type}.pdf")
                        print("save_path", save_path)
                        plt.savefig(save_path, bbox_inches='tight')
                    else:
                        plt.figure()
                        scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=true_labels, cmap=color, s=20, alpha=1)
                        plt.xlabel('PCA Dimension 1')
                        plt.ylabel('PCA Dimension 2')
                        cbar = plt.colorbar(scatter, ticks=range(4))
                        cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                        save_path = os.path.join(exp_sample_dir, f"object_repr_pca_normalized_{obj_type}.pdf")
                        print("save_path", save_path)
                        plt.savefig(save_path, bbox_inches='tight')
                    
                    # visualize the k-means clustering results
                    plt.figure()
                    scatter = plt.scatter(results_for_clustering[:, 0], results_for_clustering[:, 1], c=cluster_labels, cmap=color, s=20, alpha=1)
                    dim_name = "t-SNE" if tsne else "PCA"
                    plt.xlabel(f'{dim_name} Dimension 1')
                    plt.ylabel(f'{dim_name} Dimension 2')
                    cbar = plt.colorbar(scatter, ticks=range(n_clusters))
                    cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
                    save_path = os.path.join(exp_sample_dir, f"object_repr_kmeans_{obj_type}.pdf")
                    save_path = ''.join(save_path.split(".")[:-1]) + "_normalized.pdf"
                    save_path = ''.join(save_path.split(".")[:-1]) + f"_after_{dim_name}.pdf"
                    print("save_path", save_path)
                    plt.savefig(save_path, bbox_inches='tight')
    
    avg_silhouette_score = np.mean(silhouette_score_list)
    avg_davies_bouldin_score = np.mean(davies_bouldin_score_list)
    avg_calinski_harabasz_score = np.mean(calinski_harabasz_score_list)
    avg_adjusted_rand_score = np.mean(adjusted_rand_score_list)
    print(f"avg_silhouette_score: {avg_silhouette_score}")
    print(f"avg_davies_bouldin_score: {avg_davies_bouldin_score}")
    print(f"avg_calinski_harabasz_score: {avg_calinski_harabasz_score}")
    print(f"avg_adjusted_rand_score: {avg_adjusted_rand_score}")
    
    cluster_res_path = os.path.join(exp_dir, f"cluster_res.jsonl")
    silhouette_score_list_converted = [float(score) for score in silhouette_score_list]
    davies_bouldin_score_list_converted = [float(score) for score in davies_bouldin_score_list]
    calinski_harabasz_score_list_converted = [float(score) for score in calinski_harabasz_score_list]
    adjusted_rand_score_list_converted = [float(score) for score in adjusted_rand_score_list]
    with jsonlines.open(cluster_res_path, "w") as f:
        f.write({
            "silhouette_score": silhouette_score_list_converted,
            "davies_bouldin_score": davies_bouldin_score_list_converted,
            "calinski_harabasz_score": calinski_harabasz_score_list_converted,
            "adjusted_rand_score": adjusted_rand_score_list_converted
        })            

def check_direction_language_alignment(model_name, dataset_name, device):
    
    # set save path
    save_dir = root_dir / "eval/WhatsUp/results"
    exp_dir = os.path.join(save_dir, "test_vit_direction_language_alignment", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, f"result.jsonl")
    
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    model_dir = MODEL_NAME_TO_PATH[model_name]
    model_config = AutoConfig.from_pretrained(model_dir)
    
    vit, llm = None, None
    if "qwen" in model_name:
        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
        replace_qwen2_5_vl_test_directions_processor_return_indices()
    elif "intern" in model_name:
        vision_process_func = load_image_intern
    else:
        vit, llm = model.vision_tower, model.language_model
        vision_process_func = None
    
    # load data
    whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
    whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
    dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
    image_files = os.listdir(dir_name)
    image_names = [file_name.split('.')[0] for file_name in image_files]
    image_pairs = []
    for i, img_name in enumerate(image_names):
        obj_satellite = img_name.split('_')[0]
        obj_nucleus = img_name.split('_')[-1]
        image_pairs.append((obj_satellite, obj_nucleus))
    image_pairs = list(set(image_pairs))
    
    bboxes = {}
    bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
    with jsonlines.open(bboxes_path, "r") as f:
        for sample in f:
            bboxes[sample["image"]] = sample["bbox"]
    
    # annotation_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data"
    # if "whatsup_a" in dataset_name:
    #     annotation_path = os.path.join(annotation_dir, "controlled_images_dataset.json")
    # else:
    #     annotation_path = os.path.join(annotation_dir, "controlled_clevr_dataset.json")
    # annotations = json.load(open(annotation_path))
    
    # start
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    labels = ["left", "right", "behind", "in front of"]
    for pair_id, image_pair in enumerate(image_pairs):
        obj_satellite, obj_nucleus = image_pair
        sample_dir = os.path.join(exp_dir, f"{obj_satellite}_{obj_nucleus}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # preparation
        if True:
            # get four images for a pair
            sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
            sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
            sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
            
            # get inputs
            descriptions = []
            relation_phrases = ["to the left of", "to the right of", "behind", "in front of"]
            for idx, image_name in enumerate(sample_image_names):
                descriptions.append(f"What is {relation_phrases[idx]} the {obj_nucleus}?")
            
            model_inputs = get_model_inputs(
                model_name=model_name,
                processor=processor,
                vision_process_func=vision_process_func,
                image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
                prompts=descriptions
            )
            if "qwen" in model_name:
                inputs, image_indices = model_inputs
            else:
                inputs = model_inputs
            inputs.to(device)

            # get resized bboxes
            sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
            resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                # draw bboxes
                # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                
                # 1. get original sizes
                image = Image.open(image_path).convert('RGB')
                original_width, original_height = image.size
                image.close()
                # 2. get current size (after resize)
                if "llava" in model_name:
                    height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, height, width = image_grid_thw[idx].cpu().numpy()
                    height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                    width *= model.config.vision_config.spatial_patch_size
                # 3. get resized bboxes
                scale = width / original_width
                # print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
                # import pdb; pdb.set_trace()
                x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                
            # get object patches
            object_patch_ids = [{"satellite": None, "nucleus": None, "background": None} for _ in range(4)]
            if "llava" in model_name:
                patch_size = model_config.vision_config.patch_size
            else:
                patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                            # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids[idx][obj_name] = obj_patch_ids
                # get background patch ids
                object_patch_ids[idx]["background"] = list(set(range(int(width // patch_size) * int(height // patch_size))) - set(object_patch_ids[idx]["satellite"]) - set(object_patch_ids[idx]["nucleus"]))
    
        # get position ids for object and relation tokens
        object_ids_in_llm = [{"satellite": None, "nucleus": None, "background": None} for _ in range(4)]
        text_ids_in_llm = [None for _ in range(4)]
        relation_tokens = [["to", "the", "left", "of"], ["to", "the", "right", "of"], ["behind"], ["in", "front", "of"]]
        if "qwen" in model_name:
            # <|endoftext|>...<|endoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>What is in front of the can?<|im_end|>\n<|im_start|>assistant\n
            input_ids = inputs.input_ids
            input_content = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            input_tokens = [processor.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            # for idx in range(len(input_tokens)):
            #     input_tokens[idx] = [token for token in input_tokens[idx] if token != "<|image_pad|>"]
            # print(input_content)
            # print(input_tokens)
            vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            image_tokens_start_pos= (input_ids == vision_start_id).nonzero().tolist()  # [[0, 14], [1, 14], [2, 17], [3, 15]]
            image_tokens_end_pos = (input_ids == vision_end_id).nonzero().tolist()  # [[0, 1245], [1, 1245], [2, 1248], [3, 1246]]
            # print(f"image_tokens_start_pos: {image_tokens_start_pos}")
            # print(f"image_tokens_end_pos: {image_tokens_end_pos}")
            for idx in range(4):
                img_shift = image_tokens_start_pos[idx][1] + 1
                for key in object_ids_in_llm[idx].keys():
                    object_ids_in_llm[idx][key] = [patch_id + img_shift for patch_id in object_patch_ids[idx][key]]
            for idx in range(4):
                text_st_id = image_tokens_end_pos[idx][1] + 1
                text_tokens = input_tokens[idx][text_st_id:]
                r_tokens = relation_tokens[idx]
                print(f"relation_tokens: {r_tokens}")
                print(f"text_tokens: {text_tokens}")
                r_token_pos_in_text = []
                for text_pos, text_token in enumerate(text_tokens):
                    if r_tokens[0] in text_token:
                        r_token_pos_in_text = [text_pos + k for k in range(len(r_tokens))]
                        break
                text_ids_in_llm[idx] = [text_st_id + text_pos for text_pos in r_token_pos_in_text]
            # print(f"object_ids_in_llm: {object_ids_in_llm}")
            print(f"text_ids_in_llm: {text_ids_in_llm}")
            # import pdb; pdb.set_trace()
        else:
            pass
        
        # forward and get llm outputs of each layer
        with torch.no_grad():
            outputs = model(
                **inputs, 
                return_dict=True, 
                output_hidden_states=True
            )
        all_hidden_states = outputs.hidden_states  # n_layers *(bsz, len, dim)
        all_hidden_states = all_hidden_states[1:]  # remove the first layer (embedding layer)
        
        for idx in range(4):  # for each image(direction)
            all_sims = {key: [] for key in ["satellite", "nucleus", "background", "direction"]}
            for layer_id in range(len(all_hidden_states)):
                layer_hidden_states = all_hidden_states[layer_id]
            
                # get direction vectors
                satellite_repr = layer_hidden_states[idx, object_ids_in_llm[idx]["satellite"], :].mean(dim=0)
                nucleus_repr = layer_hidden_states[idx, object_ids_in_llm[idx]["nucleus"], :].mean(dim=0)
                background_repr = layer_hidden_states[idx, object_ids_in_llm[idx]["background"], :].mean(dim=0)
                direction_repr = satellite_repr - nucleus_repr
                
                # get text vectors
                relation_repr = layer_hidden_states[idx, text_ids_in_llm[idx], :].mean(dim=0)
            
                # compute the similarities between the object/direction vectors and relation tokens
                text_satellite_sim = torch.cosine_similarity(satellite_repr, relation_repr, dim=-1).item()
                text_nucleus_sim = torch.cosine_similarity(nucleus_repr, relation_repr, dim=-1).item()
                text_background_sim = torch.cosine_similarity(background_repr, relation_repr, dim=-1).item()
                text_direction_sim = torch.cosine_similarity(direction_repr, relation_repr, dim=-1).item()
                
                # save
                all_sims["satellite"].append(text_satellite_sim)
                all_sims["nucleus"].append(text_nucleus_sim)
                all_sims["background"].append(text_background_sim)
                all_sims["direction"].append(text_direction_sim)
        
            # plot
            fig_path = os.path.join(sample_dir, f"text_sims_{relations[idx]}.pdf")
            
            plt.figure()
            colors = [mcolors.to_hex(plt.cm.viridis(i / len(labels))) for i in range(len(labels))]
            colors = colors[::-1]
            # colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
            # markers = ["o", "s", "D", "^"]
            layer_ids = list(range(len(all_hidden_states)))
            layer_ids = [layer_id + 1 for layer_id in layer_ids]
            for i, obj in enumerate(["satellite", "nucleus", "background", "direction"]):
                # mean = np.mean(all_sims[obj], axis=0)
                # std_error = np.std(all_sims[obj], axis=0) / np.sqrt(len(all_sims[obj]))
                plt.plot(
                    layer_ids,
                    all_sims[obj],
                    color=colors[i],
                    label=obj,
                    alpha=1, 
                    # marker=markers[i],
                    # markersize=6, 
                    # markerfacecolor="none",
                    # markeredgewidth=1.5
                )
                # plt.fill_between(
                #     layer_ids, 
                #     mean - std_error, 
                #     mean + std_error, 
                #     color=colors[i], 
                #     alpha=0.1, 
                # )
                
            plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
            plt.ylabel("Similarity with the relation text", color=mcolors.to_rgba('black', 1))
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
            
            plt.legend()
            
            plt.savefig(fig_path, bbox_inches='tight')
            print(fig_path)
            
        break

def check_direction_spatial_reasoning_attention(model_name, dataset_name, device):
    
    # set save path
    save_dir = root_dir / "eval/WhatsUp/results"
    exp_dir = os.path.join(save_dir, "test_vit_direction_spatial_reasoning_attention", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, f"result.jsonl")
    
    # load model
    model, processor, tokenizer = load_model(model_name, device, use_flash_attention=False)
    model_dir = MODEL_NAME_TO_PATH[model_name]
    model_config = AutoConfig.from_pretrained(model_dir)
    
    vit, llm = None, None
    if "qwen" in model_name:
        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
        replace_qwen2_5_vl_test_directions_processor_return_indices()
    elif "intern" in model_name:
        vision_process_func = load_image_intern
    else:
        vit, llm = model.vision_tower, model.language_model
        vision_process_func = None
    
    # load data
    whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
    whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
    dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
    image_files = os.listdir(dir_name)
    image_names = [file_name.split('.')[0] for file_name in image_files]
    image_pairs = []
    for i, img_name in enumerate(image_names):
        obj_satellite = img_name.split('_')[0]
        obj_nucleus = img_name.split('_')[-1]
        image_pairs.append((obj_satellite, obj_nucleus))
    image_pairs = list(set(image_pairs))
    
    bboxes = {}
    bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
    with jsonlines.open(bboxes_path, "r") as f:
        for sample in f:
            bboxes[sample["image"]] = sample["bbox"]
    
    # annotation_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data"
    # if "whatsup_a" in dataset_name:
    #     annotation_path = os.path.join(annotation_dir, "controlled_images_dataset.json")
    # else:
    #     annotation_path = os.path.join(annotation_dir, "controlled_clevr_dataset.json")
    # annotations = json.load(open(annotation_path))
    
    # start
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    labels = ["left", "right", "behind", "in front of"]
    for pair_id, image_pair in enumerate(image_pairs):
        obj_satellite, obj_nucleus = image_pair
        sample_dir = os.path.join(exp_dir, f"{obj_satellite}_{obj_nucleus}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # preparation
        if True:
            # get four images for a pair
            sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
            sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
            sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
            
            # get inputs
            descriptions = []
            relation_phrases = ["to the left of", "to the right of", "behind", "in front of"]
            for idx, image_name in enumerate(sample_image_names):
                descriptions.append(f"What is {relation_phrases[idx]} the {obj_nucleus}?")
            
            model_inputs = get_model_inputs(
                model_name=model_name,
                processor=processor,
                vision_process_func=vision_process_func,
                image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
                prompts=descriptions
            )
            if "qwen" in model_name:
                inputs, image_indices = model_inputs
            else:
                inputs = model_inputs
            inputs.to(device)

            # get resized bboxes
            sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
            resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                # draw bboxes
                # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                
                # 1. get original sizes
                image = Image.open(image_path).convert('RGB')
                original_width, original_height = image.size
                image.close()
                # 2. get current size (after resize)
                if "llava" in model_name:
                    height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, height, width = image_grid_thw[idx].cpu().numpy()
                    height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                    width *= model.config.vision_config.spatial_patch_size
                # 3. get resized bboxes
                scale = width / original_width
                # print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
                # import pdb; pdb.set_trace()
                x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                
            # get object patches
            object_patch_ids = [{"satellite": None, "nucleus": None, "background": None} for _ in range(4)]
            if "llava" in model_name:
                patch_size = model_config.vision_config.patch_size
            else:
                patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                            # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids[idx][obj_name] = obj_patch_ids
                # get background patch ids
                object_patch_ids[idx]["background"] = list(set(range(int(width // patch_size) * int(height // patch_size))) - set(object_patch_ids[idx]["satellite"]) - set(object_patch_ids[idx]["nucleus"]))
    
        # get position ids for object and relation tokens
        object_ids_in_llm = [{"satellite": None, "nucleus": None, "background": None} for _ in range(4)]
        text_ids_in_llm = [None for _ in range(4)]
        relation_tokens = [["to", "the", "left", "of"], ["to", "the", "right", "of"], ["behind"], ["in", "front", "of"]]
        if "qwen" in model_name:
            # <|endoftext|>...<|endoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>What is in front of the can?<|im_end|>\n<|im_start|>assistant\n
            input_ids = inputs.input_ids
            input_content = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            input_tokens = [processor.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            # for idx in range(len(input_tokens)):
            #     input_tokens[idx] = [token for token in input_tokens[idx] if token != "<|image_pad|>"]
            # print(input_content)
            # print(input_tokens)
            vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            image_tokens_start_pos= (input_ids == vision_start_id).nonzero().tolist()  # [[0, 14], [1, 14], [2, 17], [3, 15]]
            image_tokens_end_pos = (input_ids == vision_end_id).nonzero().tolist()  # [[0, 1245], [1, 1245], [2, 1248], [3, 1246]]
            # print(f"image_tokens_start_pos: {image_tokens_start_pos}")
            # print(f"image_tokens_end_pos: {image_tokens_end_pos}")
            for idx in range(4):
                img_shift = image_tokens_start_pos[idx][1] + 1
                for key in object_ids_in_llm[idx].keys():
                    object_ids_in_llm[idx][key] = [patch_id + img_shift for patch_id in object_patch_ids[idx][key]]
            for idx in range(4):
                text_st_id = image_tokens_end_pos[idx][1] + 1
                text_tokens = input_tokens[idx][text_st_id:]
                r_tokens = relation_tokens[idx]
                print(f"relation_tokens: {r_tokens}")
                print(f"text_tokens: {text_tokens}")
                r_token_pos_in_text = []
                for text_pos, text_token in enumerate(text_tokens):
                    if r_tokens[0] in text_token:
                        r_token_pos_in_text = [text_pos + k for k in range(len(r_tokens))]
                        break
                text_ids_in_llm[idx] = [text_st_id + text_pos for text_pos in r_token_pos_in_text]
            # print(f"object_ids_in_llm: {object_ids_in_llm}")
            print(f"text_ids_in_llm: {text_ids_in_llm}")
            # import pdb; pdb.set_trace()
        else:
            pass
        
        # forward and get llm outputs of each layer
        with torch.no_grad():
            outputs = model(
                **inputs, 
                return_dict=True, 
                output_attentions=True
            )
        all_attentions = outputs.attentions  # n_layers *(bsz, num_heads, len, len)
        
        for idx in range(4):  # for each image(direction)
            all_text_attentions = {key: [] for key in ["satellite", "nucleus", "background"]}
            for layer_id in range(len(all_attentions)):
                attentions = all_attentions[layer_id][idx]
                reduced_attentions = torch.mean(attentions, dim=0)
            
                # compute the similarities between the object/direction vectors and relation tokens
                text_satellite_attn = torch.mean(reduced_attentions[text_ids_in_llm[idx]][:, object_ids_in_llm[idx]["satellite"]]).mean().item()
                text_nucleus_attn = torch.mean(reduced_attentions[text_ids_in_llm[idx]][:, object_ids_in_llm[idx]["nucleus"]]).mean().item()
                text_background_attn = torch.mean(reduced_attentions[text_ids_in_llm[idx]][:, object_ids_in_llm[idx]["background"]]).mean().item()
                
                # save
                all_text_attentions["satellite"].append(text_satellite_attn)
                all_text_attentions["nucleus"].append(text_nucleus_attn)
                all_text_attentions["background"].append(text_background_attn)
        
            # plot
            fig_path = os.path.join(sample_dir, f"text_sims_{relations[idx]}.pdf")
            
            plt.figure()
            colors = [mcolors.to_hex(plt.cm.viridis(i / len(labels))) for i in range(len(labels))]
            colors = colors[::-1]
            # colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
            # markers = ["o", "s", "D", "^"]
            layer_ids = list(range(len(all_attentions)))
            layer_ids = [layer_id + 1 for layer_id in layer_ids]
            for i, obj in enumerate(["satellite", "nucleus", "background"]):
                # mean = np.mean(all_sims[obj], axis=0)
                # std_error = np.std(all_sims[obj], axis=0) / np.sqrt(len(all_sims[obj]))
                plt.plot(
                    layer_ids,
                    all_text_attentions[obj],
                    color=colors[i],
                    label=obj,
                    alpha=1, 
                    # marker=markers[i],
                    # markersize=6, 
                    # markerfacecolor="none",
                    # markeredgewidth=1.5
                )
                # plt.fill_between(
                #     layer_ids, 
                #     mean - std_error, 
                #     mean + std_error, 
                #     color=colors[i], 
                #     alpha=0.1, 
                # )
                
            plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
            plt.ylabel("Similarity with the relation text", color=mcolors.to_rgba('black', 1))
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
            
            plt.legend()
            
            plt.savefig(fig_path, bbox_inches='tight')
            print(fig_path)
            
        break

def explore_1d_pos_embed_visual_geometry(model_name, device, row=True):
    
    # set save path
    exp_dir = root_dir / "figures"
    exp_dir = os.path.join(exp_dir, "explore_1d_pos_embed_visual_geometry", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
        elif "intern" in model_name:
            vit, llm = model.vision_model, model.language_model
        else:
            pass
    
    # Explore pos embed
    # 1. get 1d positional embeddings
    if "llava" in model_name:
        pos_embed = vit.vision_model.embeddings.position_embedding.weight
        print(f"pos_embed shape: {pos_embed.shape}")  # [577, 1024]  577 = 1 + 24*24, 24 = 336//4
        pos_embed = pos_embed[1:]  # remove the first token
    elif "intern" in model_name:
        pos_embed = vit.embeddings.position_embedding
        print(f"pos_embed shape: {pos_embed.shape}")  # [1, 1025, 1024]  # 1025 = 1 + 32*32, 32 = 448//14
        pos_embed = pos_embed[0, 1:]  # remove the first token
    
    # 2. PCA
    pos_embed = pos_embed.cpu().detach().float().numpy()
    true_labels = []
    if "llava" in model_name:
        width = 24
    elif "intern" in model_name:
        width = 32
    else:
        pass
    tag = "row" if row else "col"
    if row:
        true_labels = [row for row in range(width) for _ in range(width)]
    else:
        true_labels = [col for _ in range(width) for col in range(width)]
    print(f"true_labels: {true_labels}")
    
    # if True:
    #     pca = PCA()
    #     pca.fit_transform(pos_embed)
    #     explained_variance_ratio = pca.explained_variance_ratio_
    #     cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    #     plt.title('Cumulative Explained Variance by Number of Principal Components')
    #     plt.xlabel('Number of Principal Components')
    #     plt.ylabel('Cumulative Explained Variance Ratio')
    #     plt.grid(True)
    #     plt.axhline(y=0.95, color='r', linestyle='-')
    #     plt.text(1, 0.96, '95% cut-off threshold', color = 'red', fontsize=10)
    #     plt.savefig(os.path.join(exp_dir, f"pca_test.pdf"), bbox_inches='tight')

    #     n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
    # pca = PCA(n_components=n_components_95)
    # pca_results = pca.fit_transform(pos_embed)
    # print(f"After dimension redunction: {pca_results.shape}")
    
    # tsne
    ppl = 30
    tsne_n_iter = 1000
    tsne = TSNE(n_components=2, random_state=41, perplexity=ppl, n_iter=tsne_n_iter, verbose=1)
    tsne_results = tsne.fit_transform(pos_embed)

    # 3. plot
    # plt.figure()
    # scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], cmap='viridis', s=20, alpha=1)
    # plt.xlabel('PCA Dimension 1')
    # plt.ylabel('PCA Dimension 2')
    # cbar = plt.colorbar(scatter, ticks=range(4))
    # cbar.set_ticklabels(['Left', 'Right', 'Behind', 'In front of'])
    # save_path = os.path.join(exp_dir, f"pos_embed_pca_normalized.pdf")
    # print("save_path", save_path)
    # plt.savefig(save_path, bbox_inches='tight')
    
    plt.figure()
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_labels, cmap='viridis', s=20, alpha=1)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    cbar = plt.colorbar(scatter, ticks=range(width))
    cbar.set_ticklabels([f"{tag} {i}" for i in range(width)])
    
    save_path = os.path.join(exp_dir, f"pos_embed_tsne_normalized_{tag}-ppl_{ppl}-n_iter_{tsne_n_iter}.pdf")
    print("save_path", save_path)
    plt.savefig(save_path, bbox_inches='tight')

def explore_1d_pos_embed_direction_vectors_old(model_name, device):
    
    # set save path
    exp_dir = root_dir / "figures"
    exp_dir = os.path.join(exp_dir, "explore_1d_pos_embed_direction_vectors", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
        elif "intern" in model_name:
            vit, llm = model.vision_model, model.language_model
        else:
            pass
    
    # Explore pos embed
    # 1. get 1d positional embeddings
    if "llava" in model_name:
        pos_embed = vit.vision_model.embeddings.position_embedding.weight
        print(f"pos_embed shape: {pos_embed.shape}")  # [577, 1024]  577 = 1 + 24*24, 24 = 336//4
        pos_embed = pos_embed[1:]  # remove the first token
    elif "intern" in model_name:
        pos_embed = vit.embeddings.position_embedding
        print(f"pos_embed shape: {pos_embed.shape}")  # [1, 1025, 1024]  # 1025 = 1 + 32*32, 32 = 448//14
        pos_embed = pos_embed[0, 1:]  # remove the first token
    
    # 2. PCA
    pos_embed = pos_embed.cpu().detach().float().numpy()
    if "llava" in model_name:
        width = 24
    elif "intern" in model_name:
        width = 32
    else:
        pass
    
    # get position points
    half_width = width // 2
    top_pos_embed = pos_embed[half_width - 1]
    bottom_pos_embed = pos_embed[(width - 1) * width + half_width - 1]
    left_pos_embed = pos_embed[half_width * width]
    right_pos_embed = pos_embed[half_width * width + (width - 1)]
    top_pos_coord = (0, half_width - 1)
    bottom_pos_coord = (width - 1, half_width - 1)
    left_pos_coord = (half_width, 0)
    right_pos_coord = (half_width, width - 1)
    print(f"top_pos_embed: {top_pos_coord}, {half_width - 1}")
    print(f"bottom_pos_embed: {bottom_pos_coord}", {(width - 1) * width + half_width - 1})
    print(f"left_pos_embed: {left_pos_coord}, {half_width * width}")
    print(f"right_pos_embed: {right_pos_coord}, {half_width * width + (width - 1)}")
    
    # get direction vectors
    top_vec = top_pos_embed - bottom_pos_embed
    bottom_vec = bottom_pos_embed - top_pos_embed
    left_vec = left_pos_embed - right_pos_embed
    right_vec = right_pos_embed - left_pos_embed
    top_left_vec = top_pos_embed - left_pos_embed
    top_right_vec = top_pos_embed - right_pos_embed
    bottom_left_vec = bottom_pos_embed - left_pos_embed
    bottom_right_vec = bottom_pos_embed - right_pos_embed
    
    # perform PCA to direction vectors (dim -> 3)
    pos_embed = np.array([top_vec, bottom_vec, left_vec, right_vec,
                          top_left_vec, top_right_vec, bottom_left_vec, bottom_right_vec])
    pca = PCA(n_components=3)
    pos_embed = pca.fit_transform(pos_embed)
    # tsne = TSNE(n_components=3, random_state=41, perplexity=7, n_iter=1000, verbose=1)
    # pos_embed = tsne.fit_transform(pos_embed)
    top_vec, bottom_vec, left_vec, right_vec, top_left_vec, top_right_vec, bottom_left_vec, bottom_right_vec = pos_embed
    
    # plot the vectors in a 3D figure
    fig = plt.figure()
    
    # plt.quiver(0, 0, top_vec[0], top_vec[1], angles='xy', scale_units='xy', scale=1, color='r', label='Top')
    # plt.quiver(0, 0, bottom_vec[0], bottom_vec[1], angles='xy', scale_units='xy', scale=1, color='b', label='Bottom')
    # plt.quiver(0, 0, left_vec[0], left_vec[1], angles='xy', scale_units='xy', scale=1, color='g', label='Left')
    # plt.quiver(0, 0, right_vec[0], right_vec[1], angles='xy', scale_units='xy', scale=1, color='c', label='Right')
    # plt.quiver(0, 0, top_left_vec[0], top_left_vec[1], angles='xy', scale_units='xy', scale=1, color='m', label='Top Left')
    # plt.quiver(0, 0, top_right_vec[0], top_right_vec[1], angles='xy', scale_units='xy', scale=1, color='y', label='Top Right')
    # plt.quiver(0, 0, bottom_left_vec[0], bottom_left_vec[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Bottom Left')
    # plt.quiver(0, 0, bottom_right_vec[0], bottom_right_vec[1], angles='xy', scale_units='xy', scale=1, color='purple', label='Bottom Right')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.legend()
    # plt.xlabel("PCA Dimension 1")
    # plt.ylabel("PCA Dimension 2")
    
    ax = fig.add_subplot(111 , projection='3d')
    ax.quiver(0, 0, 0, top_vec[0], top_vec[1], top_vec[2], color='r', label='Top', length=0.1)
    ax.quiver(0, 0, 0, bottom_vec[0], bottom_vec[1], bottom_vec[2], color='b', label='Bottom', length=0.1)
    ax.quiver(0, 0, 0, left_vec[0], left_vec[1], left_vec[2], color='g', label='Left', length=0.1)
    ax.quiver(0, 0, 0, right_vec[0], right_vec[1], right_vec[2], color='c', label='Right', length=0.1)
    ax.quiver(0, 0, 0, top_left_vec[0], top_left_vec[1], top_left_vec[2], color='m', label='Top Left', length=0.1)
    ax.quiver(0, 0, 0, top_right_vec[0], top_right_vec[1], top_right_vec[2], color='y', label='Top Right', length=0.1)
    ax.quiver(0, 0, 0, bottom_left_vec[0], bottom_left_vec[1], bottom_left_vec[2], color='orange', label='Bottom Left', length=0.1)
    ax.quiver(0, 0, 0, bottom_right_vec[0], bottom_right_vec[1], bottom_right_vec[2], color='purple', label='Bottom Right', length=0.1)
    plt.legend()
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.ylabel("PCA Dimension 3")
    
    save_path = os.path.join(exp_dir, f"pos_embed_pca_dir_vec.pdf")
    print("save_path", save_path)
    plt.savefig(save_path, bbox_inches='tight')

def explore_1d_pos_embed_direction_vectors_old1(model_name, device):
    
    # set save path
    exp_dir = root_dir / "figures"
    exp_dir = os.path.join(exp_dir, "explore_1d_pos_embed_direction_vectors", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
        elif "intern" in model_name:
            vit, llm = model.vision_model, model.language_model
        else:
            pass
    
    # Explore pos embed
    # get 1d positional embeddings
    if "llava" in model_name:
        pos_embed = vit.vision_model.embeddings.position_embedding.weight
        print(f"pos_embed shape: {pos_embed.shape}")  # [577, 1024]  577 = 1 + 24*24, 24 = 336//4
        pos_embed = pos_embed[1:]  # remove the first token
    elif "intern" in model_name:
        pos_embed = vit.embeddings.position_embedding
        print(f"pos_embed shape: {pos_embed.shape}")  # [1, 1025, 1024]  # 1025 = 1 + 32*32, 32 = 448//14
        pos_embed = pos_embed[0, 1:]  # remove the first token
    
    pos_embed = pos_embed.cpu().detach().float().numpy()
    if "llava" in model_name:
        width = 24
    elif "intern" in model_name:
        width = 32
    else:
        pass
    
    # get position points
    quarter_width = width // 4
    origin_corrd = [(x, y) for x in range(quarter_width, width - quarter_width) for y in range(quarter_width, width - quarter_width)]
    print(f"origin_corrd: {origin_corrd}")
    
    # get direction vectors
    top_vecs = []
    bottom_vecs = []
    left_vecs = []
    right_vecs = []
    top_left_vecs = []
    top_right_vecs = []
    bottom_left_vecs = []
    bottom_right_vecs = []
    for i in range(len(origin_corrd)):
        origin_x, origin_y = origin_corrd[i]
        top_pos_coord = (origin_x, origin_y - quarter_width)
        bottom_pos_coord = (origin_x, origin_y + quarter_width)
        left_pos_coord = (origin_x - quarter_width, origin_y)
        right_pos_coord = (origin_x + quarter_width, origin_y)
        # print(f"origin: {origin_corrd[i]}, top: {top_pos_coord}, bottom: {bottom_pos_coord}, left: {left_pos_coord}, right: {right_pos_coord}")
        # import pdb; pdb.set_trace()
        origin_pos_embed = pos_embed[origin_y * width + origin_x]
        top_pos_embed = pos_embed[top_pos_coord[1] * width + top_pos_coord[0]] - origin_pos_embed
        bottom_pos_embed = pos_embed[bottom_pos_coord[1] * width + bottom_pos_coord[0]] - origin_pos_embed
        left_pos_embed = pos_embed[left_pos_coord[1] * width + left_pos_coord[0]] - origin_pos_embed
        right_pos_embed = pos_embed[right_pos_coord[1] * width + right_pos_coord[0]] - origin_pos_embed
        
        top_vecs.append(top_pos_embed)
        bottom_vecs.append(bottom_pos_embed)
        left_vecs.append(left_pos_embed)
        right_vecs.append(right_pos_embed)
        top_left_vecs.append(top_pos_embed - left_pos_embed)
        top_right_vecs.append(top_pos_embed - right_pos_embed)
        bottom_left_vecs.append(bottom_pos_embed - left_pos_embed)
        bottom_right_vecs.append(bottom_pos_embed - right_pos_embed)
    
    # perform PCA to direction vectors (dim -> 3)
    pos_embed = np.array([top_vecs, bottom_vecs, left_vecs, right_vecs,
                          top_left_vecs, top_right_vecs, bottom_left_vecs, bottom_right_vecs])
    pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])  # (8 * len(origin_corrd), 1024)
    print(f"pos_embed shape after reshape: {pos_embed.shape}")  # (8 * len(origin_corrd), 1024)
    # pca = PCA(n_components=2)
    # pos_embed = pca.fit_transform(pos_embed)
    tsne = TSNE(n_components=2, random_state=41, perplexity=30, n_iter=1000, verbose=1)
    pos_embed = tsne.fit_transform(pos_embed)
    
    # get centroid of each direction
    top_vec = np.mean(pos_embed[:len(origin_corrd)], axis=0)
    bottom_vec = np.mean(pos_embed[len(origin_corrd):2*len(origin_corrd)], axis=0)
    left_vec = np.mean(pos_embed[2*len(origin_corrd):3*len(origin_corrd)], axis=0)
    right_vec = np.mean(pos_embed[3*len(origin_corrd):4*len(origin_corrd)], axis=0)
    top_left_vec = np.mean(pos_embed[4*len(origin_corrd):5*len(origin_corrd)], axis=0)
    top_right_vec = np.mean(pos_embed[5*len(origin_corrd):6*len(origin_corrd)], axis=0)
    bottom_left_vec = np.mean(pos_embed[6*len(origin_corrd):7*len(origin_corrd)], axis=0)
    bottom_right_vec = np.mean(pos_embed[7*len(origin_corrd):], axis=0)
    
    # plot
    # plot the data points in a 2D figure
    plt.figure()
    plt.scatter(pos_embed[:len(origin_corrd), 0], pos_embed[:len(origin_corrd), 1], color='r', label='Top', alpha=0.5)
    plt.scatter(pos_embed[len(origin_corrd):2*len(origin_corrd), 0], pos_embed[len(origin_corrd):2*len(origin_corrd), 1], color='b', label='Bottom', alpha=0.5)
    plt.scatter(pos_embed[2*len(origin_corrd):3*len(origin_corrd), 0], pos_embed[2*len(origin_corrd):3*len(origin_corrd), 1], color='g', label='Left', alpha=0.5)
    plt.scatter(pos_embed[3*len(origin_corrd):4*len(origin_corrd), 0], pos_embed[3*len(origin_corrd):4*len(origin_corrd), 1], color='c', label='Right', alpha=0.5)
    plt.scatter(pos_embed[4*len(origin_corrd):5*len(origin_corrd), 0], pos_embed[4*len(origin_corrd):5*len(origin_corrd), 1], color='m', label='Top Left', alpha=0.5)
    plt.scatter(pos_embed[5*len(origin_corrd):6*len(origin_corrd), 0], pos_embed[5*len(origin_corrd):6*len(origin_corrd), 1], color='y', label='Top Right', alpha=0.5)
    plt.scatter(pos_embed[6*len(origin_corrd):7*len(origin_corrd), 0], pos_embed[6*len(origin_corrd):7*len(origin_corrd), 1], color='orange', label='Bottom Left', alpha=0.5)
    plt.scatter(pos_embed[7*len(origin_corrd):, 0], pos_embed[7*len(origin_corrd):, 1], color='purple', label='Bottom Right', alpha=0.5)
    # plot the vector
    plt.quiver(0, 0, top_vec[0], top_vec[1], angles='xy', scale_units='xy', scale=1, color='r', label='Top')
    plt.quiver(0, 0, bottom_vec[0], bottom_vec[1], angles='xy', scale_units='xy', scale=1, color='b', label='Bottom')
    plt.quiver(0, 0, left_vec[0], left_vec[1], angles='xy', scale_units='xy', scale=1, color='g', label='Left')
    plt.quiver(0, 0, right_vec[0], right_vec[1], angles='xy', scale_units='xy', scale=1, color='c', label='Right')
    plt.quiver(0, 0, top_left_vec[0], top_left_vec[1], angles='xy', scale_units='xy', scale=1, color='m', label='Top Left')
    plt.quiver(0, 0, top_right_vec[0], top_right_vec[1], angles='xy', scale_units='xy', scale=1, color='y', label='Top Right')
    plt.quiver(0, 0, bottom_left_vec[0], bottom_left_vec[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Bottom Left')
    plt.quiver(0, 0, bottom_right_vec[0], bottom_right_vec[1], angles='xy', scale_units='xy', scale=1, color='purple', label='Bottom Right')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.legend()
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    
    save_path = os.path.join(exp_dir, f"pos_embed_pca_dir_vec.pdf")
    print("save_path", save_path)
    plt.savefig(save_path, bbox_inches='tight')

def explore_1d_pos_embed_direction_vectors(model_name, device):
    
    # set save path
    exp_dir = root_dir / "figures"
    exp_dir = os.path.join(exp_dir, "explore_1d_pos_embed_direction_vectors", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
        elif "intern" in model_name:
            vit, llm = model.vision_model, model.language_model
        else:
            pass
    
    # Explore pos embed
    # get 1d positional embeddings
    if "llava" in model_name:
        pos_embed = vit.vision_model.embeddings.position_embedding.weight
        print(f"pos_embed shape: {pos_embed.shape}")  # [577, 1024]  577 = 1 + 24*24, 24 = 336//4
        pos_embed = pos_embed[1:]  # remove the first token
    elif "intern" in model_name:
        pos_embed = vit.embeddings.position_embedding
        print(f"pos_embed shape: {pos_embed.shape}")  # [1, 1025, 1024]  # 1025 = 1 + 32*32, 32 = 448//14
        pos_embed = pos_embed[0, 1:]  # remove the first token
    
    pos_embed = pos_embed.cpu().detach().float().numpy()
    if "llava" in model_name:
        width = 24
    elif "intern" in model_name:
        width = 32
    else:
        pass
    
    # get position points
    quarter_width = width // 4
    origin_corrd = [(x, y) for x in range(quarter_width, width - quarter_width) for y in range(quarter_width, width - quarter_width)]
    print(f"origin_corrd: {origin_corrd}")
    
    # get direction vectors
    top_vecs = []
    bottom_vecs = []
    left_vecs = []
    right_vecs = []
    for i in range(len(origin_corrd)):
        origin_x, origin_y = origin_corrd[i]
        top_pos_coord = (origin_x, origin_y - quarter_width)
        bottom_pos_coord = (origin_x, origin_y + quarter_width)
        left_pos_coord = (origin_x - quarter_width, origin_y)
        right_pos_coord = (origin_x + quarter_width, origin_y)
        # print(f"origin: {origin_corrd[i]}, top: {top_pos_coord}, bottom: {bottom_pos_coord}, left: {left_pos_coord}, right: {right_pos_coord}")
        # import pdb; pdb.set_trace()
        origin_pos_embed = pos_embed[origin_y * width + origin_x]
        top_pos_embed = pos_embed[top_pos_coord[1] * width + top_pos_coord[0]] - origin_pos_embed
        bottom_pos_embed = pos_embed[bottom_pos_coord[1] * width + bottom_pos_coord[0]] - origin_pos_embed
        left_pos_embed = pos_embed[left_pos_coord[1] * width + left_pos_coord[0]] - origin_pos_embed
        right_pos_embed = pos_embed[right_pos_coord[1] * width + right_pos_coord[0]] - origin_pos_embed
        
        top_vecs.append(top_pos_embed)
        bottom_vecs.append(bottom_pos_embed)
        left_vecs.append(left_pos_embed)
        right_vecs.append(right_pos_embed)
    
    # perform PCA to direction vectors (dim -> 3)
    pos_embed = np.array([top_vecs, bottom_vecs, left_vecs, right_vecs])
    pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])  # (8 * len(origin_corrd), 1024)
    print(f"pos_embed shape after reshape: {pos_embed.shape}")  # (8 * len(origin_corrd), 1024)
    # pca = PCA(n_components=2)
    # pos_embed = pca.fit_transform(pos_embed)
    tsne = TSNE(n_components=2, random_state=41, perplexity=30, n_iter=1000, verbose=1)
    pos_embed = tsne.fit_transform(pos_embed)
    
    # get centroid of each direction
    top_vec = np.mean(pos_embed[:len(origin_corrd)], axis=0)
    bottom_vec = np.mean(pos_embed[len(origin_corrd):2*len(origin_corrd)], axis=0)
    left_vec = np.mean(pos_embed[2*len(origin_corrd):3*len(origin_corrd)], axis=0)
    right_vec = np.mean(pos_embed[3*len(origin_corrd):4*len(origin_corrd)], axis=0)
    
    # plot
    # plot the data points in a 2D figure
    plt.figure()
    colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
    plt.scatter(pos_embed[:len(origin_corrd), 0], pos_embed[:len(origin_corrd), 1], color=colors[0], label='Top')
    plt.scatter(pos_embed[len(origin_corrd):2*len(origin_corrd), 0], pos_embed[len(origin_corrd):2*len(origin_corrd), 1], color=colors[1], label='Bottom')
    plt.scatter(pos_embed[2*len(origin_corrd):3*len(origin_corrd), 0], pos_embed[2*len(origin_corrd):3*len(origin_corrd), 1], color=colors[2], label='Left')
    plt.scatter(pos_embed[3*len(origin_corrd):4*len(origin_corrd), 0], pos_embed[3*len(origin_corrd):4*len(origin_corrd), 1], color=colors[3], label='Right')
    # plot the vector
    plt.quiver(0, 0, top_vec[0], top_vec[1], angles='xy', scale_units='xy', scale=1, color=colors[0], label='Top')
    plt.quiver(0, 0, bottom_vec[0], bottom_vec[1], angles='xy', scale_units='xy', scale=1, color=colors[1], label='Bottom')
    plt.quiver(0, 0, left_vec[0], left_vec[1], angles='xy', scale_units='xy', scale=1, color=colors[2], label='Left')
    plt.quiver(0, 0, right_vec[0], right_vec[1], angles='xy', scale_units='xy', scale=1, color=colors[3], label='Right')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.legend()
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    
    save_path = os.path.join(exp_dir, f"pos_embed_pca_dir_vec.pdf")
    print("save_path", save_path)
    plt.savefig(save_path, bbox_inches='tight')

def explore_1d_pos_embed_decay(model_name, dataset_name, device, batch_size=8):
    # set save path
    exp_dir = root_dir / "figures"
    exp_dir = os.path.join(exp_dir, "explore_1d_pos_embed_visual_dacay", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        model.eval()
        model_dir = MODEL_NAME_TO_PATH[model_name]
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
            vision_process_func = None
            vit_layer_num = model.config.vision_config.num_hidden_layers
        elif "qwen" in model_name:
            vit, llm = model.visual, model.model
            vision_process_func = process_vision_info_qwen
            vit_layer_num = model.config.vision_config.depth
        elif "intern" in model_name:
            vit, llm = model.vision_model, model.language_model
            vision_process_func = load_image_intern
            vit_layer_num = model.config.vision_config.num_hidden_layers
        else:
            pass
    
    # load data
    dataloader = load_data(dataset_name, model_name, processor, data_num=None, random=False, batch_size=batch_size)
    
    # 1. get 1d positional embeddings
    if "llava" in model_name:
        pos_embed = vit.vision_model.embeddings.position_embedding.weight
        print(f"pos_embed shape: {pos_embed.shape}")  # [577, 1024]  577 = 1 + 24*24, 24 = 336//4
        pos_embed = pos_embed[1:]  # remove the first token
    elif "intern" in model_name:
        pos_embed = vit.embeddings.position_embedding
        print(f"pos_embed shape: {pos_embed.shape}")  # [1, 1025, 1024]  # 1025 = 1 + 32*32, 32 = 448//14
        pos_embed = pos_embed[0, 1:]  # remove the first token
    else:
        pass
    
    # define weight(the influence of a vector on the other)
    def compute_weight_batch(vector_batch_a, vector_batch_b):
        """
        Compute the influence of vector a on vector b for each sample in a batch: a * b / ||b||
        vector_batch_a: (len, dim)
        vector_batch_b: (bsz, len, dim)
        return: (bsz)  
        """
        batch_size = vector_batch_b.shape[0]
        vector_batch_a = vector_batch_a.expand(batch_size, -1, -1)  # (bsz, len, dim)
        vector_batch_a = vector_batch_a.reshape(-1, vector_batch_a.shape[-1])  # (bsz*len, dim)
        vector_batch_b = vector_batch_b.reshape(-1, vector_batch_b.shape[-1])
        ret = torch.sum(vector_batch_a * vector_batch_b, dim=1) / torch.norm(vector_batch_b, dim=-1)
        ret = ret.cpu().detach()
        return ret
    
    all_weights = [[] for _ in range(vit_layer_num + 1)]
    for idx, batch in enumerate(tqdm(dataloader)):
        inputs, samples = batch
        if "llava" in model_name:
            inputs = inputs.to(device)
        elif "intern" in model_name:
            for key in inputs.keys():
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)
        else:
            pass
    
        # forward and get llm outputs of each layer
        with torch.no_grad():
            if "llava" in model_name:
                vision_outputs = vit(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True
                )  # (bsz, 577, dim)
            elif "intern" in model_name:
               vision_outputs = vit(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True
                )  # (n_block_image, 1025, dim)
            else:
                pass
        all_hidden_states = vision_outputs.hidden_states  # n_layers *(bsz, len, dim)
        # print(f"num_hidden_states: {len(all_hidden_states)}")  # 25
        # print(f"hidden_states shape: {all_hidden_states[0].shape}")  # [8, 577, 1024]
        
        # (embedding layer)
        layer_weights = compute_weight_batch(pos_embed, all_hidden_states[0][:, 1:, :])
        all_weights[0].append(torch.mean(layer_weights, dim=0, keepdim=True))
        for layer_id in range(vit_layer_num):
            layer_hidden_states = all_hidden_states[layer_id + 1][:, 1:, :]
            layer_weights = compute_weight_batch(pos_embed, layer_hidden_states)
            all_weights[layer_id + 1].append(torch.mean(layer_weights, dim=0, keepdim=True))
    
    # normalize
    for layer_id in range(vit_layer_num + 1):
        all_weights[layer_id] = torch.cat(all_weights[layer_id], dim=0)
    all_weights = torch.stack(all_weights, dim=0)  # (n_layers, num_samples*len)
    all_weights = all_weights.reshape(all_weights.shape[0] * all_weights.shape[1])
    all_weights = (all_weights - all_weights.min()) / (all_weights.max() - all_weights.min())
    all_weights = all_weights.reshape(vit_layer_num + 1, -1)  # (n_layers, num_samples*len)
    all_weights = all_weights.float().numpy()
    
    # plot
    layer_ids = list(range(vit_layer_num + 1))
    mean = np.mean(all_weights, axis=1)
    std_error = np.std(all_weights, axis=1) / np.sqrt(len(all_weights[0]))
    print(np.std(all_weights, axis=1))
    print(np.sqrt(len(all_weights[0])))
    print(std_error)
    plt.figure()
    plt.plot(
        layer_ids,
        mean,
        color="blue",
        alpha=1,
        marker='o',
        markersize=5, 
        markeredgecolor='blue',
        markeredgewidth=1.5
    )
    plt.fill_between(
        layer_ids, 
        mean - std_error, 
        mean + std_error, 
        color="blue", 
        alpha=0.1, 
    )

    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel(f"Relative positional embedding weight", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    # plt.legend()
    save_path = os.path.join(exp_dir, f"pos_embed_decay.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(save_path)

def erase_object_in_llm(model_name, dataset_name, device, replaced_objects=["nucleus"], new_objects=["background"], erase_thumbnail=False):
    """
    When performing spatial reasoning, replace one object with another object and see if the model's performance changes.
    Args:
        
    """
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    labels = ["left", "right", "behind", "in front of"]
    
    # set save path
    save_dir = root_dir / "eval/WhatsUp/results"
    if erase_thumbnail:
        exp_dir = os.path.join(save_dir, "test_vit_erase_one_object", f"{model_name}_erase_thumbnail")
    else:
        exp_dir = os.path.join(save_dir, "test_vit_erase_one_object", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, f"erase_{'_'.join(replaced_objects)}_with_{'_'.join(new_objects)}.jsonl")
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        model_dir = MODEL_NAME_TO_PATH[model_name]
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        
        vit, llm = None, None
        if "llava" in model_name:
            vit, llm = model.vision_tower, model.language_model
            vision_process_func = None
            replace_llava1_5_receive_vit_output()
        elif "qwen" in model_name:
            vit, llm = model.visual, model.model
            vision_process_func = process_vision_info_qwen
            replace_qwen2_5_vl_test_directions_processor_return_indices()
            replace_qwen2_5_vl_receive_vit_output()
        elif "intern" in model_name:
            vision_process_func = load_image_intern_return_wh
            # to use intern's forward, we need to perform some pre work in batch_chat() before the `generate` method and the `forward` method, which has been moved to get_intern_inputs_embed, and the vit output is also passed via this method. 
        else:
            pass
    
    # load data
    if True:
        whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
        whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
        dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
        image_files = os.listdir(dir_name)
        image_names = [file_name.split('.')[0] for file_name in image_files]
        image_pairs = []
        for i, img_name in enumerate(image_names):
            obj_satellite = img_name.split('_')[0]
            obj_nucleus = img_name.split('_')[-1]
            image_pairs.append((obj_satellite, obj_nucleus))
        image_pairs = list(set(image_pairs))
        
        bboxes = {}
        bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
        with jsonlines.open(bboxes_path, "r") as f:
            for sample in f:
                bboxes[sample["image"]] = sample["bbox"]

    # prepare direction probs
    if True:
        token_ids = {}
        if "llava" in model_name:
            token_ids = {"in": ["in", "In", "▁In"], "behind": ["behind", "Behind", "▁Be"], "left": ["left", "Left", "▁Left"], "right": ["right", "Right", "▁Right"]}
        elif "qwen" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["Ġbehind", "Behind"], "left": ["left", "Left"], "right": ["right", "Right"]}
        elif "intern" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["beh", "Beh"], "left": ["left", "Left"], "right": ["right", "Right"]}
        else:
            pass
        
        for key, value in token_ids.items():
            token_ids[key] = [tokenizer.convert_tokens_to_ids(token) for token in value]
            
        direction_ids = {
            "in front of": token_ids["in"],
            "behind": token_ids["behind"],
            "left": token_ids["left"],
            "right": token_ids["right"]
        }
        
    # forward
    probs_normal = []
    probs_erased = []
    probs_normal_dirs = {relation: [] for relation in relations}
    probs_erased_dirs = {relation: [] for relation in relations}
    for pair_id, image_pair in enumerate(image_pairs):
        probs_normal_sample = []
        probs_erased_sample = []
        obj_satellite, obj_nucleus = image_pair
            
        # preparation
        if True:
            # get four images for a pair
            sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
            sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
            sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
            
            # get inputs
            instruction = f"Is the {obj_satellite} in front of/behind/to the left of/to the right of the {obj_nucleus}? Please choose the best answer from the four options: [In front of, Behind, Left, Right], and reply with only one word. \nYour answer is:"
            model_inputs = get_model_inputs(
                model_name=model_name,
                processor=processor,
                vision_process_func=vision_process_func,
                image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
                prompts=[instruction for _ in range(4)]
            )
            if "llava" in model_name:
                inputs = model_inputs
            elif "qwen" in model_name:
                inputs, image_indices = model_inputs
            elif "intern" in model_name:
                inputs = model_inputs
                image_indices = []
                st = 0
                n_patches_per_tile = int(model_config.vision_config.image_size // (model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)) ) ** 2
                for idx in range(len(inputs["num_patches_list"])):
                    image_indices.append(slice(st, st + inputs["num_patches_list"][idx] * n_patches_per_tile))
                    st += inputs["num_patches_list"][idx] * n_patches_per_tile
                    
            if "intern" in model_name:
                for key in inputs.keys():
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(device)
            else:
                inputs.to(device)

            # get resized bboxes
            sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
            resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
            if erase_thumbnail:
                resized_bboxes_thumbnail = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                # draw bboxes
                # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                
                # 1. get original sizes
                image = Image.open(image_path).convert('RGB')
                original_width, original_height = image.size
                image.close()
                # 2. get current size (after resize)
                if "llava" in model_name:
                    height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, height, width = image_grid_thw[idx].cpu().numpy()
                    height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                    width *= model.config.vision_config.spatial_patch_size
                elif "intern" in model_name:
                    block_image_width, block_image_height = inputs["block_wh_list"][idx]
                    height = block_image_height * model_config.vision_config.image_size
                    width = block_image_width * model_config.vision_config.image_size
                    height_thumbnail, width_thumbnail = model_config.vision_config.image_size, model_config.vision_config.image_size
                # 3. get resized bboxes
                scale = width / original_width
                # print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale}")
                # import pdb; pdb.set_trace()
                x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                if erase_thumbnail:
                    scale_w_thumbnail = width_thumbnail / original_width
                    scale_h_thumbnail = height_thumbnail / original_height
                    x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                    x1, y1, x2, y2 = int(x1 * scale_w_thumbnail), int(y1 * scale_h_thumbnail), int(x2 * scale_w_thumbnail), int(y2 * scale_h_thumbnail)
                    resized_bboxes_thumbnail[idx]["satellite"] = (x1, y1, x2, y2)
                    x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                    x1, y1, x2, y2 = int(x1 * scale_w_thumbnail), int(y1 * scale_h_thumbnail), int(x2 * scale_w_thumbnail), int(y2 * scale_h_thumbnail)
                    resized_bboxes_thumbnail[idx]["nucleus"] = (x1, y1, x2, y2)
                    
                
            # get object patches
            object_patch_ids = [{"satellite": None, "nucleus": None, "background": None} for _ in range(4)]
            if erase_thumbnail:
                object_patch_ids_thumbnail = [{"satellite": None, "nucleus": None, "background": None} for _ in range(4)]
            if "llava" in model_name:
                patch_size = model_config.vision_config.patch_size
            elif "qwen" in model_name:
                patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
            elif "intern" in model_name:
                patch_size = model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                            # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids[idx][obj_name] = obj_patch_ids
                # get background patch ids
                object_patch_ids[idx]["background"] = list(set(range(int(width // patch_size) * int(height // patch_size))) - set(object_patch_ids[idx]["satellite"]) - set(object_patch_ids[idx]["nucleus"]))
            if erase_thumbnail:
                for idx, image_path in enumerate(sample_image_paths):
                    for obj_name, obj_bbox in resized_bboxes_thumbnail[idx].items():
                        obj_patch_ids = []
                        x1, y1, x2, y2 = obj_bbox
                        for i in range(int(model_config.vision_config.image_size // patch_size)):  # 448 // (14*2) = 16
                            for j in range(int(model_config.vision_config.image_size // patch_size)):
                                x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                                if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                                    obj_patch_ids.append(j * int(model_config.vision_config.image_size // patch_size) + i)
                        object_patch_ids_thumbnail[idx][obj_name] = obj_patch_ids
                    # get background patch ids
                    object_patch_ids_thumbnail[idx]["background"] = list(set(range(int(model_config.vision_config.image_size // patch_size) * int(model_config.vision_config.image_size // patch_size))) - set(object_patch_ids_thumbnail[idx]["satellite"]) - set(object_patch_ids_thumbnail[idx]["nucleus"]))
        

        # first get the normal output
        if True:
            with torch.no_grad():
                if "intern" in model_name:
                    input_embeds, attention_mask = get_intern_inputs_embeds(model, tokenizer, model_inputs)
                    outputs = model.language_model(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        return_dict=True, 
                        output_hidden_states=True
                    )
                else:
                    outputs = model(
                        **inputs, 
                        return_dict=True, 
                        output_hidden_states=True
                    )
            logits = outputs.logits[:, -1, :]  # (bsz, vocab_size)
            for idx in range(4):
                correct_label = labels[idx]
                correct_token_ids = direction_ids[correct_label]
                probs = F.softmax(logits[idx], dim=-1)  # (vocab_size)
                probs_correct = torch.sum(probs[correct_token_ids], dim=-1).tolist()
                probs_normal_sample.append(probs_correct)
        
        # then intervene with the opposite direction
        with torch.no_grad():
            # forward ViT (get ViT outputs after connector)
            image_reprs = None  # llava: (bsz, len(24*24), dim)  qwen2.5-vl: (len, dim)
            if "llava" in model_name:
                image_reprs = model.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    vision_feature_layer=model.config.vision_feature_layer,
                    vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                )
            elif "qwen" in model_name:
                inputs["pixel_values"] = inputs["pixel_values"].type(vit.dtype)
                image_reprs = vit(
                    inputs["pixel_values"], 
                    grid_thw=inputs["image_grid_thw"], 
                )
            elif "intern" in model_name:
                image_reprs = model.extract_feature(
                    pixel_values=inputs["pixel_values"],
                )  # (n_tiles, num_patches (24*24), dim)
                # print(f"image_reprs shape: {image_reprs.shape}")  # [52, 256, 4096] = (n_tiles, num_patches (16*16), dim)
                intern_image_len = image_reprs.shape[1]  # 256
                num_tiles_per_image = image_reprs.shape[0] // 4 # 4 images, each image has num_tiles_per_image tiles
                image_reprs = image_reprs.reshape(-1, image_reprs.shape[-1])  # (n_tiles * num_patches (16*16), dim)
            else:
                pass
            
            # fix thumbnail patch ids
            if erase_thumbnail:
                thumbnail_st = int(intern_image_len * (num_tiles_per_image - 1))
                for idx in range(4):
                    for key in object_patch_ids_thumbnail[idx].keys():
                        object_patch_ids_thumbnail[idx][key] = [id_ + thumbnail_st for id_ in object_patch_ids_thumbnail[idx][key]]
            
            # get the relation representation
            for idx, image_path in enumerate(sample_image_paths):
                if "llava" in model_name:
                    img_repr = image_reprs[idx]
                elif "qwen" in model_name:
                    img_repr = image_reprs[image_indices[idx]]
                elif "intern" in model_name:
                    img_repr = image_reprs[image_indices[idx]]
                    # print("img_repr", img_repr.shape)  # (num_patches, dim)
                    # import pdb; pdb.set_trace()
                else:
                    pass
                satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].mean(dim=0).unsqueeze(0)  # (1, dim)
                nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].mean(dim=0).unsqueeze(0)
                background_repr = img_repr[object_patch_ids[idx]["background"], :].mean(dim=0).unsqueeze(0)
                object_reprs = {"satellite": satellite_repr, "nucleus": nucleus_repr, "background": background_repr}
                if erase_thumbnail:
                    # print(object_patch_ids_thumbnail[idx])
                    # print("image_indices[idx]", image_indices[idx])
                    # import pdb; pdb.set_trace()
                    satellite_repr_thumbnail = img_repr[object_patch_ids_thumbnail[idx]["satellite"], :].mean(dim=0).unsqueeze(0)
                    nucleus_repr_thumbnail = img_repr[object_patch_ids_thumbnail[idx]["nucleus"], :].mean(dim=0).unsqueeze(0)
                    background_repr_thumbnail = img_repr[object_patch_ids_thumbnail[idx]["background"], :].mean(dim=0).unsqueeze(0)
                    object_reprs_thumbnail = {"satellite": satellite_repr_thumbnail, "nucleus": nucleus_repr_thumbnail, "background": background_repr_thumbnail}

                # erase
                if "llava" in model_name:
                    intervened_image_repr = image_reprs[idx]
                elif "qwen" in model_name:
                    intervened_image_repr = image_reprs[image_indices[idx]]
                elif "intern" in model_name:
                    intervened_image_repr = image_reprs[image_indices[idx]]
                else:
                    pass
                for replaced_object, new_object in zip(replaced_objects, new_objects):
                    replaced_object_ids = object_patch_ids[idx][replaced_object]
                    intervened_image_repr[replaced_object_ids, :] = object_reprs[new_object].expand(len(replaced_object_ids), -1)
                if "intern" in model_name:
                    # replace the object in thumbnail reprs to the background reprs
                    if erase_thumbnail:
                        for replaced_object, new_object in zip(replaced_objects, new_objects):
                            replaced_object_ids = object_patch_ids_thumbnail[idx][replaced_object]
                            intervened_image_repr[replaced_object_ids, :] = object_reprs_thumbnail[new_object].expand(len(replaced_object_ids), -1)
                    else:  # replace the whole thumbnail with background
                        thumbnail_len = intern_image_len
                        intervened_image_repr[-thumbnail_len:, :] = object_reprs["background"].expand(thumbnail_len, -1)
                    
                # forward LLM
                single_inputs = copy.deepcopy(inputs)
                for key in single_inputs.keys():
                    if isinstance(single_inputs[key], torch.Tensor):
                        single_inputs[key] = single_inputs[key][idx].unsqueeze(0)
                    else:
                        single_inputs[key] = [single_inputs[key][idx]]
                    
                if "llava" in model_name:
                    outputs = model(
                        **single_inputs,
                        image_features=intervened_image_repr,
                        return_dict=True, 
                        output_hidden_states=True
                    )
                elif "qwen" in model_name:
                    outputs = model(
                        **single_inputs,
                        image_embeds=intervened_image_repr,
                        return_dict=True, 
                        output_hidden_states=True
                    )
                elif "intern" in model_name:
                    # print(intervened_image_repr.shape)  # [3328, 4096] = [13*256, 4096]
                    intervened_image_repr = intervened_image_repr.reshape(-1, intern_image_len, image_reprs.shape[-1])
                    input_embeds, attention_mask = get_intern_inputs_embeds(
                        model, 
                        tokenizer, 
                        single_inputs,
                        vit_embeds=intervened_image_repr,
                    )
                    outputs = model.language_model(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        return_dict=True, 
                        output_hidden_states=True
                    )

                logits = outputs.logits[0, -1, :]  # (vocab_size)
                correct_label = labels[idx]
                correct_token_ids = direction_ids[correct_label]
                probs = F.softmax(logits, dim=-1)  # (vocab_size)
                probs_correct = torch.sum(probs[correct_token_ids], dim=-1).tolist()
                probs_erased_sample.append(probs_correct)
                
        probs_normal.extend(probs_normal_sample)
        probs_erased.extend(probs_erased_sample)
        for idx, rel in enumerate(relations):
            probs_normal_dirs[rel].append(probs_normal_sample[idx])
            probs_erased_dirs[rel].append(probs_erased_sample[idx])  
        print(f"pair_id: {pair_id}, probs_normal: {probs_normal_sample}, probs_erased: {probs_erased_sample}")

    # compute avg
    print("-" * 20)
    probs_normal = np.mean(probs_normal, axis=0)
    probs_erased = np.mean(probs_erased, axis=0)
    probs_normal_dirs = {k: float(np.mean(v)) for k, v in probs_normal_dirs.items()}
    probs_erased_dirs = {k: float(np.mean(v)) for k, v in probs_erased_dirs.items()}
    with jsonlines.open(save_path, "w") as f:
        data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "replaced_object": replaced_object,
            "new_object": new_object,
            "probs_normal": probs_normal,
            "probs_erased": probs_erased,
            "probs_normal_dirs": probs_normal_dirs,
            "probs_erased_dirs": probs_erased_dirs
        }
        f.write(data)
    print(f"probs_normal: {probs_normal}, probs_erased: {probs_erased}")
    print(f"probs_normal_dirs: {probs_normal_dirs}, probs_erased_dirs: {probs_erased_dirs}")

def check_relation_pair_similarity(model_name, dataset_name, device, before_connector=False):
    # load model
    model, processor, tokenizer = load_model(model_name, device)
    model_dir = MODEL_NAME_TO_PATH[model_name]
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    
    vit, llm = None, None
    if "qwen" in model_name:
        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
        replace_qwen2_5_vl_test_directions_processor_return_indices()
        if before_connector:
            replace_qwen2_5_vl_test_directions_vit_return_hidden_states()
    elif "intern" in model_name:
        vision_process_func = load_image_intern_return_wh
    else:
        vit, llm = model.vision_tower, model.language_model
        vision_process_func = None
    
    # load batch of images
    whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
    whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
    dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
    image_files = os.listdir(dir_name)
    image_names = [file_name.split('.')[0] for file_name in image_files]
    image_pairs = []
    for i, img_name in enumerate(image_names):
        obj_satellite = img_name.split('_')[0]
        obj_nucleus = img_name.split('_')[-1]
        image_pairs.append((obj_satellite, obj_nucleus))
    image_pairs = list(set(image_pairs))
    
    bboxes = {}
    bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
    with jsonlines.open(bboxes_path, "r") as f:
        for sample in f:
            bboxes[sample["image"]] = sample["bbox"]

    # forward
    relations = ["left_of", "right_of", "behind", "in-front_of"]
    sims_cls = []
    sims_satellite = []
    sims_nucleus = []
    for pair_id, image_pair in enumerate(image_pairs):
        obj_satellite, obj_nucleus = image_pair
        
        # get four images for a pair
        sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
        sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
        sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
        
        # get inputs
        model_inputs = get_model_inputs(
            model_name=model_name,
            processor=processor,
            vision_process_func=vision_process_func,
            image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
            prompts=["describe the image and tell me what is the main object in the image" for _ in range(4)]
        )
        if "llava" in model_name:
            inputs = model_inputs
        if "qwen" in model_name:
            inputs, image_indices = model_inputs
        elif "intern" in model_name:
            inputs = model_inputs
            image_indices = []
            st = 0
            n_patches_per_tile = int(model_config.vision_config.image_size // (model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)) ) ** 2
            for idx in range(len(inputs["num_patches_list"])):
                image_indices.append(slice(st, st + inputs["num_patches_list"][idx] * n_patches_per_tile))
                st += inputs["num_patches_list"][idx] * n_patches_per_tile
            # print("n_patches_list", inputs["num_patches_list"])
            # print("image_indices", image_indices)
        
        # get resized bboxes
        sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
        resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
        for idx, image_path in enumerate(sample_image_paths):
            # draw bboxes
            # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
            
            # 1. get original sizes
            image = Image.open(image_path).convert('RGB')
            original_width, original_height = image.size
            image.close()
            # 2. get current size (after resize)
            if "llava" in model_name:
                height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
            elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                _, height, width = image_grid_thw[idx].cpu().numpy()
                height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                width *= model.config.vision_config.spatial_patch_size
            elif "intern" in model_name:
                block_image_width, block_image_height = inputs["block_wh_list"][idx]
                height = block_image_height * model_config.vision_config.image_size
                width = block_image_width * model_config.vision_config.image_size
            # 3. get resized bboxes
            scale_width = width / original_width
            scale_height = height / original_height
            print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale_width}, {scale_height}")
            # import pdb; pdb.set_trace()
            x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
            resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
            x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
            x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
            resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
            
        # get object patches (patch ids after connector)
        object_patch_ids = [{"satellite": None, "nucleus": None} for _ in range(4)]
        if "llava" in model_name:
            patch_size = model_config.vision_config.patch_size
        elif "qwen" in model_name:
            patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
        elif "intern" in model_name:
            patch_size = model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)
        for idx, image_path in enumerate(sample_image_paths):
            for obj_name, obj_bbox in resized_bboxes[idx].items():
                obj_patch_ids = []
                x1, y1, x2, y2 = obj_bbox
                for i in range(int(width // patch_size)):
                    for j in range(int(height // patch_size)):
                        x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                        if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                        # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                            obj_patch_ids.append(j * int(width // patch_size) + i)
                object_patch_ids[idx][obj_name] = obj_patch_ids
        if before_connector:
            # get object patches (patch ids before connector)
            if "qwen" in model_name:
                patch_size = model.config.vision_config.spatial_patch_size
            elif "intern" in model_name:
                patch_size = model_config.vision_config.patch_size
            object_patch_ids_before_connector = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids_before_connector[idx][obj_name] = obj_patch_ids
        
        # forward as a batch     
        # get vit output (before llm or before connector)
        image_reprs = None  # llava: (bsz, len(24*24), dim)  qwen2.5-vl: (len, dim)
        with torch.no_grad():
            if "intern" in model_name:
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(device)
            else:
                inputs.to(device)
                
            if before_connector:
                if "llava" in model_name:
                    image_reprs = vit(
                        pixel_values=inputs["pixel_values"],
                        output_hidden_states=True,
                    ).hidden_states  # (1+24)*(batch_size, seq_len, dim)
                    selected_layer = model.config.vision_feature_layer  # -2
                    image_reprs = image_reprs[selected_layer]  # (bsz, 1 + len(24*24), dim)
                    
                elif "qwen" in model_name:
                    image_reprs_after_connector, image_reprs = vit(
                        hidden_states=inputs["pixel_values"],
                        grid_thw=inputs["image_grid_thw"], 
                    )  # image_reprs: (bsz, len, dim)
                    # print("image_reprs_after_connector", image_reprs_after_connector.shape)  # (all_len, dim) [4920, 3584]
                    # print("image_reprs", image_reprs.shape)  # (len, dim) [19680, 1280]
                    # import pdb; pdb.set_trace()
                elif "intern" in model_name:
                    image_reprs = vit(
                        pixel_values=inputs["pixel_values"],
                        output_hidden_states=True,
                        return_dict=True
                    ).hidden_states
                    selected_layer = model.config.vision_feature_layer  # -2
                    image_reprs = image_reprs[selected_layer]  # (bsz, 1 + len(24*24), dim)
                else:
                    pass
            else:
                if "llava" in model_name:
                    image_reprs = model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        vision_feature_layer=model.config.vision_feature_layer,
                        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    )  # (bsz, len(24*24), dim), the [cls] token is removed
                elif "qwen" in model_name:
                    inputs["pixel_values"] = inputs["pixel_values"].type(vit.dtype)
                    image_reprs = vit(
                        inputs["pixel_values"], 
                        grid_thw=inputs["image_grid_thw"], 
                    )
                    # original_image size: [1280, 960], image_size: [1148, 840]
                    # print("pixel_values", inputs["pixel_values"].shape)  # [19680, 1176]
                    # print("image_reprs", image_reprs.shape)  # [4920, 3584]  4920 = 4 * (1148 / 28) * (840 / 28)
                    # import pdb; pdb.set_trace()
                elif "intern" in model_name:
                    image_reprs = model.extract_feature(
                        pixel_values=inputs["pixel_values"],
                    )
                    image_reprs = image_reprs.reshape(-1, image_reprs.shape[-1])  # (n_tiles * num_patches, dim)
                else:
                    pass
        
        # compute the similarity between [cls] tokens
        if before_connector:
            sample_sim_cls = []
            if "qwen" in model_name:
                pass  # qwen2 series of models does not have [cls] token
            else:
                image_reprs_cls = torch.nn.functional.normalize(image_reprs[:, 0, :], dim=-1)  # (bsz, dim)
                sim_matrix = torch.matmul(image_reprs_cls, image_reprs_cls.T)  # (bsz, bsz)
                sim_matrix = sim_matrix.float().cpu().numpy()
                print(f"pair_id: {pair_id}, sim_matrix:\n {sim_matrix}")
                for i in range(4):
                    for j in range(4):
                        if i < j:
                            sample_sim_cls.append(sim_matrix[i][j].item())
                sims_cls.append(np.mean(sample_sim_cls))
            
        # compute the similarity between the objects in four images
        object_reprs = {"satellite": [], "nucleus": []}
        for idx, image_path in enumerate(sample_image_paths):
            # 1. get the pooled representation for each object
            if before_connector:
                if "llava" in model_name:
                    img_repr = image_reprs[idx][1:]  # remove the [cls] token
                    satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].mean(dim=0).unsqueeze(0)
                    nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].mean(dim=0).unsqueeze(0)
                elif "qwen" in model_name:
                    indice = image_indices[idx]
                    merger_size = model.config.vision_config.spatial_merge_size
                    new_slice = slice(indice.start * merger_size ** 2, indice.stop * merger_size ** 2)
                    img_repr = image_reprs[new_slice]
                    # print("len_img_repr", img_repr.shape)  # (len, dim)
                    # print("satellite:", object_patch_ids_before_connector[idx]["satellite"])
                    # print("nucleus:", object_patch_ids_before_connector[idx]["nucleus"])
                    # import pdb; pdb.set_trace()
                    satellite_repr = img_repr[object_patch_ids_before_connector[idx]["satellite"], :].mean(dim=0).unsqueeze(0)
                    nucleus_repr = img_repr[object_patch_ids_before_connector[idx]["nucleus"], :].mean(dim=0).unsqueeze(0)
                elif "intern" in model_name:
                    img_repr = image_reprs[image_indices[idx]][1:]
                else:
                    pass
            else:
                if "llava" in model_name:
                    img_repr = image_reprs[idx]
                    satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].mean(dim=0).unsqueeze(0)
                    nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].mean(dim=0).unsqueeze(0)
                elif "qwen" in model_name:
                    img_repr = image_reprs[image_indices[idx]]
                    satellite_repr = img_repr[object_patch_ids[idx]["satellite"], :].mean(dim=0).unsqueeze(0)
                    nucleus_repr = img_repr[object_patch_ids[idx]["nucleus"], :].mean(dim=0).unsqueeze(0)
                elif "intern" in model_name:
                    img_repr = image_reprs[image_indices[idx]]
                else:
                    pass
            object_reprs["satellite"].append(satellite_repr)
            object_reprs["nucleus"].append(nucleus_repr)
        # print(object_reprs["satellite"][0])
        # print(object_reprs["satellite"][1])
        # print(object_reprs["satellite"][2])
        # print(object_reprs["satellite"][3])
        
        sample_sim_satellite = []
        sample_sim_nucleus = []
        satellite_reprs = torch.cat(object_reprs["satellite"], dim=0)  # (4, dim)
        satellite_reprs = torch.nn.functional.normalize(satellite_reprs, dim=-1)  # (4, dim)
        print(satellite_reprs)
        satellite_sim_matrix = torch.matmul(satellite_reprs, satellite_reprs.T)  # (4, 4)
        satellite_sim_matrix = satellite_sim_matrix.float().cpu().numpy()
        nucleus_reprs = torch.cat(object_reprs["nucleus"], dim=0)  # (4, dim)
        nucleus_reprs = torch.nn.functional.normalize(nucleus_reprs, dim=-1)  # (4, dim)
        nucleus_sim_matrix = torch.matmul(nucleus_reprs, nucleus_reprs.T)  # (4, 4)
        nucleus_sim_matrix = nucleus_sim_matrix.float().cpu().numpy()
        for i in range(4):
            for j in range(4):
                if i < j:
                    sample_sim_satellite.append(satellite_sim_matrix[i][j].item())
                    sample_sim_nucleus.append(nucleus_sim_matrix[i][j].item())
        sims_satellite.append(np.mean(sample_sim_satellite))
        sims_nucleus.append(np.mean(sample_sim_nucleus))
        print(f"pair_id: {pair_id}, satellite_sim_matrix:\n {satellite_sim_matrix}")
        print(f"pair_id: {pair_id}, nucleus_sim_matrix:\n {nucleus_sim_matrix}")
        print("-" * 20)
    
    save_dir = root_dir / "eval/WhatsUp/results/test_vit_direction_vit_sim"
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "res.jsonl")
    data = [
        {"before_connector": before_connector, "sims_cls": sims_cls},
        {"before_connector": before_connector, "sims_satellite": sims_satellite},
        {"before_connector": before_connector, "sims_nucleus": sims_nucleus},
        ]
    with jsonlines.open(save_path, 'a') as f:
        for item in data:
            f.write(item)
        
    print(f"Avg sim cls: {np.mean(sims_cls)}")
    print(f"Avg sim satellite: {np.mean(sims_satellite)}")
    print(f"Avg sim nucleus: {np.mean(sims_nucleus)}")

def explore_rope_attention_by_dimension_group(model_name, dataset_name, device, only_center=False, normalize_together=False, num_samples=10, head_id=None, accumulated_attn=False):
    """
    only_center: whether to only use the center patch of the object
    normalize_together: whether to normalize the attention weights corresponding to left and right together
    """
    # set save path
    exp_dir = root_dir / "figures"
    if head_id is None:
        exp_dir = os.path.join(exp_dir, "explore_rope_attention_by_dimension_group", f"{model_name}")
    else:
        exp_dir = os.path.join(exp_dir, "explore_rope_attention_by_dimension_group_headwise", f"{model_name}", f"head_{head_id}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # patch
    if "qwen2_5" in model_name:
        replace_qwen2_5_vl_test_directions_processor_return_indices()
        replace_qwen2_5_vl_attention_pattern_dimension_group(head_id=head_id)
    elif "qwen2" in model_name:
        replace_qwen2_vl_test_directions_processor_return_indices()
        replace_qwen2_vl_attention_pattern_dimension_group(head_id=head_id)
    else:
        raise NotImplementedError(f"Model {model_name} is not supported for this exploration.")
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device)
        model.eval()
        model_dir = MODEL_NAME_TO_PATH[model_name]
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

        vit, llm = model.visual, model.model
        vision_process_func = process_vision_info_qwen
        vit_layer_num = model.config.vision_config.depth

    
    # load data
    # dataloader = load_data(dataset_name, model_name, processor, data_num=None, random=False, batch_size=batch_size)

    # load batch of images
    if True:
        whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
        whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
        dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
        image_files = os.listdir(dir_name)
        image_names = [file_name.split('.')[0] for file_name in image_files]
        image_pairs = []
        for i, img_name in enumerate(image_names):
            obj_satellite = img_name.split('_')[0]
            obj_nucleus = img_name.split('_')[-1]
            image_pairs.append((obj_satellite, obj_nucleus))
        image_pairs = list(set(image_pairs))
        
        bboxes = {}
        bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
        with jsonlines.open(bboxes_path, "r") as f:
            for sample in f:
                bboxes[sample["image"]] = sample["bbox"]


    relations = ["left_of", "right_of", "behind", "in-front_of"]
    # layer_num * attn_h & attn_w & attn_weights & attn_s_s & attn_s_n & attn_n_n & attn_n_s
    attention_left = [[[] for _ in range(vit_layer_num)] for _ in range(7)]
    attention_right = [[[] for _ in range(vit_layer_num)] for _ in range(7)]
    attention_behind = [[[] for _ in range(vit_layer_num)] for _ in range(7)]
    attention_front = [[[] for _ in range(vit_layer_num)] for _ in range(7)]
    attention_all_relations = [attention_left, attention_right, attention_behind, attention_front]
    
    # size_diff_ratios = []
    for pair_id, image_pair in tqdm(enumerate(image_pairs)):
        
        if pair_id >= num_samples:
            break
        
        obj_satellite, obj_nucleus = image_pair
        
        # get four images for a pair
        sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
        sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
        sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
        
        # get inputs
        model_inputs = get_model_inputs(
            model_name=model_name,
            processor=processor,
            vision_process_func=vision_process_func,
            image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
            prompts=["describe the image and tell me what is the main object in the image" for _ in range(4)]
        )
        
        inputs, image_indices = model_inputs
        # inputs = inputs.to(device)
        
        # get object info
        if True:
            # get resized bboxes
            sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
            resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                # draw bboxes
                # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                
                # 1. get original sizes
                image = Image.open(image_path).convert('RGB')
                original_width, original_height = image.size
                image.close()
                # 2. get current size (after resize)
                if "llava" in model_name:
                    height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, height, width = image_grid_thw[idx].cpu().numpy()
                    height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                    width *= model.config.vision_config.spatial_patch_size
                elif "intern" in model_name:
                    block_image_width, block_image_height = inputs["block_wh_list"][idx]
                    height = block_image_height * model_config.vision_config.image_size
                    width = block_image_width * model_config.vision_config.image_size
                # 3. get resized bboxes
                scale_width = width / original_width
                scale_height = height / original_height
                # print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale_width}, {scale_height}")
                # import pdb; pdb.set_trace()
                x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                
            # get object patches
            object_patch_ids = [{"satellite": None, "nucleus": None} for _ in range(4)]
            if "llava" in model_name:
                patch_size = model_config.vision_config.patch_size
            elif "qwen" in model_name:
                # patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
                patch_size = model.config.vision_config.spatial_patch_size
            elif "intern" in model_name:
                # patch_size = model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)
                patch_size = model_config.vision_config.patch_size
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                            # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids[idx][obj_name] = obj_patch_ids
        
        # skip bad cases
        if False:
            if direction == "fb" and not only_center:
                # for behind/front
                object_max_b = max(len(object_patch_ids[2]["satellite"]), len(object_patch_ids[2]["nucleus"]))
                object_min_b = min(len(object_patch_ids[2]["satellite"]), len(object_patch_ids[2]["nucleus"]))
                object_max_f = max(len(object_patch_ids[3]["satellite"]), len(object_patch_ids[3]["nucleus"]))
                object_min_f = min(len(object_patch_ids[3]["satellite"]), len(object_patch_ids[3]["nucleus"]))
                ratio_0 = object_min_b / object_max_b
                ratio_1 = object_min_f / object_max_f
                if object_min_b / object_max_b < 0.4 or object_min_f / object_max_f < 0.4:
                    print(f"pass {pair_id}: {image_pair}, ratio: {ratio_0, ratio_1}, {object_min_b}/{object_max_b}, {object_min_f}/{object_max_f}")
                    continue
            else:
                # for behind/front
                object_max_l = max(len(object_patch_ids[0]["satellite"]), len(object_patch_ids[0]["nucleus"]))
                object_min_l = min(len(object_patch_ids[0]["satellite"]), len(object_patch_ids[0]["nucleus"]))
                object_max_r = max(len(object_patch_ids[1]["satellite"]), len(object_patch_ids[1]["nucleus"]))
                object_min_r = min(len(object_patch_ids[1]["satellite"]), len(object_patch_ids[1]["nucleus"]))
                ratio_0 = object_min_l / object_max_l
                ratio_1 = object_min_r / object_max_r
                print(f"{pair_id}: {image_pair}, ratio: {ratio_0, ratio_1}, {object_min_l}/{object_max_l}, {object_min_r}/{object_max_r}")
                size_diff_ratios.extend([ratio_0, ratio_1])
                continue
        
        # forward and get the attention outputs of each layer
        tmp_attn_h = [[] for _ in range(vit_layer_num)]  # attn_h_left, attn_h_right, attn_h_behind, attn_h_in_front
        tmp_attn_w = [[] for _ in range(vit_layer_num)]  # attn_w_left, attn_w_right, attn_w_behind, attn_w_in_front
        tmp_attn_weights_s_to_s = [[] for _ in range(vit_layer_num)]  # attn_weights_left, attn_weights_right, attn_weights_behind, attn_weights_in_front
        tmp_attn_weights_s_to_n = [[] for _ in range(vit_layer_num)]
        tmp_attn_weights_n_to_n = [[] for _ in range(vit_layer_num)]
        tmp_attn_weights_n_to_s = [[] for _ in range(vit_layer_num)]
        # for each direction
        for idx in range(4):
            model_inputs = get_model_inputs(
                model_name=model_name,
                processor=processor,
                vision_process_func=vision_process_func,
                image_paths=[sample_image_paths[idx]],
                prompts=["describe the image and tell me what is the main object in the image"]
            )
            inputs, image_indices = model_inputs
            inputs = inputs.to(device)
            with torch.no_grad():  # Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionBlock， VisionAttention
                vision_outputs, sample_attn_weights_h, sample_attn_weights_w, sample_attn_weights = vit(
                    hidden_states=inputs["pixel_values"],
                    grid_thw=inputs["image_grid_thw"],
                )
        
            # process the attention weights layer by layer
            # get attentions from satellite to nucleus
            satellite_patch_ids = object_patch_ids[idx]["satellite"]
            nucleus_patch_ids = object_patch_ids[idx]["nucleus"]
            print(f"pair_id: {pair_id}, image_pair: {image_pair}, dir {idx}, satellite_patch_ids: {len(satellite_patch_ids)}, nucleus_patch_ids: {len(nucleus_patch_ids)}")
            
            for layer_id in range(vit_layer_num):
                
                # get attention weights
                attn_h = sample_attn_weights_h[layer_id]  # [len, len]
                attn_w = sample_attn_weights_w[layer_id]  # [len, len]
                attn_weights = sample_attn_weights[layer_id]  # [len, len]
                # print(f"attn_h: {attn_h.shape}, attn_w: {attn_w.shape}, attn_weights: {attn_weights.shape}")
                # import pdb; pdb.set_trace()
                
                if only_center:
                    # method1: get the attention score of the center patch
                    satellite_center_id = satellite_patch_ids[len(satellite_patch_ids) // 2]
                    nucleus_center_id = nucleus_patch_ids[len(nucleus_patch_ids) // 2]
                    attn_h_s_to_n = attn_h[satellite_center_id][nucleus_center_id].item()
                    attn_w_s_to_n = attn_w[satellite_center_id][nucleus_center_id].item()
                    attn_weights_s_to_n = attn_weights[satellite_center_id][nucleus_center_id].item()
                else:
                    # method2: get the average of relevant attention weights
                    attn_h_s_to_n = attn_h[satellite_patch_ids][:, nucleus_patch_ids].mean().item()
                    attn_w_s_to_n = attn_w[satellite_patch_ids][:, nucleus_patch_ids].mean().item()
                    attn_weights_s_to_n = attn_weights[satellite_patch_ids][:, nucleus_patch_ids].mean().item()
                    attn_weights_n_to_n = attn_weights[nucleus_patch_ids][:, nucleus_patch_ids].mean().item()
                    attn_weights_s_to_s = attn_weights[satellite_patch_ids][:, satellite_patch_ids].mean().item()
                    attn_weights_n_to_s = attn_weights[nucleus_patch_ids][:, satellite_patch_ids].mean().item()
                    # attn_weights_s_to_n = attn_weights_s_to_n.sum().item()
                    # print(f"dir {idx}, layer{layer_id}, Attn_sum: {attn_weights.sum()}, Attn_shape: {attn_weights.shape}, attn_sum: {attn_weights_s_to_n.sum()}, ele_num: {attn_weights_s_to_n.numel()}")
                tmp_attn_h[layer_id].append(attn_h_s_to_n)
                tmp_attn_w[layer_id].append(attn_w_s_to_n)
                tmp_attn_weights_s_to_n[layer_id].append(attn_weights_s_to_n)
                tmp_attn_weights_s_to_s[layer_id].append(attn_weights_s_to_s)
                tmp_attn_weights_n_to_n[layer_id].append(attn_weights_n_to_n)
                tmp_attn_weights_n_to_s[layer_id].append(attn_weights_n_to_s)
            # import pdb; pdb.set_trace()
        
        # For each layer, post processing layer by layer
        if accumulated_attn:
            n_in_s_ratio = [0 for _ in range(4)]
            n_in_n_ratio = [1 for _ in range(4)]
        for layer_id in range(vit_layer_num):
            # print("----"*10)
            # print("layer_id:", layer_id)
            if normalize_together:  # compare h_x, h_y, w_x, w_y together
                # For left and right:
                attn_h_w_sum = tmp_attn_h[layer_id][0] + tmp_attn_h[layer_id][1] + tmp_attn_w[layer_id][0] + tmp_attn_w[layer_id][1]
                attn_h_left_ratio = tmp_attn_h[layer_id][0] / attn_h_w_sum
                attn_w_left_ratio = tmp_attn_w[layer_id][0] / attn_h_w_sum
                attn_h_right_ratio = tmp_attn_h[layer_id][1] / attn_h_w_sum
                attn_w_right_ratio = tmp_attn_w[layer_id][1] / attn_h_w_sum
                # print(f"pair_id: {pair_id}, relation: left v.s. right, attn_h_left: {attn_h_left_ratio}, attn_h_right: {attn_h_right_ratio}, attn_w_left: {attn_w_left_ratio}, attn_w_right: {attn_w_right_ratio}")
                # For behind and in front of:
                attn_h_b_f_sum = tmp_attn_h[layer_id][2] + tmp_attn_h[layer_id][3] + tmp_attn_w[layer_id][2] + tmp_attn_w[layer_id][3]
                attn_h_behind_ratio = tmp_attn_h[layer_id][2] / attn_h_b_f_sum
                attn_w_behind_ratio = tmp_attn_w[layer_id][2] / attn_h_b_f_sum
                attn_h_in_front_ratio = tmp_attn_h[layer_id][3] / attn_h_b_f_sum
                attn_w_in_front_ratio = tmp_attn_w[layer_id][3] / attn_h_b_f_sum
                # print(f"pair_id: {pair_id}, relation: behind v.s. in front of, attn_h_behind: {attn_h_behind_ratio}, attn_h_in_front: {attn_h_in_front_ratio}, attn_w_behind: {attn_w_behind_ratio}, attn_w_in_front: {attn_w_in_front_ratio}")
            else:  # compare h_x, h_y, w_x, w_y separately
                for idx in range(4):
                    attn_h = tmp_attn_h[layer_id][idx]
                    attn_w = tmp_attn_w[layer_id][idx]
                    # # get the exp(x-x_{max})
                    # print(f"exp({attn_h - max(attn_h, attn_w)}), exp({attn_w - max(attn_h, attn_w)})")
                    # attn_h = np.exp(attn_h - max(attn_h, attn_w))
                    # attn_w = np.exp(attn_w - max(attn_h, attn_w))
                    # print(f"attn_h: {attn_h}, attn_w: {attn_w}")
                    # calculate the ratio
                    try:
                        attn_h_ratio = attn_h / (attn_h + attn_w)
                        attn_w_ratio = attn_w / (attn_h + attn_w)
                        # attn_h_ratio = attn_h
                        # attn_w_ratio = attn_w
                    except:
                        continue
                    if idx == 0:
                        attn_h_left_ratio = attn_h_ratio
                        attn_w_left_ratio = attn_w_ratio
                    elif idx == 1:
                        attn_h_right_ratio = attn_h_ratio
                        attn_w_right_ratio = attn_w_ratio
                    elif idx == 2:
                        attn_h_behind_ratio = attn_h_ratio
                        attn_w_behind_ratio = attn_w_ratio
                    elif idx == 3:
                        attn_h_in_front_ratio = attn_h_ratio
                        attn_w_in_front_ratio = attn_w_ratio
                        
            attn_weights_left = tmp_attn_weights_s_to_n[layer_id][0]
            attn_weights_right = tmp_attn_weights_s_to_n[layer_id][1]
            attn_weights_behind = tmp_attn_weights_s_to_n[layer_id][2]
            attn_weights_in_front = tmp_attn_weights_s_to_n[layer_id][3]
            if accumulated_attn:
                # accumulate the attention weights
                curr_n_in_s_ratio_left = (1 + tmp_attn_weights_s_to_s[layer_id][0]) * n_in_s_ratio[0] + tmp_attn_weights_s_to_n[layer_id][0] * n_in_n_ratio[0]
                curr_n_in_s_ratio_right = (1 + tmp_attn_weights_s_to_s[layer_id][1]) * n_in_s_ratio[1] + tmp_attn_weights_s_to_n[layer_id][1] * n_in_n_ratio[1]
                curr_n_in_s_ratio_behind = (1 + tmp_attn_weights_s_to_s[layer_id][2]) * n_in_s_ratio[2] + tmp_attn_weights_s_to_n[layer_id][2] * n_in_n_ratio[2]
                curr_n_in_s_ratio_in_front = (1 + tmp_attn_weights_s_to_s[layer_id][3]) * n_in_s_ratio[3] + tmp_attn_weights_s_to_n[layer_id][3] * n_in_n_ratio[3]
                curr_n_in_n_ratio_left = (1 + tmp_attn_weights_n_to_n[layer_id][0]) * n_in_n_ratio[0] + tmp_attn_weights_n_to_s[layer_id][0] * n_in_s_ratio[0]
                curr_n_in_n_ratio_right = (1 + tmp_attn_weights_n_to_n[layer_id][1]) * n_in_n_ratio[1] + tmp_attn_weights_n_to_s[layer_id][1] * n_in_s_ratio[1]
                curr_n_in_n_ratio_behind = (1 + tmp_attn_weights_n_to_n[layer_id][2]) * n_in_n_ratio[2] + tmp_attn_weights_n_to_s[layer_id][2] * n_in_s_ratio[2]
                curr_n_in_n_ratio_in_front = (1 + tmp_attn_weights_n_to_n[layer_id][3]) * n_in_n_ratio[3] + tmp_attn_weights_n_to_s[layer_id][3] * n_in_s_ratio[3]
                attn_weights_left, attn_weights_right, attn_weights_behind, attn_weights_in_front = curr_n_in_s_ratio_left, curr_n_in_s_ratio_right, curr_n_in_s_ratio_behind, curr_n_in_s_ratio_in_front
                # update the ratio
                n_in_s_ratio[0] = curr_n_in_s_ratio_left
                n_in_s_ratio[1] = curr_n_in_s_ratio_right
                n_in_s_ratio[2] = curr_n_in_s_ratio_behind
                n_in_s_ratio[3] = curr_n_in_s_ratio_in_front
                n_in_n_ratio[0] = curr_n_in_n_ratio_left
                n_in_n_ratio[1] = curr_n_in_n_ratio_right
                n_in_n_ratio[2] = curr_n_in_n_ratio_behind
                n_in_n_ratio[3] = curr_n_in_n_ratio_in_front                
            
            # save
            attn_res_left = [attn_h_left_ratio, attn_w_left_ratio, attn_weights_left, tmp_attn_weights_s_to_s[layer_id][0], tmp_attn_weights_s_to_n[layer_id][0], tmp_attn_weights_n_to_n[layer_id][0], tmp_attn_weights_n_to_s[layer_id][0]]
            attn_res_right = [attn_h_right_ratio, attn_w_right_ratio, attn_weights_right, tmp_attn_weights_s_to_s[layer_id][1], tmp_attn_weights_s_to_n[layer_id][1], tmp_attn_weights_n_to_n[layer_id][1], tmp_attn_weights_n_to_s[layer_id][1]]
            attn_res_behind = [attn_h_behind_ratio, attn_w_behind_ratio, attn_weights_behind, tmp_attn_weights_s_to_s[layer_id][2], tmp_attn_weights_s_to_n[layer_id][2], tmp_attn_weights_n_to_n[layer_id][2], tmp_attn_weights_n_to_s[layer_id][2]]
            attn_res_in_front = [attn_h_in_front_ratio, attn_w_in_front_ratio, attn_weights_in_front, tmp_attn_weights_s_to_s[layer_id][3], tmp_attn_weights_s_to_n[layer_id][3], tmp_attn_weights_n_to_n[layer_id][3], tmp_attn_weights_n_to_s[layer_id][3]]
            attn_res = [attn_res_left, attn_res_right, attn_res_behind, attn_res_in_front]
            for idx in range(4):
                attention_all_relations[idx][0][layer_id].append(attn_res[idx][0])
                attention_all_relations[idx][1][layer_id].append(attn_res[idx][1])
                attention_all_relations[idx][2][layer_id].append(attn_res[idx][2])
                attention_all_relations[idx][3][layer_id].append(attn_res[idx][3])
                attention_all_relations[idx][4][layer_id].append(attn_res[idx][4])
                attention_all_relations[idx][5][layer_id].append(attn_res[idx][5])
                attention_all_relations[idx][6][layer_id].append(attn_res[idx][6])
        # break
    # avg_ratio = np.mean(size_diff_ratios)
    # print(avg_ratio)
    
    # save the attention results
    for idx in range(len(relations)):
        rel = relations[idx]
        save_path = os.path.join(exp_dir, f"{rel}.jsonl")
        with jsonlines.open(save_path, 'w') as f:
            for layer_id in range(vit_layer_num):
                data = {
                    "attn_h": attention_all_relations[idx][0][layer_id],
                    "attn_w": attention_all_relations[idx][1][layer_id],
                    "attn_weights_s_to_n": attention_all_relations[idx][3][layer_id],
                    "attn_weights_s_to_s": attention_all_relations[idx][4][layer_id],
                    "attn_weights_n_to_n": attention_all_relations[idx][5][layer_id],
                    "attn_weights_n_to_s": attention_all_relations[idx][6][layer_id],
                }
                f.write(data)
        
    # plot
    # x_axis: layer_id
    # y_axis: attn_left_h, attn_left_w, attn_right_h, attn_right_w, attn_behind_h, attn_behind_w, attn_in_front_h, attn_in_front_w
    if False:
        directions = ["lr", "fb"]
        relation_ids = [[0, 1], [2, 3]]  # left/right, behind/in front of
        for k in range(2):
            direction = directions[k]
            save_path = os.path.join(exp_dir, f"{model_name}.pdf")
            prefix = ""
            if only_center:
                prefix += "-only_center"
            if normalize_together:
                prefix += "-normalize_together"
            prefix += f"-{num_samples}_samples"
            prefix += f"-{direction}"
            save_path = save_path.replace(".pdf", f"{prefix}.pdf")
                
            plt.figure()
            h_marker = 'o'
            w_marker = 's'
            relation_colors = [mcolors.to_hex(plt.cm.viridis(i / len(relations))) for i in range(len(relations))]
            relation_colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
            relation_labels = ["Left", "Right", "Behind", "In front of"]
            layer_ids = [i + 1 for i in range(vit_layer_num)]
            for idx in relation_ids[k]:
                # attention_height
                mean = np.mean(attention_all_relations[idx][1], axis=-1)
                std_error = np.std(attention_all_relations[idx][1], axis=-1) / np.sqrt(len(attention_all_relations[idx][1]))
                plt.plot(
                    layer_ids,
                    mean,
                    color=relation_colors[idx],
                    label=f"{relation_labels[idx]} (X Axis)",
                    alpha=1, 
                    marker=h_marker,
                    markersize=6, 
                    # markerfacecolor="none",
                    # markeredgecolor=colors[i],
                    markeredgewidth=1.5
                )
                # attention_width
                mean = np.mean(attention_all_relations[idx][0], axis=-1)
                std_error = np.std(attention_all_relations[idx][0], axis=-1) / np.sqrt(len(attention_all_relations[idx][0]))
                plt.plot(
                    layer_ids,
                    mean,
                    color=relation_colors[idx],
                    label=f"{relation_labels[idx]} (Y Axis)",
                    alpha=1, 
                    marker=w_marker,
                    markersize=6, 
                    # markerfacecolor="none",
                    # markeredgecolor=colors[i],
                    markeredgewidth=1.5
                )
            
            plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
            plt.ylabel("Contributions to Attention (satellite -> nucleus)", color=mcolors.to_rgba('black', 1))
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
            
            plt.legend()
            plt.savefig(save_path, bbox_inches='tight')
            print(save_path)
    
    if True:
        directions = ["l", "r", "b", "f"]
        for idx in range(4):
            direction = directions[idx]
            save_path = os.path.join(exp_dir, f"{model_name}.pdf")
            prefix = ""
            if only_center:
                prefix += "-only_center"
            if normalize_together:
                prefix += "-normalize_together"
            prefix += f"-{num_samples}_samples"
            prefix += f"-{direction}"
            save_path = save_path.replace(".pdf", f"{prefix}.pdf")
                
            plt.figure()
            h_marker = 'o'
            w_marker = 's'
            relation_colors = [mcolors.to_hex(plt.cm.viridis(i / len(relations))) for i in range(len(relations))]
            relation_colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
            relation_labels = ["Left", "Right", "Behind", "In front of"]
            layer_ids = [i + 1 for i in range(vit_layer_num)]
            
            # attention_height
            mean = np.mean(attention_all_relations[idx][1], axis=-1)
            std_error = np.std(attention_all_relations[idx][1], axis=-1) / np.sqrt(len(attention_all_relations[idx][1]))
            plt.plot(
                layer_ids,
                mean,
                color=relation_colors[idx],
                label=f"{relation_labels[idx]} (X Axis)",
                alpha=1, 
                marker=h_marker,
                markersize=6, 
                # markerfacecolor="none",
                # markeredgecolor=colors[i],
                markeredgewidth=1.5
            )
            # attention_width
            mean = np.mean(attention_all_relations[idx][0], axis=-1)
            std_error = np.std(attention_all_relations[idx][0], axis=-1) / np.sqrt(len(attention_all_relations[idx][0]))
            plt.plot(
                layer_ids,
                mean,
                color=relation_colors[idx],
                label=f"{relation_labels[idx]} (Y Axis)",
                alpha=1, 
                marker=w_marker,
                markersize=6, 
                # markerfacecolor="none",
                # markeredgecolor=colors[i],
                markeredgewidth=1.5
            )
            
            plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
            plt.ylabel("Contributions to Attention (satellite -> nucleus)", color=mcolors.to_rgba('black', 1))
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
            
            plt.legend()
            plt.savefig(save_path, bbox_inches='tight')
            print(save_path)
    
    if True:
        # plot attention weights
        save_path = os.path.join(exp_dir, f"attn_weights_{model_name}.pdf")
        prefix = ""
        if only_center:
            prefix += "-only_center"
        if normalize_together:
            prefix += "-normalize_together"
        if accumulated_attn:
            prefix += "-accumulated"
        prefix += f"-{num_samples}_samples"
        save_path = save_path.replace(".pdf", f"{prefix}.pdf")
                
        plt.figure()
        markers = ['o', 'o', 'o', 'o']
        relation_colors = [mcolors.to_hex(plt.cm.viridis(i / len(relations))) for i in range(len(relations))]
        relation_colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
        relation_labels = ["Left", "Right", "Behind", "In front of"]
        layer_ids = [i + 1 for i in range(vit_layer_num)]
        for idx in range(len(relations)):
            # if direction == "lr" and idx > 1:
            #     continue
            # if direction == "fb" and idx < 2:
            #     continue
            # # attention_weights
            mean = np.mean(attention_all_relations[idx][2], axis=-1)
            std_error = np.std(attention_all_relations[idx][2], axis=-1) / np.sqrt(len(attention_all_relations[idx][2]))
            plt.plot(
                layer_ids,
                mean,
                color=relation_colors[idx],
                label=f"{relation_labels[idx]}",
                alpha=1, 
                marker=markers[idx],
                markersize=6, 
                # markerfacecolor="none",
                # markeredgecolor=colors[i],
                markeredgewidth=1.5
            )
            plt.fill_between(
                layer_ids, 
                mean - std_error, 
                mean + std_error, 
                color=relation_colors[idx], 
                alpha=0.1, 
            )
        
        plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
        if accumulated_attn:
            plt.ylabel("Proportion of the nucleus in the satellite per patch", color=mcolors.to_rgba('black', 1))
        else:
            plt.ylabel("Attention weights (satellite -> nucleus) per patch", color=mcolors.to_rgba('black', 1))
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight')
        print(save_path)

def explore_rope_by_sensitivity_score(model_name, dataset_name, device, num_samples=10, head_id=None, activation_name=None):
    
    # set save path
    exp_dir = root_dir / "figures"
    if head_id is None:
        exp_dir = os.path.join(exp_dir, "explore_rope_by_sensitivity_score", f"{model_name}")
    else:
        exp_dir = os.path.join(exp_dir, "explore_rope_by_sensitivity_score", f"{model_name}", f"head_{head_id}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # load model
    if True:
        model, processor, tokenizer = load_model(model_name, device, use_flash_attention=False)
        model.eval()
        model_dir = MODEL_NAME_TO_PATH[model_name]
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        vit, llm = None, None
        if "qwen2_5" in model_name:
            vit, llm = model.vision_tower, model.language_model
            vision_process_func = process_vision_info_qwen
            vit_layer_num = model.config.vision_config.num_hidden_layers
            replace_qwen2_5_vl_test_directions_processor_return_indices()
        elif "qwen2" in model_name:
            vit, llm = model.visual, model.model
            vision_process_func = process_vision_info_qwen
            vit_layer_num = model.config.vision_config.depth
            replace_qwen2_vl_test_directions_processor_return_indices()
            replace_qwen2_vl_rope_sensitivity(activation_name=activation_name)
        else:
            pass

    # load batch of images
    if True:
        whatsup_a_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_images"
        whatsup_b_dir = data_dir / "whatsup_vlms/data/whatsup_vlms_data/controlled_clevr"
        dir_name = whatsup_a_dir if "whatsup_a" in dataset_name else whatsup_b_dir
        image_files = os.listdir(dir_name)
        image_names = [file_name.split('.')[0] for file_name in image_files]
        image_pairs = []
        for i, img_name in enumerate(image_names):
            obj_satellite = img_name.split('_')[0]
            obj_nucleus = img_name.split('_')[-1]
            image_pairs.append((obj_satellite, obj_nucleus))
        image_pairs = list(set(image_pairs))
        
        bboxes = {}
        bboxes_path = root_dir / "eval/WhatsUp/bboxes/bbox.jsonl"
        with jsonlines.open(bboxes_path, "r") as f:
            for sample in f:
                bboxes[sample["image"]] = sample["bbox"]

    # prepare direction probs
    if True:
        token_ids = {}
        if "llava" in model_name:
            token_ids = {"in": ["in", "In", "▁In"], "behind": ["behind", "Behind", "▁Be"], "left": ["left", "Left", "▁Left"], "right": ["right", "Right", "▁Right"]}
        elif "qwen" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["Ġbehind", "Behind"], "left": ["left", "Left"], "right": ["right", "Right"]}
        elif "intern" in model_name:
            token_ids = {"in": ["in", "In"], "behind": ["beh", "Beh"], "left": ["left", "Left"], "right": ["right", "Right"]}
        else:
            pass
        
        for key, value in token_ids.items():
            token_ids[key] = [tokenizer.convert_tokens_to_ids(token) for token in value]
            
        direction_ids = {
            "in": token_ids["in"],
            "behind": token_ids["behind"],
            "left": token_ids["left"],
            "right": token_ids["right"]
        }
    
    # start
    for pair_id, image_pair in tqdm(enumerate(image_pairs)):
        
        if pair_id >= num_samples:
            break
        
        obj_satellite, obj_nucleus = image_pair
        
        # get four images for a pair
        relations = ["left_of", "right_of", "behind", "in-front_of"]
        sample_image_files = [f"{image_pair[0]}_{relation}_{image_pair[1]}.jpeg" for relation in relations]
        sample_image_paths = [os.path.join(dir_name, file_name) for file_name in sample_image_files]
        sample_image_names = [f"{image_pair[0]}_{relation}_{image_pair[1]}" for relation in relations]
        
        # get inputs
        instruction = f"Is the {obj_satellite} in front of/behind/to the left of/to the right of the {obj_nucleus}? Please choose the best answer from the four options: [In front of, Behind, Left, Right], and reply with only one word. \nYour answer is:"
        model_inputs = get_model_inputs(
            model_name=model_name,
            processor=processor,
            vision_process_func=vision_process_func,
            image_paths=[os.path.join(dir_name, file_name) for file_name in sample_image_files],
            prompts=[instruction for _ in range(4)]
        )
        
        inputs, image_indices = model_inputs
        
        # get inputs info
        if True:
            # get resized bboxes
            sample_bboxes = [bboxes[image_name] for image_name in sample_image_names]
            resized_bboxes = [{"satellite": None, "nucleus": None} for _ in range(4)]
            for idx, image_path in enumerate(sample_image_paths):
                # draw bboxes
                # draw_bbox(image_path, sample_bboxes[idx], pair_check_dir, save_format="pdf")
                
                # 1. get original sizes
                image = Image.open(image_path).convert('RGB')
                original_width, original_height = image.size
                image.close()
                # 2. get current size (after resize)
                if "llava" in model_name:
                    height, width = inputs["pixel_values"][idx].shape[1:]  # after resize (336*336)
                elif "qwen" in model_name:  # qwen concat samples together in one pixel values, so we can't leverage the pixel values to get the resized h and w
                    image_grid_thw = inputs["image_grid_thw"]   # after resize & patching, before merging
                    _, height, width = image_grid_thw[idx].cpu().numpy()
                    height *= model.config.vision_config.spatial_patch_size  # after resize, before patching
                    width *= model.config.vision_config.spatial_patch_size
                elif "intern" in model_name:
                    block_image_width, block_image_height = inputs["block_wh_list"][idx]
                    height = block_image_height * model_config.vision_config.image_size
                    width = block_image_width * model_config.vision_config.image_size
                # 3. get resized bboxes
                scale_width = width / original_width
                scale_height = height / original_height
                print(f"pair_id: {pair_id}, original_image size: {original_width}, {original_height}, image_size: {width}, {height}, scale: {scale_width}, {scale_height}")
                # import pdb; pdb.set_trace()
                x1, y1, x2, y2 = sample_bboxes[idx][obj_satellite]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                resized_bboxes[idx]["satellite"] = (x1, y1, x2, y2)
                x1, y1, x2, y2 = sample_bboxes[idx][obj_nucleus]  # original coordinates
                x1, y1, x2, y2 = int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)
                resized_bboxes[idx]["nucleus"] = (x1, y1, x2, y2)
                
            # get object patches
            object_patch_ids = [{"satellite": None, "nucleus": None} for _ in range(4)]
            if "llava" in model_name:
                patch_size = model_config.vision_config.patch_size
            elif "qwen" in model_name:
                patch_size = model.config.vision_config.spatial_patch_size * model.config.vision_config.spatial_merge_size
            elif "intern" in model_name:
                patch_size = model_config.vision_config.patch_size * (1 / model_config.downsample_ratio)
            for idx, image_path in enumerate(sample_image_paths):
                for obj_name, obj_bbox in resized_bboxes[idx].items():
                    obj_patch_ids = []
                    x1, y1, x2, y2 = obj_bbox
                    for i in range(int(width // patch_size)):
                        for j in range(int(height // patch_size)):
                            x1_, y1_, x2_, y2_ = i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size
                            if not (x1_ > x2 or x2_ < x1 or y1_ > y2 or y2_ < y1):
                            # if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                                obj_patch_ids.append(j * int(width // patch_size) + i)
                    object_patch_ids[idx][obj_name] = obj_patch_ids
            
        
        # forward
        for idx in range(4):
            model_inputs = get_model_inputs(
                model_name=model_name,
                processor=processor,
                vision_process_func=vision_process_func,
                image_paths=[sample_image_paths[idx]],
                prompts=[instruction]
            )
            inputs, image_indices = model_inputs
            inputs = inputs.to(device)
            
            with torch.enable_grad():
                outputs, all_activations = model(
                    **inputs,
                    return_dict=True, 
                )
                print(f"sample {idx}")
        
                # get the logits
                logits = outputs.logits
                last_token_logits = logits[0, -1, :]  # [batch_size, vocab_size]
                scale_factor = torch.max(torch.abs(last_token_logits))
                if scale_factor > 1e3:
                    last_token_logits = last_token_logits / scale_factor
                # probs = F.softmax(last_token_logits, dim=-1)
                probs = last_token_logits
        
                left_prob = probs[direction_ids["left"]].sum()
                right_prob = probs[direction_ids["right"]].sum()
                behind_prob = probs[direction_ids["behind"]].sum()
                in_front_prob = probs[direction_ids["in"]].sum()
                if idx < 2:
                    print(f"pair_id: {pair_id}, sample {idx}, left prob: {left_prob.item()}, right prob: {right_prob.item()}")
                else:
                    print(f"pair_id: {pair_id}, sample {idx}, behind prob: {behind_prob.item()}, in front prob: {in_front_prob.item()}")

                # define loss
                if idx == 0:
                    # loss = left_prob - right_prob - behind_prob - in_front_prob
                    loss = left_prob - right_prob
                elif idx == 1:
                    # loss = right_prob - left_prob - behind_prob - in_front_prob
                    loss = right_prob - left_prob
                elif idx == 2:
                    # loss = behind_prob - in_front_prob - left_prob - right_prob
                    loss = behind_prob - in_front_prob
                elif idx == 3:
                    # loss = in_front_prob - behind_prob - left_prob - right_prob
                    loss = in_front_prob - behind_prob
                else:
                    raise ValueError(f"Invalid idx: {idx}")
                
                # backward
                # q_gradients = []
                # k_gradients = []
                # def save_q_gradient(grad):
                #     q_gradients.append(grad.clone())
                    
                # def save_k_gradient(grad):
                #     k_gradients.append(grad.clone())

                params = []  # get all q and k parameters
                if activation_name == "qk":
                    for item in all_activations:
                        q, k = item
                        q.retain_grad()
                        k.retain_grad()
                        params.append(q)
                        params.append(k)
                elif activation_name == "attn_output":
                    for item in all_activations:
                        item.retain_grad()
                        params.append(item)
                else:
                    raise ValueError(f"Invalid activation_name: {activation_name}")
                        
                model.zero_grad()
                loss.backward()
                # max_grad_norm = 0.1
                # torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                
            # get variables & partial derivatives
            for layer_id in range(vit_layer_num):
                
                satellite_patch_ids = object_patch_ids[idx]["satellite"]
                nucleus_patch_ids = object_patch_ids[idx]["nucleus"]
                if activation_name == "qk":
                    qk = all_activations[layer_id]  # q/k: 2 * [num_heads, len, head_dim]
                    h_importance, w_importance = process_qk_sensitivity(qk, satellite_patch_ids, nucleus_patch_ids)
                    print(f"gold_ans: {relations[idx]}, w_importance: {w_importance}, h_importance: {h_importance}")
                elif activation_name == "attn_output":
                    attn_output = all_activations[layer_id]  # attn_output: [bsz*len, n_heads, head_dim]
                    head_idxs = process_attn_output_sensitivity(attn_output, satellite_patch_ids, nucleus_patch_ids)
                    print(f"head_idxs({relations[idx]}): {head_idxs}")
            import pdb; pdb.set_trace()

def process_qk_sensitivity(qk, satellite_patch_ids, nucleus_patch_ids):
    q, k = qk
    q = q * q.grad.detach().float()
    k = k * k.grad.detach().float()
    
    n_heads, seq_length, head_dim = q.shape
    half_dim = head_dim // 2
    quarter_dim = half_dim // 2
    # q_h_1 = q.grad[..., :quarter_dim].detach().float()    # 1-20
    # q_w_1 = q.grad[..., half_dim:half_dim+quarter_dim].detach().float()    # 21-40
    # q_h_2 = q.grad[..., quarter_dim:half_dim].detach().float()  # 41-60
    # q_w_2 = q.grad[..., half_dim+quarter_dim:].detach().float()   # 61-80
    
    q_h_1 = q[..., :quarter_dim]
    q_h_2 = q[..., half_dim:half_dim+quarter_dim]
    q_w_1 = q[..., quarter_dim:half_dim]
    q_w_2 = q[..., half_dim+quarter_dim:]
    k_h_1 = k[..., :quarter_dim]
    k_h_2 = k[..., half_dim:half_dim+quarter_dim]
    k_w_1 = k[..., quarter_dim:half_dim]
    k_w_2 = k[..., half_dim+quarter_dim:]
    
    q_h = torch.cat([q_h_1, q_h_2], dim=-1)  # [num_heads, len, head_dim // 2]
    q_w = torch.cat([q_w_1, q_w_2], dim=-1)  # [num_heads, len, head_dim // 2]
    k_h = torch.cat([k_h_1, k_h_2], dim=-1)  # [num_heads, len, head_dim // 2]
    k_w = torch.cat([k_w_1, k_w_2], dim=-1)  # [num_heads, len, head_dim // 2]
    
    # print(q_h_1.shape)  # [num_heads, len, head_dim // 4]
    
    # get target gradients
    # head_ids = [2, 13]
    # q_h = q_h[head_ids]
    # q_w = q_w[head_ids]
    # k_h = k_h[head_ids]
    # k_w = k_w[head_ids]
    
    patch_ids = satellite_patch_ids + nucleus_patch_ids
    
    # fix nan/inf
    if True:
        q_h = torch.nan_to_num(q_h[:, patch_ids], nan=0.0, posinf=0.0, neginf=0.0)
        q_w = torch.nan_to_num(q_w[:, patch_ids], nan=0.0, posinf=0.0, neginf=0.0)
        k_h = torch.nan_to_num(k_h[:, patch_ids], nan=0.0, posinf=0.0, neginf=0.0)
        k_w = torch.nan_to_num(k_w[:, patch_ids], nan=0.0, posinf=0.0, neginf=0.0)
    
    # get the importance
    # print(q_h_1.abs())
    # import pdb; pdb.set_trace()
    q_h_importance = q_h.abs().sum(dim=0).sum(dim=-1).mean().item()
    q_w_importance = q_w.abs().sum(dim=0).sum(dim=-1).mean().item()
    k_h_importance = k_h.abs().sum(dim=0).sum(dim=-1).mean().item()
    k_w_importance = k_w.abs().sum(dim=0).sum(dim=-1).mean().item()
    # print(q_h_importance)
    # print(q_h_importance)
    
    total_importance = q_h_importance + q_w_importance + k_h_importance + k_w_importance
    try:
        h_importance = (q_h_importance + k_h_importance) / total_importance
        w_importance = (q_w_importance + k_w_importance) / total_importance
    except:
        h_importance, w_importance = None, None
    
    return h_importance, w_importance

def process_attn_output_sensitivity(attn_output, satellite_patch_ids, nucleus_patch_ids):
    
    # print("attn_output.grad:", attn_output.grad)
    
    attn_output = attn_output.grad.detach().float()
    # attn_output = attn_output * attn_output.grad.detach().float()
    
    attn_output = attn_output.transpose(0, 1)  # [n_heads, bsz*len, head_dim]
    # print(f"attn_output: {attn_output[0]}")
    
    if True:
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=0.0, neginf=0.0)
    
    n_heads, seq_length, head_dim = attn_output.shape
    head_importances = attn_output.abs().sum(dim=-1).sum(dim=-1).tolist()
    # print(f"head_importances: {head_importances}")
    
    head_idxs = [i for i in range(n_heads)]
    # print(f"head_importances: {head_importances}")
    head_idxs.sort(key=lambda x: head_importances[x], reverse=True)
    
    return head_idxs

def test_rope_attention_h_w_separate(
    model_name="qwen2_5_vl",
    dataset_name="whatsup_b",
    device="cuda:1",
    tag="2",
    data_num=1000, 
    batch_size=8,
    random=False,
    method="layernorm"
    ):
    
    if "qwen2_5" in model_name:
        pass
    elif "qwen2" in model_name:
        replace_qwen2_vl_rope_attention_h_w_separate(method=method)
        pass
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
    
    model, processor, tokenizer = load_model(model_name, device)
    
    official_dataset_name = DATASET_NAME_TO_OFFICIAL.get(dataset_name, dataset_name)
    save_dir = root_dir / f"eval/{official_dataset_name}/results/test_rope_attention_h_w_separate"
    save_dir = os.path.join(save_dir, f"{model_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_{tag}.jsonl")
    
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
        print(f"acc: {acc}")  # 0.612, 0.607, 0.624, 0.617, 0.629, 0.614

        return acc

def test_rope_scaling(
    model_name="qwen2_5_vl",
    dataset_name="whatsup_b",
    device="cuda:1",
    tag="2",
    data_num=1000, 
    batch_size=8,
    random=False,
    scaling_type="linear",
    alpha=1.0,
    gamma=2.0,
    beta=0.1,
    base=512,
    poly_p=8,
    poly_alpha=99,
    sig_alpha=99,
    sig_mid_point=0.5,
    sig_k=20.0,
    ):
    
    if "qwen2_5" in model_name:
        pass
    elif "qwen2" in model_name:
        replace_qwen2_vl_scaling_rope(
            scaling_type=scaling_type, 
            alpha=alpha, 
            gamma=gamma, 
            beta=beta,
            base=base,
            poly_p=poly_p,
            poly_alpha=poly_alpha,
            sig_alpha=sig_alpha,
            sig_mid_point=sig_mid_point,
            sig_k=sig_k
        )
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
    
    model, processor, tokenizer = load_model(model_name, device)
    
    official_dataset_name = DATASET_NAME_TO_OFFICIAL.get(dataset_name, dataset_name)
    para_str = f"{scaling_type}_alpha_{alpha}_gamma_{gamma}_beta_{beta}_base_{base}_poly_p_{poly_p}_poly_alpha_{poly_alpha}_sig_alpha_{sig_alpha}_sig_mid_point_{sig_mid_point}_sig_k_{sig_k}"
    save_dir = root_dir / f"eval/{official_dataset_name}/results/test_rope_scaling_{para_str}"
    save_dir = os.path.join(save_dir, f"{model_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_{tag}.jsonl")
    
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
        print(f"acc: {acc}")  # 0.612, 0.607, 0.624, 0.617, 0.629, 0.614

        return acc

def plot_vit_direction_vlm(model_name):
    
    res_whatsup_dir = root_dir / "eval/WhatsUp/results/test_vit_directions_vlm"
    res_vsr_dir = root_dir / "eval/VSR/results/test_vit_directions_vlm"
    res_whatsup_dir = os.path.join(res_whatsup_dir, f"{model_name}")
    res_vsr_dir = os.path.join(res_vsr_dir, f"{model_name}")
    
    files_whatsup_a = glob.glob(f"{res_whatsup_dir}/final_whatsup_a_*.jsonl")
    files_whatsup_b = glob.glob(f"{res_whatsup_dir}/final_whatsup_b_*.jsonl")
    files_cocoqa_1 = glob.glob(f"{res_whatsup_dir}/final_cocoqa_1_*.jsonl")
    files_cocoqa_2 = glob.glob(f"{res_whatsup_dir}/final_cocoqa_2_*.jsonl")
    files_gqa_1 = glob.glob(f"{res_whatsup_dir}/final_gqa_1_*.jsonl")
    files_gqa_2 = glob.glob(f"{res_whatsup_dir}/final_gqa_2_*.jsonl")
    files_vsr = glob.glob(f"{res_vsr_dir}/final_vsr_*.jsonl")
    
    layer_ids = []
    with jsonlines.open(files_whatsup_a[0], "r") as f:
        for sample in f:
            layer_ids.append(sample["settings"]["layer_id"])
    
    dataset_names = [
        "whatsup_a",
        "whatsup_b",
        "cocoqa_1",
        "cocoqa_2",
        "gqa_1",
        "gqa_2",
        "vsr"
    ]
    dataset_files = [
        files_whatsup_a, 
        files_whatsup_b, 
        files_cocoqa_1, 
        files_cocoqa_2, 
        files_gqa_1, 
        files_gqa_2, 
        files_vsr
    ]
    
    out_dataset_ids = []
    # out_dataset_ids = [2]  # whatsup_a, whatsup_b, cocoqa_1, cocoqa_2, gqa_1, gqa_2, vsr
    dataset_acc_list = [[] for _ in range(len(dataset_files))]
    for i, files in enumerate(dataset_files):
        for file in files:
            accs = []
            with jsonlines.open(file, "r") as f:
                for sample in f:
                    accs.append(sample["acc"])
            dataset_acc_list[i].append(accs)
        if not all(len(acc_list) == len(layer_ids) for acc_list in dataset_acc_list[i]):
            out_dataset_ids.append(i)

    dataset_accs = [[] for _ in range(len(dataset_files))]
    for i, accs_list in enumerate(dataset_acc_list):
        if i in out_dataset_ids:
            continue
        print(f"dataset {dataset_names[i]}: {len(accs_list)} groups of data")
        dataset_accs[i] = np.mean(accs_list, axis=0)
        # get the max
        
    figure_dir = root_dir / "figures"
    save_path = os.path.join(figure_dir, f"delete_vit_layer-{model_name}-test_directions_vlm.pdf")

    plt.figure(figsize=(7, 5.5))
    plt.axhline(y=0.25, color='black', linestyle='--', alpha=0.55, label="Random guess (1/4)")
    plt.axhline(y=0.5, color='black', linestyle='-.', alpha=0.55, label="Random guess (1/2)")
    current_yticks = list(plt.yticks()[0])
    if not all ([y_line in current_yticks for y_line in [0.25, 0.5]]):
        current_yticks.extend([0.25, 0.5])
        plt.yticks(sorted(current_yticks))
    
    # colors = ['#0090b3', "blue", "gold", "orange", "limegreen", "green", "peru"]
    colors = ['#0090b3', "blue", "gold", "orange", "limegreen", "green", "saddlebrown"]
    labels = ["What's Up (Subset A)", "What's Up (Subset B)", "COCO-QA-Spatial (one obj)", "COCO-QA-Spatial (two obj)", "GQA-Spatial (one obj)", "GQA-Spatial (two obj)", "VSR"]
    for i in range(len(dataset_names)):
        if i in out_dataset_ids:
            continue
        plt.plot(
            layer_ids,
            dataset_accs[i],
            color=colors[i],
            label=labels[i],
            # alpha=0.55,
            # marker='o',
            # markersize=5,
            # markeredgecolor=colors[i],
            # markeredgewidth=1.5
        )

    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel("Accuracy", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    # ax.spines["left"].set_color((0.3, 0.3, 0.3))
    # ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    # ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 1))
    
    plt.savefig(save_path, bbox_inches='tight')

def plot_vit_direction_left_vs_on(model_name):
    res_whatsup_dir = root_dir / "eval/WhatsUp/results/test_vit_directions_vlm"
    res_whatsup_dir = os.path.join(res_whatsup_dir, f"{model_name}")
    
    files_whatsup_A_left = glob.glob(f"{res_whatsup_dir}/final_whatsup_a_left_right_*.jsonl")
    files_whatsup_A_on = glob.glob(f"{res_whatsup_dir}/final_whatsup_a_on_under_*.jsonl")
    files_whatsup_B_behind = glob.glob(f"{res_whatsup_dir}/final_whatsup_b_behind_*.jsonl")
    files_whatsup_B_left = glob.glob(f"{res_whatsup_dir}/final_whatsup_b_left_*.jsonl")
    
    layer_ids = []
    with jsonlines.open(files_whatsup_A_left[0], "r") as f:
        for sample in f:
            layer_ids.append(sample["settings"]["layer_id"])
    
    dataset_names = [
        "whatsup_on_under",
        "whatsup_behind_in_front_of",
        "whatsup_left_right",
        "whatsup_left_right"
    ]
    dataset_files = [
        files_whatsup_A_on,
        files_whatsup_B_behind,
        files_whatsup_A_left,
        files_whatsup_B_left
    ]
    
    out_dataset_ids = []
    # out_dataset_ids = [2]  # whatsup_a, whatsup_b, cocoqa_1, cocoqa_2, gqa_1, gqa_2, vsr
    dataset_acc_list = [[] for _ in range(len(dataset_files))]
    for i, files in enumerate(dataset_files):
        for file in files:
            accs = []
            with jsonlines.open(file, "r") as f:
                for sample in f:
                    accs.append(sample["acc"])
            dataset_acc_list[i].append(accs)
        if not all(len(acc_list) == len(layer_ids) for acc_list in dataset_acc_list[i]):
            out_dataset_ids.append(i)

    dataset_accs = [[] for _ in range(len(dataset_files))]
    for i, accs_list in enumerate(dataset_acc_list):
        if i in out_dataset_ids:
            continue
        print(f"dataset {dataset_names[i]}: {len(accs_list)} groups of data")
        dataset_accs[i] = np.mean(accs_list, axis=0)
        # get the max
        
    figure_dir = root_dir / "figures/delete_vit_layer-test_directions-left_vs_on"
    save_dir = os.path.join(figure_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"delete_vit_layer-{model_name}-test_directions_left_vs_on.pdf")

    plt.figure(figsize=(7, 5.5))
    plt.axhline(y=0.25, color='black', linestyle='--', alpha=0.55, label="Random guess (1/4)")
    # plt.axhline(y=0.5, color='black', linestyle='-.', alpha=0.55, label="Random guess (1/2)")
    current_yticks = list(plt.yticks()[0])
    if not all ([y_line in current_yticks for y_line in [0.25, 0.5]]):
        current_yticks.extend([0.25, 0.5])
        plt.yticks(sorted(current_yticks))
    
    # colors = ['#0090b3', "blue", "gold", "orange", "limegreen", "green", "peru"]
    colors = ["blue", "cornflowerblue", "green", "lightgreen"]
    labels = ["What's Up A (On / Under)", "What's Up B (Behind / In front of)", "What's Up A (Left / Right)", "What's Up B (Left / Right)"]
    for i in range(len(dataset_names)):
        if i in out_dataset_ids:
            continue
        plt.plot(
            layer_ids,
            dataset_accs[i],
            color=colors[i],
            label=labels[i],
            # alpha=0.55,
            # marker='o',
            # markersize=5,
            # markeredgecolor=colors[i],
            # markeredgewidth=1.5
        )
        std_error = np.std(dataset_acc_list[i], axis=0) / np.sqrt(len(dataset_acc_list[i]))
        plt.fill_between(
            layer_ids,
            dataset_accs[i] - std_error,
            dataset_accs[i] + std_error,
            color=colors[i],
            alpha=0.1,
        )

    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel("Accuracy", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.spines["bottom"].set_color((0.3, 0.3, 0.3))
    # ax.spines["left"].set_color((0.3, 0.3, 0.3))
    # ax.tick_params(axis='x', colors=(0.3, 0.3, 0.3))
    # ax.tick_params(axis='y', colors=(0.3, 0.3, 0.3))
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_color((0.3, 0.3, 0.3))
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 1))
    
    plt.savefig(save_path, bbox_inches='tight')

def plot_vit_direction_separate(model_name):
    res_whatsup_dir = root_dir / "eval/WhatsUp/results/test_vit_directions_vlm"
    res_whatsup_dir = os.path.join(res_whatsup_dir, f"{model_name}")
    
    files_whatsup_A_left = glob.glob(f"{res_whatsup_dir}/final_whatsup_a_left_right_*.jsonl")  # √
    files_whatsup_A_on = glob.glob(f"{res_whatsup_dir}/final_whatsup_a_on_*.jsonl")
    files_whatsup_A_under = glob.glob(f"{res_whatsup_dir}/final_whatsup_a_under_*.jsonl")  # √
    files_whatsup_B_left = glob.glob(f"{res_whatsup_dir}/final_whatsup_b_left_*.jsonl")  # √
    files_whatsup_B_in_front_of = glob.glob(f"{res_whatsup_dir}/final_whatsup_b_in_front_of_*.jsonl")  # √
    files_whatsup_B_behind = glob.glob(f"{res_whatsup_dir}/final_whatsup_b_behind_*.jsonl")
    
    files_whatsup_A_on = [file for file in files_whatsup_A_on if len(file.strip(res_whatsup_dir).strip("final_whatsup_a_").split("_")) == 2]
    files_whatsup_B_behind = [file for file in files_whatsup_B_behind if len(file.strip(res_whatsup_dir).strip("final_whatsup_b_").split("_")) == 2]
    
    print("files_whatsup_A_on", files_whatsup_A_on)
    print("files_whatsup_A_under", files_whatsup_A_under)
    print("files_whatsup_B_behind", files_whatsup_B_behind)
    print("files_whatsup_B_in_front_of", files_whatsup_B_in_front_of)
    print("files_whatsup_A_left", files_whatsup_A_left)
    print("files_whatsup_B_left", files_whatsup_B_left)
    
    layer_ids = []
    with jsonlines.open(files_whatsup_A_left[0], "r") as f:
        for sample in f:
            layer_ids.append(sample["settings"]["layer_id"])
    
    dataset_names = [
        "whatsup_a_on",
        "whatsup_a_under",
        "whatsup_b_behind",
        "whatsup_b_in_front_of",
        "whatsup_a_left_right",
        "whatsup_b_left_right"
    ]
    dataset_files = [
        files_whatsup_A_on,
        files_whatsup_A_under,
        files_whatsup_B_behind,
        files_whatsup_B_in_front_of,
        files_whatsup_A_left,
        files_whatsup_B_left
    ]
    
    out_dataset_ids = []
    # out_dataset_ids = [2]  # whatsup_a, whatsup_b, cocoqa_1, cocoqa_2, gqa_1, gqa_2, vsr
    dataset_acc_list = [[] for _ in range(len(dataset_files))]
    for i, files in enumerate(dataset_files):
        for file in files:
            accs = []
            with jsonlines.open(file, "r") as f:
                for sample in f:
                    accs.append(sample["acc"])
            dataset_acc_list[i].append(accs)
        if not all(len(acc_list) == len(layer_ids) for acc_list in dataset_acc_list[i]):
            out_dataset_ids.append(i)

    dataset_accs = [[] for _ in range(len(dataset_files))]
    for i, accs_list in enumerate(dataset_acc_list):
        if i in out_dataset_ids:
            continue
        print(f"dataset {dataset_names[i]}: {len(accs_list)} groups of data")
        dataset_accs[i] = np.mean(accs_list, axis=0)
        # get the max
        
    figure_dir = root_dir / "figures/delete_vit_layer-test_directions-left_vs_on"
    save_dir = os.path.join(figure_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"delete_vit_layer-{model_name}-test_directions_directions_separate.pdf")

    plt.figure(figsize=(7, 5.5))
    plt.axhline(y=0.25, color='black', linestyle='--', alpha=0.55, label="Random guess (1/4)")
    # plt.axhline(y=0.5, color='black', linestyle='-.', alpha=0.55, label="Random guess (1/2)")
    current_yticks = list(plt.yticks()[0])
    if not all ([y_line in current_yticks for y_line in [0.25, 0.5]]):
        current_yticks.extend([0.25, 0.5])
        plt.yticks(sorted(current_yticks))
    
    # colors = ['#0090b3', "blue", "gold", "orange", "limegreen", "green", "peru"]
    colors = ["blue", "cornflowerblue", "green", "lightgreen", "orange", "peru"]
    labels = ["What's Up A (On)", "What's Up A (Under)", "What's Up B (Behind)", "What's Up B (In front of)",  "What's Up A (Left / Right)", "What's Up B (Left / Right)"]
    for i in range(len(dataset_names)):
        if i in out_dataset_ids:
            continue
        plt.plot(
            layer_ids,
            dataset_accs[i],
            color=colors[i],
            label=labels[i],
            # alpha=0.55,
            # marker='o',
            # markersize=5,
            # markeredgecolor=colors[i],
            # markeredgewidth=1.5
        )
        std_error = np.std(dataset_acc_list[i], axis=0) / np.sqrt(len(dataset_acc_list[i]))
        plt.fill_between(
            layer_ids,
            dataset_accs[i] - std_error,
            dataset_accs[i] + std_error,
            color=colors[i],
            alpha=0.1,
        )

    plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel("Accuracy", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # plt.grid(which="major", axis='y', linestyle='-', linewidth='0.5', color='0.75')
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color(mcolors.to_rgba('black', 1))
    
    plt.savefig(save_path, bbox_inches='tight')

def plot_vit_direction_left_vs_on_logits(model_name):
    
    exp_dir = root_dir / "eval/WhatsUp/results/test_vit_directions_left_vs_on"
    res_dir = os.path.join(exp_dir, f"{model_name}")
    files = os.listdir(res_dir)
    
    
    save_dir = root_dir / "figures/delete_vit_layer-test_directions-left_vs_on"
    save_dir = os.path.join(save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    for file_name in files:
        layer_ids = []
        dataset_name = ""
        with jsonlines.open(os.path.join(res_dir, file_name), "r") as f:
            for sample in f:
                dataset_name = sample["settings"]["dataset_name"]
                layer_ids.append(sample["settings"]["layer_id"])
        layer_ids = [layer_id + 1 for layer_id in layer_ids]
        
        if "behind" in dataset_name or "in_front_of" in dataset_name:
            labels = ["in front of", "behind", "left", "right"]
        else:
            labels = ["on", "under", "left", "right"]
        probs = defaultdict(list)
        with jsonlines.open(os.path.join(res_dir, file_name), "r") as f:
            for line in f:
                for word in labels:
                    key = "in" if word == "in front of" else word
                    probs[word].append(np.array(line["probs"][key]))
        
        plt.figure()  # figsize=(7, 5.5)
        colors = [mcolors.to_hex(plt.cm.viridis(i / len(labels))) for i in range(len(labels))]
        for i, label in enumerate(labels):
            mean = np.mean(probs[label], axis=1)
            std_error = np.std(probs[label], axis=1) / np.sqrt(len(probs[label]))
            plt.plot(
                layer_ids,
                mean,
                color=colors[i],
                label=f'"{labels[i]}"',
            )
            plt.fill_between(
                layer_ids, 
                mean - std_error, 
                mean + std_error, 
                color=colors[i], 
                alpha=0.1, 
                # label=f'Standard Error for "{labels[i]}"'
            )

        plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
        plt.ylabel("Token Probability", color=mcolors.to_rgba('black', 1))
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        legend = plt.legend()
        for text in legend.get_texts():
            text.set_color(mcolors.to_rgba('black', 1))
        
        save_path = os.path.join(save_dir, f"{model_name}-{dataset_name}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(save_path)

def plot_vit_direction_intervene(model_name, dataset_name, ori_dir_id=None, curr_dir_id=None, positive=False):
    
    labels = ["left", "right", "behind", "in front of"]
    
    save_dir = work_dir / "figures/whatsup/results"
    exp_dir = save_dir / "test_vit_intervene_spatial" / model_name
    if ori_dir_id is not None and curr_dir_id is not None:
        res_path = exp_dir / f"result_{labels[ori_dir_id]}_to_{'_'.join(labels[curr_dir_id].split(' '))}.jsonl"
    else:
        res_path = exp_dir / "result.jsonl"
    if positive:
        # res_path = res_path.replace(".jsonl", "_positive.jsonl")
        res_path = res_path.with_name(res_path.stem + "_positive.jsonl")
    
    save_dir = root_dir / "figures/intervene_spatial_reasoning" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    alphas = []
    probs = defaultdict(list)  # {"left": [], ...}
    with jsonlines.open(res_path, "r") as f:
        for line in f:
            alphas = list(line["probs_intervene"].keys())
            for label in labels:
                sample_dir_vals = []
                key = "in" if label == "in front of" else label
                # sample_dir_vals.append(line["probs_normal"][key])
                for alpha in line["probs_intervene"].keys():
                    sample_dir_vals.append(line["probs_intervene"][alpha][key])  # {0: [], 0.5: [], ...}
                # assert len(sample_dir_vals) == 16
                probs[label].append(sample_dir_vals)
    
    # print(probs)
        
    plt.figure()
    # colors = [mcolors.to_hex(plt.cm.viridis(i / len(labels))) for i in range(len(labels))]
    # colors = colors[::-1]
    colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
    markers = ["o", "s", "D", "^"]
    for i, label in enumerate(labels):
        if positive and i != ori_dir_id:
            continue
        mean = np.mean(probs[label], axis=0)  # probs[label]: (num_samples, len(alphas))
        std_error = np.std(probs[label], axis=0) / np.sqrt(len(probs[label]))
        plt.plot(
            alphas,
            mean,
            color=colors[i],
            label=f'"{labels[i]}"',
            alpha=1, 
            marker=markers[i],
            markersize=6, 
            # markerfacecolor="none",
            # markeredgecolor=colors[i],
            markeredgewidth=1.5
        )
        plt.fill_between(
            alphas, 
            mean - std_error, 
            mean + std_error, 
            color=colors[i], 
            alpha=0.1, 
        )
    # draw line
    if not positive:
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.55, label=f"Random guess ({labels[ori_dir_id]} vs. {labels[curr_dir_id]})")

    plt.xlabel("Intervention intensity", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel("Token Probability", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    plt.legend()
    # legend = plt.legend()
    # for text in legend.get_texts():
    #     text.set_color(mcolors.to_rgba('black', 1))
    
    save_path = os.path.join(save_dir, f"{model_name}-{dataset_name}-{labels[ori_dir_id]}_to_{'_'.join(labels[curr_dir_id].split(' '))}.pdf")
    if positive:
        save_path = ''.join(save_path.split(".")[:-1]) + "_positive.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(save_path)

def plot_vit_direction_intervene_positive(model_name, dataset_name, intervene_groups=None):
    """ 
    intervene_group: [[0, 1], [2, 3], ...] means intervene in 0 with 1, 2 with 3
    """
    labels = ["left", "right", "behind", "in front of"]
    
    save_dir = root_dir / "eval/WhatsUp/results"
    exp_dir = os.path.join(save_dir, "test_vit_intervene_spatial", f"{model_name}")
    
    fig_save_dir = root_dir / "figures/intervene_spatial_reasoning"
    fig_save_dir = os.path.join(fig_save_dir, f"{model_name}")
    os.makedirs(fig_save_dir, exist_ok=True)
    
    plt.figure()
    colors = ["#3d83bf", "#199c37", "#aab911", "#2b702f"]
    markers = ["o", "s", "D", "^"]
    for group in intervene_groups:
        ori_dir_id, curr_dir_id = group
        res_path = os.path.join(exp_dir, f"result_{labels[ori_dir_id]}_to_{'_'.join(labels[curr_dir_id].split(' '))}_positive.jsonl")
        alphas = []
        probs = defaultdict(list)  # {"left": [], ...}
        with jsonlines.open(res_path, "r") as f:
            for line in f:
                alphas = list(line["probs_intervene"].keys())
                for label in labels:
                    sample_dir_vals = []
                    key = "in" if label == "in front of" else label
                    # sample_dir_vals.append(line["probs_normal"][key])
                    for alpha in line["probs_intervene"].keys():
                        sample_dir_vals.append(line["probs_intervene"][alpha][key])  # {0: [], 0.5: [], ...}
                    # assert len(sample_dir_vals) == 16
                    probs[label].append(sample_dir_vals)
        label = labels[ori_dir_id]
        mean = np.mean(probs[label], axis=0)
        std_error = np.std(probs[label], axis=0) / np.sqrt(len(probs[label]))
        plt.plot(
            alphas,
            mean,
            color=colors[ori_dir_id],
            label=f'"{labels[ori_dir_id]}"',
            alpha=1, 
            marker=markers[ori_dir_id],
            markersize=6, 
            markerfacecolor="none",
            # markeredgecolor=colors[i],
            markeredgewidth=1.5
        )
        plt.fill_between(
            alphas, 
            mean - std_error, 
            mean + std_error, 
            color=colors[ori_dir_id], 
            alpha=0.1, 
        )
    
    plt.xlabel("Intervention intensity", loc="right", color=mcolors.to_rgba('black', 1))
    plt.ylabel("Token Probability", color=mcolors.to_rgba('black', 1))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
    
    plt.legend()
    save_path = os.path.join(fig_save_dir, f"{model_name}-{dataset_name}-intervene_positive.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    print(save_path)

def plot_direction_vector_cluster_res_layerwise(model_name, tsne=True):
    
    save_dir = root_dir / "eval/WhatsUp/results"
    exp_dir = os.path.join(save_dir, "test_vit_directions_relation_layerwise", f"{model_name}")
    
    cluster_name = "t-SNE" if tsne else "PCA"
    res_path = root_dir / f"eval/WhatsUp/results/test_vit_directions_relation_layerwise/{model_name}/cluster_res_{cluster_name}.jsonl"
    
    data = None
    with jsonlines.open(res_path, "r") as f:
        for line in f:
            data = line
            break
    silhouette_score_list = data["silhouette_score"]
    davies_bouldin_score_list = data["davies_bouldin_score"]
    calinski_harabasz_score_list = data["calinski_harabasz_score"]
    adjusted_rand_score_list = data["adjusted_rand_score"]
    orthogonality_dist_list = data["visual_geometry_score"]
    
    cluster_data = [silhouette_score_list, davies_bouldin_score_list, calinski_harabasz_score_list, adjusted_rand_score_list, orthogonality_dist_list]
    metric_label = ["Silhouette Score", "Davies Bouldin Score", "Calinski Harabasz Score", "Adjusted Rand Score", "Visual Geometry Score"]
    colors = ["blue", "blue", "blue", "blue", "blue"]
    markers = ["o", "o", "o", "o", "o"]
    layer_ids = list(range(len(silhouette_score_list)))
    layer_ids = [i + 1 for i in layer_ids]
    for i in range(len(metric_label)):
        print(i)
        mean = np.mean(cluster_data[i], axis=1)
        std_error = np.std(cluster_data[i], axis=1) / np.sqrt(len(cluster_data[i][1]))
        plt.figure()
        plt.plot(
            layer_ids,
            mean,
            color=colors[i],
            label=f"{metric_label[i]}",
            alpha=1, 
            marker=markers[i],
            markersize=6, 
            markerfacecolor="none",
            markeredgewidth=1.5
        )
        plt.fill_between(
            layer_ids, 
            mean - std_error, 
            mean + std_error, 
            color=colors[i], 
            alpha=0.1, 
        )
    
        plt.xlabel("Layer ID", loc="right", color=mcolors.to_rgba('black', 1))
        plt.ylabel(f"{metric_label[i]}", color=mcolors.to_rgba('black', 1))
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.grid(which="both", linestyle='-', linewidth='0.5', color='0.75')
        
        # plt.legend()
        save_path = os.path.join(exp_dir, f"cluster_res_{cluster_name}_{metric_label[i]}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        print(save_path)

def plot_vit_sim_violin_plot(model_name):
    res_path = os.path.join(root_dir / "eval/WhatsUp/results/test_vit_direction_vit_sim", model_name, "res.jsonl")
    sims_cls = []
    sims_satellite_before_connector = []
    sims_nucleus_before_connector = []
    sims_satellite_after_connector = []
    sims_nucleus_after_connector = []
    with jsonlines.open(res_path, 'r') as f:
        for line in f:
            if line["before_connector"]:
                if "sims_cls" in line:
                    sims_cls = line["sims_cls"]
                elif "sims_satellite" in line:
                    sims_satellite_before_connector = line["sims_satellite"]
                elif "sims_nucleus" in line:
                    sims_nucleus_before_connector = line["sims_nucleus"]
                else:
                    pass
            else:
                if "sims_satellite" in line:
                    sims_satellite_after_connector = line["sims_satellite"]
                elif "sims_nucleus" in line:
                    sims_nucleus_after_connector = line["sims_nucleus"]
                else:
                    pass
    group_names = ['sim-[cls]', 'sim-satellite after connector', 'sim-nucleus after connector', 'sim-satellite before connector', 'sim-nucleus before connector']
    df = pd.DataFrame({
        'Group': np.repeat(group_names, len(sims_cls)),
        'Values': sims_cls + sims_satellite_after_connector + sims_nucleus_after_connector + sims_satellite_before_connector + sims_nucleus_before_connector
    })
    color = "Greens"  # pastel, "Greens"
    ax = sns.violinplot(x='Group', y='Values', data=df, inner='box', palette=color, hue="Group", linewidth=0.6)
    # ax.set_xticklabels([''] * len(group_names))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.xaxis.set_ticks([])
    
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=sns.color_palette(color)[i], label=group_names[i]) for i in range(len(group_names))]
    plt.legend(handles=legend_patches, framealpha=0.5, fontsize=9)
    
    save_path = root_dir / "figures/vit_sims_{model_name}.pdf"
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    
    def test_direction_vlm():
        device = "cuda:0"
        model_name = "qwen2_5_vl"  # qwen2_5_vl llava1_5_7b llava1_6_mistral_7b  internvl2_5_8b
        dataset_name = "whatsup_b"  # "whatsup_a" "whatsup_b" "cocoqa_1"  "cocoqa_2"  "gqa_1"  "gqa_2" "vsr" "GQA", "whatsup_a_left_right", "whatsup_a_on_under", whatsup_b_behind_in_front_of, whatsup_b_left_right
        tag = "0"
        batch_size = 4
        shuffle_image_tokens = True
        test_vit_direction_vlm(device, model_name, dataset_name, tag, batch_size, shuffle_image_tokens=shuffle_image_tokens)  # intern: 30000M (bsz=2)
    
    def test_direction_left_vs_on():
        device = "cuda:3"
        model_name = "llava1_5_7b"  # qwen2_5_vl qwen2_vl_2b llava1_5_7b llava1_6_mistral_7b  internvl2_5_8b
        dataset_name = "whatsup_b_left"  # "whatsup_a" "whatsup_b" "cocoqa_1"  "cocoqa_2"  "gqa_1"  "gqa_2" "vsr" "GQA", "whatsup_a_left_right", "whatsup_a_on_under"
        tag = "0"
        batch_size = 2
        test_vit_direction_left_vs_on(device, model_name, dataset_name, tag, batch_size)  # intern: 30000M (bsz=2)
    
    def test_direction_clip():
        device = "cuda:0"
        model_name = "ViT-L/14@336px"  # ViT-L/14@336px  ViT-B/32
        dataset_name = "whatsup_a"
        test_vit_direction_clip(device, model_name, dataset_name)
    
    def get_relations():
        device = "cuda:0"
        model_name = "qwen2_5_vl"  # qwen2_vl_2b qwen2_5_vl_3b qwen2_5_vl llava1_5_7b llava1_6_mistral_7b  internvl2_5_8b  
        dataset_name = "whatsup_b"  # "whatsup_a" "whatsup_b"
        color = "viridis"
        delete_pos_embed=False
        add_repr = True
        draw = True
        get_relation_representations(model_name, dataset_name, device, before_connector=False, add_repr=add_repr, tsne=False, draw=draw, color=color, objects=None, delete_pos_embed=delete_pos_embed)  # ,objects=None
    
    def get_relations_layerwise():  # 17446M for qwen2_5_vl
        device = "cuda:0"
        model_name = "internvl2_5_8b"  # qwen2_5_vl qwen2_5_vl_3b qwen2_vl_2b llava1_5_7b internvl2_5_8b
        dataset_name = "whatsup_b"  # "whatsup_a" "whatsup_b"
        use_thumbnail = True
        get_relation_representations_layerwise(model_name, dataset_name, device, before_connector=False, objects=None, tsne=False, delete_pos_embed=False, use_thumbnail=use_thumbnail)  # ,objects=None
    
    def get_relations_one_object():  # not direction vectors, but the direction info in one object
        device = "cuda:3"
        model_name = "llava1_5_7b"  # qwen2_5_vl qwen2_5_vl_3b qwen2_vl_2b llava1_5_7b llava1_6_mistral_7b  internvl2_5_8b
        dataset_name = "whatsup_b"  # "whatsup_a" "whatsup_b"
        color = "cividis"
        get_relation_representations_one_object(model_name, dataset_name, device, before_connector=False, objects=None, tsne=True, color=color)  # ,objects=None
        
    # test_direction_vlm()
    # test_direction_left_vs_on()
    # test_direction_clip()
    
    # plot_vit_direction_vlm("internvl2_5_8b")
    # plot_vit_direction_separate("internvl2_5_8b")
    
    # plot_vit_direction_left_vs_on("internvl2_5_8b")
    # plot_vit_direction_left_vs_on_logits("internvl2_5_8b")  # "qwen2_5_vl"  "llava1_5_7b"  "internvl2_5_8b"
    
    # get_relations()  # Dark2
    # get_relations_layerwise()
    # get_relations_one_object()
    
    # explore_relation_representations_samplewise("internvl2_5_8b", device="cuda:2", normalize=True, cluster_after_pca=True)  # "qwen2_5_vl"  "llava1_5_7b"  "internvl2_5_8b"
    # explore_relation_representations_samplewise("llava1_5_7b", device="cuda:1", objects=None, normalize=True, cluster_after_pca=True, draw=True)  # "qwen2_5_vl"  "llava1_5_7b"
    # plot_direction_vector_cluster_res_layerwise("qwen2_5_vl", tsne=False)  # "qwen2_5_vl"
    
    # intervene_in_spatial_reasoning("qwen2_5_vl", "whatsup_b", device="cuda:1", ori_dir_id=0, curr_dir_id=1)  # "qwen2_5_vl"  "llava1_5_7b"
    # intervene_in_spatial_reasoning("qwen2_5_vl", "whatsup_b", device="cuda:1", ori_dir_id=2, curr_dir_id=3, positive=True)  # "qwen2_5_vl"  "llava1_5_7b"
    plot_vit_direction_intervene("qwen2_5_vl", "whatsup_b", ori_dir_id=0, curr_dir_id=2, positive=False)  # "qwen2_5_vl"  "llava1_5_7b"
    # plot_vit_direction_intervene_positive("qwen2_5_vl", "whatsup_b", intervene_groups=[[0, 1], [2, 3]])  # "qwen2_5_vl"  "llava1_5_7b"
    
    # check_orthogonality("qwen2_5_vl", device="cuda:1", objects=None)  # "qwen2_5_vl"  "llava1_5_7b"  ["bowl", "candle"]
    # check_orthogonality("qwen2_5_vl", device="cuda:1", objects=["bowl", "candle"])  # "qwen2_5_vl"  "llava1_5_7b"  ["bowl", "candle"]
    
    # check_direction_language_alignment("qwen2_5_vl", "whatsup_b", device="cuda:3")  # "qwen2_5_vl"  "llava1_5_7b"
    # check_direction_spatial_reasoning_attention("qwen2_5_vl", "whatsup_b", device="cuda:3")  # "qwen2_5_vl"  "llava1_5_7b"
    
    # explore_1d_pos_embed_visual_geometry("internvl2_5_8b", device="cuda:2")  # "llava1_5_7b"  "internvl2_5_8b"
    # explore_1d_pos_embed_visual_geometry("llava1_5_7b", device="cuda:2", row=False)  # "llava1_5_7b"  "internvl2_5_8b"
    # explore_1d_pos_embed_direction_vectors("internvl2_5_8b", device="cuda:1")
    # explore_1d_pos_embed_decay("internvl2_5_8b", "vsr", device="cuda:1", batch_size=4)  # "llava1_5_7b"  "internvl2_5_8b"
    
    # explore_relation_in_rope("qwen2_vl_2b", "whatsup_b", "cuda:2")
    
    # satellite, nucleus, background
    # ["satellite"], ["background"]
    # ["nucleus"], ["background"]
    # ["satellite"], ["nucleus"]
    # ["nucleus"], ["satellite"]
    # ["satellite", "nucleus"], ["background", "background"]
    # erase_object_in_llm("llava1_5_7b", "whatsup_b", device="cuda:0", replaced_objects=["satellite", "nucleus"], new_objects=["background", "background"])  # "qwen2_5_vl" "qwen2_vl_2b"  "llava1_5_7b"
    # erase_object_in_llm("internvl2_5_8b", "whatsup_b", device="cuda:0", replaced_objects=["nucleus"], new_objects=["satellite"], erase_thumbnail=True)  # "qwen2_5_vl" "qwen2_vl_2b"
    
    # check_relation_pair_similarity("qwen2_5_vl", "whatsup_b", device="cuda:0", before_connector=True)  # "qwen2_5_vl"  "llava1_5_7b"
    # check_relation_pair_similarity("llava1_5_7b", "whatsup_b", device="cuda:2", before_connector=True)  # "qwen2_5_vl"  "llava1_5_7b"
    # plot_vit_sim_violin_plot("llava1_5_7b")
    # llava1.5 before connector
    # Avg sim cls: 0.958090788398693
    # Avg sim satellite: 0.8810380923202615
    # Avg sim nucleus: 0.8212124693627451
    # llava1.5 after connector
    # Avg sim cls: nan
    # Avg sim satellite: 0.9088477839052288
    # Avg sim nucleus: 0.8479051776960784
    
    # for i in range(16):
    #     # if i != 3:
    #     #     continue
    #     explore_rope_attention_by_dimension_group(
    #         "qwen2_vl_7b",  # "qwen2_vl_2b", "qwen2_5_vl_3b"
    #         "whatsup_b", 
    #         device="cuda:0", 
    #         only_center=False, 
    #         normalize_together=False, 
    #         num_samples=100, 
    #         head_id=i,
    #     )
    
    # explore_rope_attention_by_dimension_group(
    #     "qwen2_vl_7b",  # "qwen2_vl_2b", "qwen2_5_vl_3b"
    #     "whatsup_b", 
    #     device="cuda:0", 
    #     only_center=False, 
    #     normalize_together=False, 
    #     num_samples=100, 
    #     head_id=None,
    #     # direction="fb",  # lr, fb
    #     accumulated_attn=False,
    # )

    # explore_rope_by_sensitivity_score("qwen2_vl_2b", "whatsup_b", device="cuda:0", activation_name="attn_output")

    # =================================================================================================== Rope Scaling
    # model_name = "qwen2_vl_2b"  # qwen2_5_vl, qwen2_5_vl_3b, qwen2_vl_2b, llava1_5_7b, internvl2_5_8b
    # accs = []
    # for i in range(1, 6):
    #     acc = test_rope_scaling(
    #         model_name=model_name,  # qwen2_5_vl qwen2_5_vl_3b qwen2_5_vl_3b qwen2_vl_2b, llava1_5_7b, internvl2_5_8b
    #         dataset_name="whatsup_b",  # "GQA", "whatsup_b", "gqa_no_spatial", "whatsup_b_left_right", "whatsup_b_behind_in_front_of"
    #         device="cuda:0",
    #         tag=str(i),
    #         data_num=1000,
    #         batch_size=8,
    #         random=False,
    #         scaling_type="poly",
    #         alpha=1.0,
    #         gamma=2.0,
    #         beta=0.1,
    #         base=5000,
    #         poly_p=8,
    #         poly_alpha=99,
    #         sig_alpha=99,
    #         sig_mid_point=0.5,
    #         sig_k=20.0,
    #     )
    #     accs.append(acc)
    # try:
    #     avg_acc = sum(accs) / len(accs)
    # except:
    #     avg_acc = 0.0
    # print(f"accs ({model_name}): avg_acc: {avg_acc}, {accs}")
    
    
    #  grid search
    # model_name = "qwen2_vl_7b"  # qwen2_5_vl, qwen2_5_vl_3b, qwen2_vl_2b, llava1_5_7b, internvl2_5_8b
    # dataset_name = "cocoqa_1"
    # save_dir = '/raid_sdd/lyy/Interpretability/lyy/mm/figures/test_rope_scaling'
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"rope_scaling_grid_search_{model_name}_{dataset_name}.jsonl")
    # res = []
    # poly_ps = [6, 8, 10, 12, 14, 16, 18, 20, 24]
    # poly_alphas = [9, 19, 49, 79, 99, 119, 149, 199]
    # for poly_p in poly_ps:
    #     for poly_alpha in poly_alphas:
    #         accs = []
    #         loop_num = 6
    #         if dataset_name == "vsr":
    #             loop_num = 1
    #         if "cocoqa" in dataset_name:
    #             loop_num = 1
    #         for i in range(loop_num):
    #             acc = test_rope_scaling(
    #                 model_name=model_name,  # qwen2_5_vl qwen2_5_vl_3b qwen2_5_vl_3b qwen2_vl_2b, llava1_5_7b, internvl2_5_8b
    #                 dataset_name=dataset_name,  # "GQA", "whatsup_b", "gqa_no_spatial", "whatsup_b_left_right", "whatsup_b_behind_in_front_of"
    #                 device="cuda:3",
    #                 tag=str(i),
    #                 data_num=1000,
    #                 batch_size=8,
    #                 random=False,
    #                 scaling_type="poly",
    #                 alpha=1.0,
    #                 gamma=2.0,
    #                 beta=0.1,
    #                 base=5000,
    #                 poly_p=poly_p,
    #                 poly_alpha=poly_alpha,
    #                 sig_alpha=99,
    #                 sig_mid_point=0.5,
    #                 sig_k=20.0,
    #             )
    #             accs.append(acc)
    #         try:
    #             avg_acc = sum(accs) / len(accs)
    #         except:
    #             avg_acc = 0.0
    #         print(f"accs ({model_name}): avg_acc: {avg_acc}, {accs}")
    #         _res = {
    #             "model_name": model_name,
    #             "dataset_name": dataset_name,
    #             "poly_p": poly_p,
    #             "poly_alpha": poly_alpha,
    #             "avg_acc": avg_acc,
    #             "accs": accs
    #         }
    #         with jsonlines.open(save_path, mode='a') as f:
    #             f.write(_res)
    #         res.append(_res)
    # for item in res:
    #     print(item)
