"""
Dateset for GQA. 
"""

from torch.utils.data import Dataset
import numpy as np
import os
import random
import json
import jsonlines
from PIL import Image
import torch


class VisionDecoderDataset(Dataset):
    
    def __init__(self, split="train"):
        
        data_root = "/archive/private/liyueyan"
        GQA_path = os.path.join(data_root, "GQA", "images")
        COCO_path = os.path.join(data_root, "Hallucination", "coco", "train2014")
        TextVQA_path = os.path.join(data_root, "TextVQA", "data", "images")
        
        val_data = []
        val_dataset_path = "/home/liyueyan/Interpretability/mm/train_unembedding/data/val_dataset.jsonl"
        with jsonlines.open(val_dataset_path, "r") as f:
            for line in f:
                val_data.append({
                    "image": line["image"],
                    "question": line["question"],
                })

        train_data = []
        for file in os.listdir(COCO_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                train_data.append({
                    "image": os.path.join(COCO_path, file),
                    "question": "What is in this image?",
                })
        for file in os.listdir(TextVQA_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                train_data.append({
                    "image": os.path.join(TextVQA_path, file),
                    "question": "What is in this image?",
                })
        train_data = [item for item in train_data if item["image"] not in [data["image"] for data in val_data]]
        random.shuffle(train_data)

        self.data = train_data if split == "train" else val_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class VisionDecoderCollator:
    def __init__(self, processor, vision_process_func=None, model_name=None):
        self.processor = processor
        self.vision_process_func = vision_process_func
        self.model_name = model_name
    
    def __call__(self, samples):
        
        image_paths = [sample["image"] for sample in samples]
        prompts = [sample["question"] for sample in samples]
      
        if any(name in self.model_name for name in ["qwen"]):
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
            texts = [self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in batch_messages]
            
            all_images = []
            for msg in batch_messages:
                images, _ = self.vision_process_func(msg)
                if images:
                    all_images.extend(images)
                
            inputs = self.processor(
                text=texts,
                images=all_images if all_images else None,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        elif any(name in self.model_name for name in ["llava"]):
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
            texts = [self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in batch_messages]
            
            all_images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            inputs = self.processor(
                text=texts,
                images=all_images if all_images else None,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        elif any(name in self.model_name for name in ["intern"]):
            # question_length = [len(prompt) for prompt in prompts]
            pixel_values = [self.vision_process_func(image_path).to(torch.bfloat16) for image_path in image_paths]
            num_patches_list = [pixel_value.size(0) for pixel_value in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            inputs = {
                "questions": prompts,
                "pixel_values": pixel_values,
                "num_patches_list": num_patches_list,
                # "question_length": question_length,
            }
        else:
            raise ValueError("Model name not supported.")
        
        return inputs

