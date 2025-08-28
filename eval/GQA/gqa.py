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


class GQADataset(Dataset):
    
    def __init__(self, data_path, data_num=10000, random_select=False):
        
        self.data = []
        text_path, image_path = data_path
        if "jsonl" in text_path:
            f = jsonlines.open(text_path, 'r')
            self.data = [line for line in f]
        else:
            all_data = json.load(open(text_path, "r"))
            for qn_id, content in all_data.items():
                question = content["question"]
                imageID = content["imageId"]
                image = os.path.join(image_path, imageID + ".jpg")
                answer = content["answer"]
                self.data.append({"id": qn_id, "question": question, "image": image, "answer": answer})

        if random_select:
            random.shuffle(self.data)
            
        if data_num:
            self.data = self.data[:data_num]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class GQACollator:
    def __init__(self, processor, vision_process_func=None, model_name=None, QA_mode=True):
        self.processor = processor
        self.vision_process_func = vision_process_func
        self.model_name = model_name
        self.qa_mode = QA_mode
    
    def __call__(self, samples):
        
        image_paths = [sample["image"] for sample in samples]
        if self.qa_mode:
            prompts = [sample["question"] + "Please answer with a single word in lowercase." for sample in samples]
        else:
            prompts = ["describe the image and tell me what is the main object in the image" for _ in range(len(samples))]
        
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
        
        return inputs, samples

