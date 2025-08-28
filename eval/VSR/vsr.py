from torch.utils.data import Dataset
import numpy as np
import os
import random
import json
import jsonlines
from PIL import Image
import torch

class VSRDataset(Dataset):
    
    def __init__(self, data_path, data_num=10000, random_select=False):
        
        self.data = []
        text_path, image_path = data_path
        with jsonlines.open(text_path, mode='r') as f:
            for line in f:
                img_path = os.path.join(image_path, line["image"])
                caption = line["caption"]
                answer = line["label"]
                relation = line["relation"]
                self.data.append({"caption": caption, "image": img_path, "answer": answer, "relation": relation})

        if random_select:
            random.shuffle(self.data)
        self.data = self.data[:data_num]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class VSRCollator:
    def __init__(self, processor, vision_process_func=None, model_name=None):
        self.processor = processor
        self.vision_process_func = vision_process_func
        self.model_name = model_name
    
    def __call__(self, samples):
        
        image_paths = [sample["image"] for sample in samples]
        prompts = []
        
        if self.processor is None:
            return None, samples
        
        for i, sample in enumerate(samples):
            instruction = f"""
            Please check if the following description is correct:\n\n
            Description: {sample["caption"]}\n\n
            Please answer with either "True" or "False".
            Your answer is:
            """
            prompts.append(instruction)
        # print(samples)
        # import pdb; pdb.set_trace()
        
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