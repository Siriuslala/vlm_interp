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


class SQADataset(Dataset):
    
    def __init__(self, data_path, data_num=10000, random_select=False):
        
        self.data = []
        text_path = os.path.join(data_path, "test_data.jsonl")
        image_path = os.path.join(data_path, "test_images")
        with jsonlines.open(text_path, 'r') as f:
            for line in f:
                question_id = line["question_id"]
                question = line["question"]
                image = os.path.join(image_path, line["image"]) if line["image"] is not None else None
                choices = line["choices"]
                self.data.append({
                    "question_id": question_id,
                    "question": question, 
                    "image": image, 
                    "choices": choices,
                    "answer": ["A", "B", "C", "D", "E"][line["answer"]]
                })


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SQACollator:
    def __init__(self, processor, vision_process_func=None, model_name=None, QA_mode=True):
        self.processor = processor
        self.vision_process_func = vision_process_func
        self.model_name = model_name
        self.qa_mode = QA_mode
    
    def __call__(self, samples):
        
        image_paths = [sample["image"] for sample in samples]
        prompts = []
        for sample in samples:
            if self.qa_mode:
                question = sample["question"]
                choices = sample["choices"]
                if len(choices) == 2:
                    qn = "Please select the correct option and return only the letter of the option.\nYour choice is:" if "llava" in self.model_name else "Please answer with A or B.\nYour answer is:"
                    instruction = f"""
                    {question}\n
                    A: {choices[0]}\n
                    B: {choices[1]}\n
                    {qn}
                    """
                elif len(choices) == 3:
                    qn = "Please select the correct option and return only the letter of the option.\nYour choice is:" if "llava" in self.model_name else "Please answer with A, B or C.\nYour answer is:"
                    instruction = f"""
                    {question}\n
                    A: {choices[0]}\n
                    B: {choices[1]}\n
                    C: {choices[2]}\n
                    {qn}
                    """
                elif len(choices) == 4:
                    qn = "Please select the correct option and return only the letter of the option.\nYour choice is:" if "llava" in self.model_name else "Please answer with A, B, C or D.\nYour answer is:"
                    instruction = f"""
                    {question}\n
                    A: {choices[0]}\n
                    B: {choices[1]}\n
                    C: {choices[2]}\n
                    D: {choices[3]}\n
                    {qn}
                    """
                elif len(choices) == 5:
                    qn = "Please select the correct option and return only the letter of the option.\nYour choice is:" if "llava" in self.model_name else "Please answer with A, B, C, D or E.\nYour answer is:"
                    instruction = f"""
                    {question}\n
                    A: {choices[0]}\n
                    B: {choices[1]}\n
                    C: {choices[2]}\n
                    D: {choices[3]}\n
                    E: {choices[4]}\n
                    {qn}
                    """
                else:
                    raise ValueError("QA mode is not supported.")
            prompts.append(instruction)
        
        if any(name in self.model_name for name in ["qwen"]):
            batch_messages = []
            for img_path, prompt in zip(image_paths, prompts):
                if img_path is None:
                    msg = {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                else:
                    msg = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": prompt},
                        ]
                    }
                batch_messages.append([msg])
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
                if img_path is None:
                    msg = {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                else:
                    msg = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": prompt},
                        ]
                    }
                batch_messages.append([msg])

            texts = [self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in batch_messages]
            
            all_images = [Image.open(image_path).convert('RGB') for image_path in image_paths if image_path is not None]
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

