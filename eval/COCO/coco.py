from torch.utils.data import Dataset
import numpy as np
import random
import json
import jsonlines
from PIL import Image
import torch

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

import sys
sys.path.append(str(root_dir))


class COCODataset(Dataset):
    
    def __init__(self, data_path, data_num=10000, random_select=False):
        
        self.data = []
        text_path, image_path = data_path
        data = json.load(open(text_path, 'r')) if text_path else None
        # print(data.keys())  # ['info', 'images', 'licenses', 'annotations']
        # print(data['annotations'][0])
        data = data['annotations']
        
        random.seed(14)  # Set a seed for reproducibility
        if random_select:
            random.shuffle(data)
        
        if data_num:
            data = data[:data_num]
        
        for item in data:
            id = item["id"]
            image_id = item["image_id"]
            image_id_in_name = str(image_id).zfill(12)
            image_name = f"COCO_train2014_{image_id_in_name}.jpg"

            img_path = os.path.join(image_path, image_name)
            if not os.path.exists(img_path):
                print(f"Image {img_path} does not exist, skipping.")
                continue
            
            # try:
            #     Image.open(img_path).convert('RGB')  # Check if image can be opened
            # except Exception as e:
            #     print(f"Error opening image {img_path}: {e}")
            #     continue
            caption = item["caption"]
            self.data.append({"id": id, "image_id": image_id, "image": img_path, "caption_gold": caption})
        print(f"Loaded {len(self.data)} samples from COCO dataset.")     

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class COCOCollator:
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
            instruction = "Describe this image as detailed as possible"
            prompts.append(instruction)
        
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

if __name__ == "__main__":
    # Example usage
    data_path = [
        data_dir / "Hallucination/coco/annotations/captions_train2014.json",
        data_dir / "Hallucination/coco/train2014"
    ]
    dataset = COCODataset(data_path, data_num=1000, random_select=True)
    collator = COCOCollator(processor=None, vision_process_func=None, model_name="llava")
    
   