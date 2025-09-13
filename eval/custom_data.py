""" 
Custom dataset.
"""

from typing import Any
import os
import random
import json

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

import sys
sys.path.append(str(root_dir))


class GQASquareImages():
    def __init__(self, image_dir, data_num=None, random_select=True) -> None:
        image_files = os.listdir(image_dir)
        image_ids = []
        self.data = []
        
        train_scene_path = data_dir / "GQA/scene/train_sceneGraphs.json"
        val_scene_path = data_dir / "GQA/scene/val_sceneGraphs.json"
        train_scene_data = json.load(open(train_scene_path, "r"))
        val_scene_data = json.load(open(val_scene_path, "r"))
        
        for file_name in image_files:
            image_id = file_name.split('.')[0]
            if not (image_id in train_scene_data or image_id in val_scene_data):
                continue
            image_ids.append(image_id)
            self.data.append({"image": os.path.join(image_dir, file_name)})
        if random_select:
            random.shuffle(self.data)
        if data_num:
            self.data = self.data[:data_num]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    