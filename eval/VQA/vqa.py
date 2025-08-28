"""
Dateset for VQA. 
"""

from torch.utils.data import Dataset
import numpy as np
import os
import random
import json
import jsonlines
from PIL import Image
import torch
from tqdm import tqdm


class VQADataset(Dataset):
    
    def __init__(self, data_path, data_num=10000, random_select=False):
        
        self.data = []
        qn_path, ans_path, image_path = data_path
        qn_data = json.load(open(qn_path, 'r'))
        anno_data = json.load(open(ans_path, 'r'))
        questions = qn_data["questions"]
        annotations = anno_data["annotations"]
        
        for idx in tqdm(range(len(questions))):
            question = questions[idx]
            annotation = annotations[idx]
            question_id = question["question_id"]
            image_id = question["image_id"]
            image_id_in_name = str(image_id).zfill(12)
            image_name = f"COCO_val2014_{image_id_in_name}.jpg"
            img_path = os.path.join(image_path, image_name)
            if not os.path.exists(img_path):
                print(f"Image {img_path} does not exist, skipping.")
                continue
            qn = question["question"]
            answers = annotation["answers"]
            answer_list = [ans["answer"] for ans in answers]
            
            sample_data = {
                "question_id": question_id,
                "image": img_path,
                "question": qn,
                "answers": answer_list
            }
            
            self.data.append(sample_data)
        
        if random_select:
            random.shuffle(self.data)
        
        if data_num:
            self.data = self.data[:data_num]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class VQACollator:
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


# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import copy

class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotation_file == None and not question_file == None:
            print('loading VQA annotations and questions into memory...')
            dataset = json.load(open(annotation_file, 'r'))
            questions = json.load(open(question_file, 'r'))
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa =  {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            imgToQA[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
            qqa[ques['question_id']] = ques
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print("%s: %s" %(key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for given question types
                ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds 	  = imgIds    if type(imgIds)    == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],[])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['question_id'] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
                quesTypes (str array)   : get image ids for given question types
                ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds   = quesIds   if type(quesIds)   == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(quesIds) == 0:
                anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa],[])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['image_id'] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" %(self.qqa[quesId]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" %(ans['answer_id'], ans['answer']))

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))
        res.dataset['info'] = copy.deepcopy(self.questions['info'])
        res.dataset['task_type'] = copy.deepcopy(self.questions['task_type'])
        res.dataset['data_type'] = copy.deepcopy(self.questions['data_type'])
        res.dataset['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
        res.dataset['license'] = copy.deepcopy(self.questions['license'])

        print('Loading and preparing results...')
        # load results
        # anns    = json.load(open(resFile))
        import jsonlines
        anns = list(jsonlines.open(resFile, 'r'))
        
        assert type(anns) == list, 'results is not an array of objects'
        # annsQuesIds = [ann['question_id'] for ann in anns]
        # assert set(annsQuesIds) == set(self.getQuesIds()), \
        # 'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
        
        for ann in anns:
            quesId 			     = ann['question_id']
            if res.dataset['task_type'] == 'Multiple Choice':
                assert ann['answer'] in self.qqa[quesId]['multiple_choices'], 'predicted answer is not one of the multiple choices'
            qaAnn                = self.qa[quesId]
            ann['image_id']      = qaAnn['image_id'] 
            ann['question_type'] = qaAnn['question_type']
            ann['answer_type']   = qaAnn['answer_type']
        res.dataset['annotations'] = anns
        
        res.qn_ids = [ann['question_id'] for ann in anns]
        
        res.createIndex()
        return res

def evaluate_vqa(anno_path, qn_path, res_path, n=2):
    """
    Evaluate VQA results.
    :param vqa (VQA object)   : VQA object containing annotations
    :param vqaRes (VQA object): VQA result object
    :param n (int)            : precision of accuracy (number of places after decimal), default is 2
    :return: vqaEval (VQAEval object): VQAEval object containing evaluation results
    """
    from .vqa_eval import VQAEval
    vqa = VQA(anno_path, qn_path)
    vqaRes = vqa.loadRes(res_path, qn_path)
    vqaEval = VQAEval(vqa, vqaRes, n)
    qn_ids = vqaRes.qn_ids
    vqaEval.evaluate(qn_ids)
    acc = vqaEval.accuracy['overall']
    return acc