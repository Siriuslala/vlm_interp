# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
import platform
import functools
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qwen_vl_finetune.qwenvl.train.trainer import replace_qwen2_vl_attention_class

import torch.nn as nn
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from model.qwen2_vl_rope_scaling import Qwen2VLForConditionalGeneration_rope_scaling, Qwen2_5_VLForConditionalGeneration_rope_scaling

from qwen_vl_finetune.qwenvl.data.data_qwen import make_supervised_data_module
from qwen_vl_finetune.qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwen_vl_finetune.qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training

os.environ["WANDB_MODE"] = "offline"

logging.basicConfig(level=logging.INFO)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

def get_lora_model(model, model_args: ModelArguments, training_args: TrainingArguments):
    """Converts the base model to a PEFT LoRA model."""
    rank0_print("Setting up LoRA for PEFT...")
    # Prepare model for k-bit training if quantization is used
    if training_args.load_in_8bit or training_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if model_args.lora_target_modules:
        target_modules = model_args.lora_target_modules.split(',')
    else:
        # A simple heuristic to find all linear layers
        target_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and "proj" in name]

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    
    rank0_print("Trainable parameters with LoRA:")
    model.print_trainable_parameters()
    
    return model

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # MODIFIED: Add device_map for quantized loading
    device_map = {"": "cuda:" + str(local_rank)} if (training_args.load_in_8bit or training_args.load_in_4bit) else None
    torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if (training_args.load_in_4bit or training_args.load_in_8bit) else None)
    
    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            # MODIFIED: Add quantization and device_map
            load_in_8bit=training_args.load_in_8bit,
            load_in_4bit=training_args.load_in_4bit,
            device_map=device_map,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        if model_args.rope_scaling:
            model_class = Qwen2VLForConditionalGeneration_rope_scaling
        else:
            model_class = Qwen2VLForConditionalGeneration
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            # MODIFIED: Add quantization and device_map
            load_in_8bit=training_args.load_in_8bit,
            load_in_4bit=training_args.load_in_4bit,
            device_map=device_map,
            # ignore_mismatched_sizes=True,
        )
        if model_args.rope_scaling:
            if model_args.scaling_type == "poly":
                model.visual.rotary_pos_emb.scaling_type = "poly"
                model.visual.rotary_pos_emb.poly_alpha = model_args.poly_alpha
                model.visual.rotary_pos_emb.poly_p = model_args.poly_p
            elif model_args.scaling_type == "sigmoid":
                model.visual.rotary_pos_emb.scaling_type = "sigmoid"
                model.visual.rotary_pos_emb.sig_alpha = model_args.sig_alpha
                model.visual.rotary_pos_emb.sig_mid_point = model_args.sig_mid_point
                model.visual.rotary_pos_emb.sig_k = model_args.sig_k

        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Conditionally apply LoRA or full fine-tuning
    if model_args.use_lora:
        model = get_lora_model(model, model_args, training_args)        
        
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    trainer.train()
 
    for obj in trainer.state.log_history:
        logging.info(str(obj))


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
