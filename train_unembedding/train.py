import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoModel,
    AutoTokenizer, 
    AutoProcessor
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, CosineAnnealingLR
from qwen_vl_utils import process_vision_info as process_vision_info_qwen

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

import sys
sys.path.append(str(root_dir))

from patch.monkey_patch import *
from model.unembedding import VisionTokenDecoder
from dataset import VisionDecoderDataset, VisionDecoderCollator

from tqdm import tqdm
import argparse
import jsonlines


MODEL_NAME_TO_PATH = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_vl_7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2_vl_2b": "Qwen/Qwen2-VL-2B-Instruct",
    "llava1_5_7b": "llava-hf/llava-1.5-7b-hf",  # "llava-hf/llava-1.5-7b-hf", "liuhaotian/llava-v1.5-7b"
    "internvl2_5_8b": "OpenGVLab/InternVL2_5-8B",
}

def load_data(model_name, processor, batch_size=8, split="train"):

    dataset = VisionDecoderDataset(split=split)
    if "qwen" in model_name:
        vision_process_func = process_vision_info_qwen
    else:
        vision_process_func = None
    data_collator = VisionDecoderCollator(processor=processor, vision_process_func=vision_process_func, model_name=model_name)
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

def calculate_distillation_loss(
    student_logits,  # (B, L, V)
    teacher_logits,  # (B, L, V)
    temperature,
    alpha,
):
    # soft Loss
    loss_soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Hard Loss
    teacher_predictions = teacher_logits.argmax(dim=-1) # (batch_size * num_vision_tokens)
    # student_logits: (B, T, V) -> (B*T, V)
    # teacher_predictions: (B, T) -> (B*T)
    loss_hard = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_predictions.view(-1)
    )

    # --- 3. Total loss ---
    loss = alpha * loss_soft + (1 - alpha) * loss_hard
    
    return loss

def calculate_distillation_loss_batch(
    student_logits,  # (B, L, V)
    teacher_logits,  # (B, L, V)
    loss_mask,       # (B, L)
    temperature,
    alpha,
    use_batch=True
):
    if use_batch:
        # --- 1. Hard Loss (Cross-Entropy) ---
        flat_student_logits = student_logits.view(-1, student_logits.size(-1)) # (B*L, V)
        flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1)) # (B*L, V)
        flat_mask = loss_mask.view(-1) # (B*L,)
        
        teacher_predictions = flat_teacher_logits.argmax(dim=-1)
        
        loss_hard_per_token = F.cross_entropy(
            flat_student_logits,
            teacher_predictions,
            reduction='none'
        )
        
        masked_loss_hard = loss_hard_per_token * flat_mask
        num_valid_tokens = flat_mask.sum()
        
        if num_valid_tokens > 0:
            loss_hard = masked_loss_hard.sum() / num_valid_tokens
        else:
            loss_hard = torch.tensor(0.0, device=student_logits.device)
            
        # --- 2. Soft Loss (KL Divergence) ---
        # reduction='none' -> (B*L, V)
        kl_div_per_token = F.kl_div(
            F.log_softmax(flat_student_logits / temperature, dim=-1),
            F.softmax(flat_teacher_logits / temperature, dim=-1),
            reduction='none'
        ).sum(dim=-1) # sum over vocab_size dim -> (B*L,)
        
        masked_loss_soft = kl_div_per_token * flat_mask
        
        if num_valid_tokens > 0:
            loss_soft = (masked_loss_soft.sum() / num_valid_tokens) * (temperature ** 2)
        else:
            loss_soft = torch.tensor(0.0, device=student_logits.device)
    else:
        # soft Loss
        loss_soft = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # Hard Loss
        teacher_predictions = teacher_logits.argmax(dim=-1) # (batch_size * num_vision_tokens)
        # student_logits: (B, T, V) -> (B*T, V)
        # teacher_predictions: (B, T) -> (B*T)
        loss_hard = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_predictions.view(-1)
        )

    # --- 3. Total loss ---
    loss = alpha * loss_soft + (1 - alpha) * loss_hard
    
    return loss

def calculate_distillation_loss_one_by_one(
    student_logits: list,  # 学生logits的列表 [ (len1, V), (len2, V), ... ]
    teacher_logits: list,  # 教师logits的列表 [ (len1, V), (len2, V), ... ]
    temperature: float,
    alpha: float
) -> torch.Tensor:
    """
    逐个样本计算变长序列的蒸馏损失，以避免内存爆炸。
    """
    # 确保两个列表的样本数一致
    assert len(student_logits) == len(teacher_logits)

    batch_size = len(student_logits)
    if batch_size == 0:
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

    total_loss_hard = 0.0
    total_loss_soft = 0.0
    total_valid_tokens = 0

    # 遍历批次中的每一个样本
    for student_logits, teacher_logits in zip(student_logits, teacher_logits):
        # student_logits 和 teacher_logits 的形状是 (num_tokens, vocab_size)
        num_tokens = student_logits.shape[0]
        if num_tokens == 0:
            continue

        # --- 计算当前样本的 Hard Loss ---
        # F.cross_entropy 默认会对一个样本内的所有token损失取平均
        teacher_predictions = teacher_logits.argmax(dim=-1)
        loss_hard_sample = F.cross_entropy(student_logits, teacher_predictions)

        # --- 计算当前样本的 Soft Loss (KL Divergence) ---
        # reduction='batchmean' 会对 batch 内所有元素的损失取平均。
        # 在这里，batch就是num_tokens，所以它会返回这个样本的平均KL散度。
        loss_soft_sample = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # 为了得到整个batch的真实平均损失，我们需要按token数量加权
        total_loss_hard += loss_hard_sample * num_tokens
        total_loss_soft += loss_soft_sample * num_tokens
        total_valid_tokens += num_tokens
    
    # 防止批次为空或所有序列都为空
    if total_valid_tokens == 0:
        return torch.tensor(0.0, device=student_logits[0].device if student_logits else 'cpu')

    # 计算整个批次在所有有效token上的平均损失
    avg_loss_hard = total_loss_hard / total_valid_tokens
    avg_loss_soft = total_loss_soft / total_valid_tokens

    # 合并损失
    final_loss = alpha * avg_loss_soft + (1 - alpha) * avg_loss_hard
    
    return final_loss

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_punctuation_token_ids(tokenizer):
    import string
    
    meaningless_tokens = []
    # meaningless_tokens.extend(string.punctuation)
    meaningless_tokens.extend(string.whitespace)
    # meaningless_tokens.extend(string.digits)

    meaningless_token_ids = tokenizer.convert_tokens_to_ids(meaningless_tokens)
    # return [(token, token_id) for token, token_id in zip(meaningless_tokens, meaningless_token_ids) if token_id != tokenizer.pad_token_id]

    return [tid for tid in meaningless_token_ids if tid is not None]

def train(args):
    # Load model
    ## load VLM (freezed)
    model, processor, tokenizer = load_model(args.model_name, args.device)
    model.eval()  # Set model to evaluation mode
    vit, llm = None, None
    if "llava" in args.model_name:
        vit, llm = model.vision_tower, model.language_model
        replace_llava1_5_receive_vit_output_and_return_image_mask_and_specify_hidden_states()
    elif "qwen" in args.model_name:
        vit, llm = model.visual, model.model
        replace_qwen2_5_vl_test_directions_processor_return_indices()
        replace_qwen2_5_vl_receive_vit_output_and_return_image_mask_and_specify_hidden_states()
    else:
        raise ValueError("Unsupported model type.")

    ## load VisionTokenDecoder (unfreezed)
    if "llava" in args.model_name:
        input_dim = llm.config.hidden_size
        output_dim = model.config.text_config.vocab_size
    elif "qwen" in args.model_name:
        input_dim = model.config.hidden_size
        output_dim = model.config.vocab_size
    else:
        raise ValueError("Unsupported model type for decoder initialization.")
    ### initialize the decoder with the LLM's lm_head weights
    decoder = VisionTokenDecoder(hidden_size=input_dim, vocab_size=output_dim)
    if "llava" in args.model_name:
        source_weights = llm.lm_head.weight.data.clone().detach()
        decoder.decoder.weight = torch.nn.Parameter(source_weights)
    elif "qwen" in args.model_name:
        source_weights = model.lm_head.weight.data.clone().detach()
        decoder.decoder.weight = torch.nn.Parameter(source_weights)
    else:
        raise ValueError("Unsupported model type for decoder initialization.")
    decoder.to(args.device)

    # Load data
    dataloader = load_data(
        args.model_name,
        processor,
        batch_size=args.batch_size,
        split="train"
    )

    val_dataloader = load_data(
        args.model_name,
        processor,
        batch_size=args.batch_size,
        split="val"
    )

    # Load optimizer
    optimizer = torch.optim.AdamW(
        list(decoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(dataloader) * args.epochs
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"Total training steps: {num_training_steps}. LR scheduler initialized.")

    # Early Stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    save_path_dir = Path(args.save_dir) / args.run_name
    save_path_dir.mkdir(parents=True, exist_ok=True)

    # delete the content in save_path_dir if exists
    for file in save_path_dir.glob("*"):
        file.unlink()

    # Forward
    step_cnt = 0
    training_stopped = False

    for epoch_id in range(args.epochs):
        model.eval()
        decoder.train()

        for idx, batch in enumerate(tqdm(dataloader)):
            
            if "qwen" in args.model_name:
                inputs, image_indices = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:
                image_indices = None
                inputs = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                ## get vision embeddings
                if "llava" in args.model_name:
                    image_reprs = model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        vision_feature_layer=model.config.vision_feature_layer,
                        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    )  # (batch_size, num_vision_tokens, hidden_size)
                elif "qwen" in args.model_name:
                    inputs["pixel_values"] = inputs["pixel_values"].type(vit.dtype)
                    image_reprs = vit(
                        inputs["pixel_values"], 
                        grid_thw=inputs["image_grid_thw"], 
                    )  # (batch_size * num_vision_tokens, hidden_size)
                else:
                    pass

                ## get logits from the LLM decoder
                if "llava" in args.model_name:
                    outputs, image_masks = model(
                        **inputs,
                        image_features=image_reprs,
                        return_dict=True, 
                        output_hidden_states=False
                    )
                elif "qwen" in args.model_name:
                    outputs, image_masks = model(
                        **inputs,
                        image_embeds=image_reprs,
                        return_dict=True, 
                        output_hidden_states=False
                    )
                # layer_id = 24 if "llava" in args.model_name else -1
                # activations = llm.lm_head(outputs.hidden_states[layer_id])
                # we can directly use the outputs.hidden_states because of the monkey patch
                if "llava" in args.model_name:
                    activations = llm.lm_head(outputs.hidden_states)
                elif "qwen" in args.model_name:
                    activations = model.lm_head(outputs.hidden_states)
                else:
                    raise ValueError("Unsupported model type.")
                
                # prepare the vlm image logits
                # use list:
                vlm_image_logits = []
                for idx, img_mask in enumerate(image_masks):
                    img_logits = activations[idx][img_mask]
                    vlm_image_logits.append(img_logits)

                # use one sequence:
                vlm_image_logits = torch.cat(vlm_image_logits, dim=0).detach()  # (batch_size * num_vision_tokens, vocab_size)

                # use batch:
                # loss_mask = None
                # if "llava" in args.model_name:
                #     vlm_image_logits = torch.stack(vlm_image_logits, dim=0).detach()  # (batch_size, num_vision_tokens, vocab_size)
                #     # though padding is not needed, we still make a loss_mask (all 1)
                #     loss_mask = torch.ones(vlm_image_logits.shape[:2], device=vlm_image_logits.device, dtype=torch.bool)
                # elif "qwen" in args.model_name:
                #     vlm_image_logits = pad_sequence(
                #         vlm_image_logits, 
                #         batch_first=True, 
                #         padding_value=0.0 
                #     )
                #     lengths = [t.shape[0] for t in vlm_image_logits]
                #     max_len = vlm_image_logits.shape[1]
                #     loss_mask = torch.arange(max_len, device=vlm_image_logits.device)[None, :] < torch.tensor(lengths, device=vlm_image_logits.device)[:, None]
                # else:
                #     raise ValueError("Unsupported model type.")

            ## get logits from the vision token decoder
            # use list:
            # if "llava" in args.model_name:
            #     decoder_image_logits = [img_logits for img_logits in decoder(image_reprs)]
            # elif "qwen" in args.model_name:
            #     decoder_image_logits = []
            #     for idx, img_indice in enumerate(image_indices):
            #         img_logits = decoder(image_reprs[img_indice])
            #         decoder_image_logits.append(img_logits)
            # else:
            #     raise ValueError("Unsupported model type.")

            # use one sequence:
            decoder_image_logits = decoder(image_reprs)
            if "llava" in args.model_name:
                decoder_image_logits = decoder_image_logits.view(-1, model.config.vocab_size)  # (batch_size * num_vision_tokens, vocab_size)
            
            # use batch:
            # if "llava" in args.model_name:
            #     decoder_image_logits = decoder(image_reprs)  # (batch_size, num_vision_tokens, vocab_size)
            # elif "qwen" in args.model_name:
            #     decoder_image_logits = []
            #     for idx, img_indice in enumerate(image_indices):
            #         img_logits = decoder(image_reprs[img_indice])
            #         decoder_image_logits.append(img_logits)
            #     decoder_image_logits = pad_sequence(
            #         decoder_image_logits, 
            #         batch_first=True, 
            #         padding_value=0.0 
            #     )
            # else:
            #     raise ValueError("Unsupported model type.")
            
            # For meaningless tokens
            # if "qwen" in args.model_name:
            #     punctuation_ids = get_punctuation_token_ids(tokenizer)
            #     vlm_image_logits[:, punctuation_ids] -= 10.0 # make them less likely to be predicted

            # calculate loss
            loss = calculate_distillation_loss(
                student_logits=decoder_image_logits,
                teacher_logits=vlm_image_logits,
                # loss_mask=loss_mask,
                temperature=args.temperature,
                alpha=args.alpha,
                # use_batch=True
            )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (idx + 1) % args.gradient_accumulation_steps == 0 or (idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_cnt += 1

            #  save memory
            del image_reprs, outputs, image_masks, activations
            del vlm_image_logits, decoder_image_logits

            # save the loss info
            if step_cnt % args.log_steps == 0:
                print(f"Step {step_cnt}: Loss = {loss.item()}")

                # evaluate the model
                if step_cnt % args.eval_steps == 0:
                    print("Evaluating the model...")
                    decoder.eval()
                    avg_loss = evaluate(args, model, decoder, val_dataloader)
                    decoder.train()  # Set decoder back to training mode
                    print(f"Validation Loss: {avg_loss}")

                    if avg_loss < best_val_loss:
                        best_val_loss = avg_loss
                        patience_counter = 0
                        print(f"New best validation loss: {best_val_loss:.4f}. Saving best model...")
                        best_model_path = save_path_dir / "model_best.pt"
                        torch.save({"step": step_cnt, "model_state_dict": decoder.state_dict()}, best_model_path)
                    else:
                        patience_counter += 1
                        print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")

                    if patience_counter >= args.patience:
                        print("Early stopping triggered.")
                        training_stopped = True
                        break
                else:
                    avg_loss = None

                # Here you can save the loss info to a file or log it as needed
                save_path = Path(args.save_dir) / args.run_name / "loss_info.jsonl"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with jsonlines.open(save_path, mode='a') as f:
                    f.write({
                        "epoch": epoch_id,
                        "step": step_cnt,
                        "loss": loss.item(),
                        "eval_loss": avg_loss,
                        "metadata": {
                            "model_name": args.model_name,
                            "alpha": args.alpha,
                            "temperature": args.temperature,
                            "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay,
                        }
                    })

            # save the vision decoder
            if step_cnt % args.save_steps == 0:
                save_path = Path(args.save_dir) / args.run_name / f"model_step_{step_cnt}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "step": step_cnt,
                    "model_state_dict": decoder.state_dict(),
                    # "optimizer_state_dict": optimizer.state_dict(),
                }, save_path)

        if training_stopped:
            break

def evaluate(args, model, decoder, dataloader):
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            
            if "qwen" in args.model_name:
                inputs, image_indices = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            else:
                image_indices = None
                inputs = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # get vision embeddings
            if "llava" in args.model_name:
                image_reprs = model.get_image_features(
                    pixel_values=inputs["pixel_values"],
                    vision_feature_layer=model.config.vision_feature_layer,
                    vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                )
            elif "qwen" in args.model_name:
                inputs["pixel_values"] = inputs["pixel_values"].type(model.visual.dtype)
                image_reprs = model.visual(
                    inputs["pixel_values"], 
                    grid_thw=inputs["image_grid_thw"], 
                )
            else:
                raise ValueError("Unsupported model type.")

            # get logits from the LLM decoder
            if "llava" in args.model_name:
                outputs, image_masks = model(
                    **inputs,
                    image_features=image_reprs,
                    return_dict=True, 
                    output_hidden_states=False
                )  # (batch_size, num_vision_tokens, hidden_size)
            elif "qwen" in args.model_name:
                outputs, image_masks = model(
                    **inputs,
                    image_embeds=image_reprs,
                    return_dict=True, 
                    output_hidden_states=False
                )  # (batch_size * num_vision_tokens, hidden_size)
            # layer_id = 24 if "llava" in args.model_name else -1
            # activations = model.language_model.lm_head(outputs.hidden_states[layer_id])
            if "llava" in args.model_name:
                activations = model.language_model.lm_head(outputs.hidden_states)
            elif "qwen" in args.model_name:
                activations = model.lm_head(outputs.hidden_states)
            else:
                raise ValueError("Unsupported model type.")
            
            # prepare the vlm image logits
            # use list:
            vlm_image_logits = []
            for idx, img_mask in enumerate(image_masks):
                img_logits = activations[idx][img_mask]
                vlm_image_logits.append(img_logits)

            # use one sequence:
            vlm_image_logits = torch.cat(vlm_image_logits, dim=0)  # (batch_size * num_vision_tokens, vocab_size)

            # get logits from the vision token decoder
            decoder_image_logits = decoder(image_reprs)  # (batch_size, num_vision_tokens, vocab_size)
            if "llava" in args.model_name:
                decoder_image_logits = decoder_image_logits.view(-1, model.config.vocab_size)  # (batch_size * num_vision_tokens, vocab_size)

            # calculate loss
            teacher_predictions = vlm_image_logits.argmax(dim=-1)
            loss_hard = F.cross_entropy(
                decoder_image_logits.view(-1, model.config.vocab_size),
                teacher_predictions.view(-1),
            )
            total_loss += loss_hard.item()

            del image_reprs, outputs, image_masks, activations
            del vlm_image_logits, decoder_image_logits
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss}")

    return avg_loss

def parse_args():
    
    parser = argparse.ArgumentParser(description="Train Vision Token Decoder")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--run_name", type=str, default="vision_token_decoder", help="Name of the run for saving checkpoints and logs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass. Simulates a larger batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--warmup_steps", type=float, default=500, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for soft loss in the combined loss function.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax in loss calculation.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save the trained model.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Steps interval to save the model.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Steps interval to evaluate the model.")
    parser.add_argument("--log_steps", type=int, default=100, help="Steps interval to save the loss info.")    

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    train(args)