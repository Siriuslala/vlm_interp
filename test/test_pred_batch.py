from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import torch

from patch.monkey_patch import *

from typing import List, Dict, Union


def load_model(device):
    model_dir="/raid_sdd/lyy/hf/Qwen/Qwen2.5-VL-7B-Instruct"

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto"
    )
    model.to(device)

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16, //如果推理太慢也可以启用半精度来加速
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    min_pixels = 256 * 28 * 28  # 这里做了推理时候的图片压缩,平衡精度和速度,默认是全尺寸图片进行推理,很慢也要很大的显存
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_dir, min_pixels=min_pixels, max_pixels=max_pixels, padding_side='left'
    )  # left padding: <|endoftext|> 151644

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    
    return model, processor

def prepare_batch_inputs(image_paths: List[str], prompts: List[str], processor, model):
    """ 真正的批量输入准备 """
    # 构造批量消息
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
    
    # 批量处理文本模板
    texts = [processor.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    ) for msg in batch_messages]
    
    # 关键改进：批量处理视觉信息（修改process_vision_info使其支持批量）
    all_images = []
    for msg in batch_messages:
        images, _ = process_vision_info(msg)
        if images:
            all_images.extend(images)  # 收集所有图片
    
    # 创建真正的批量输入
    inputs = processor(
        text=texts,
        images=all_images if all_images else None,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    
    return inputs

def generate_batch_responses(model, processor, image_paths: List[str], prompts: List[str], max_new_tokens=256):
    """ 端到端批量生成 """
    inputs = prepare_batch_inputs(image_paths, prompts, processor, model)
    
    # 真正的批量推理
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # 批处理时建议关闭采样以保证效率
        )
    
    # 解码时跳过输入部分
    outputs = []
    for i in range(len(inputs.input_ids)):
        output_ids = generated_ids[i][len(inputs.input_ids[i]):]
        outputs.append(processor.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ))
    
    # 打印结果
    for img_path, prompt, res in zip(image_paths, prompts, outputs):
        print(f"图片: {img_path}\n提示: {prompt}\n输出: {res}\n{'-'*50}")
    
    return outputs

def run_batch_forward(model, processor, image_paths: List[str], prompts: List[str]):
    """ 端到端批量生成 """
    inputs = prepare_batch_inputs(image_paths, prompts, processor, model)
    # print("inputs", inputs)  # input_ids, attention mask, pixel_values, image_grid_thw
    
    # input_embed: (bsz, len, dim)
    # image_embed: (all_len, dim)
    
    # input_ids_0 = inputs.input_ids[0]
    # input_ids_1 = inputs.input_ids[1]
    # input_content_0 = processor.tokenizer.batch_decode(input_ids_0, skip_special_tokens=False)
    # input_content_1 = processor.tokenizer.batch_decode(input_ids_1, skip_special_tokens=False)
    
    # vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    # vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    # im_end_pos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # 真正的批量推理
    with torch.inference_mode():
        outputs = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )
    
    for i, h in enumerate(outputs.hidden_states):
        print(h.shape)  # (bsz, len, dim)
        # calculate the average norm of the hidden state over all positions
        # avg_norm = torch.mean(torch.norm(h, dim=-1), dim=1).item()
        
        # # calculate the average norm of the hidden state over image tokens
        # # get image token positions
        # avg_norm_img = torch.mean(torch.norm(h[:, image_tokens_start_pos+1:image_tokens_end_pos, :], dim=-1), dim=1).item()
        # avg_norm_text = torch.mean(torch.norm(h[:, image_tokens_end_pos+1:, :], dim=-1), dim=1).item()
        
        # if i < len(outputs.hidden_states) - 1:
        #     print(f"Layer {i} hidden states: avg_norm={avg_norm}, avg_norm_img={avg_norm_img}, avg_norm_text={avg_norm_text}")
        # else:
        #     print(f"Layer {i} outputs: avg_norm={avg_norm}, avg_norm_img={avg_norm_img}, avg_norm_text={avg_norm_text}")
        
    
    # last_token_h = outputs.hidden_states[-1][:, -1, :]
    # max_dim_value, _ = torch.max(last_token_h, dim=-1)
    # min_dim_value, _ = torch.min(last_token_h, dim=-1)
    # print("last_token_h", last_token_h.shape, last_token_h)
    # print("max_dim_value", max_dim_value.item())
    # print("min_dim_value", min_dim_value.item())    


if __name__ == "__main__":
    
    device = "cuda:3"
    model, processor = load_model(device)
    test_data = [
        ("/raid_sdd/lyy/Interpretability/lyy/mm/test_figs/test_code.jpg", "describe the image and tell me what is the main object in the image"),
        ("/raid_sdd/lyy/Interpretability/lyy/mm/test_figs/test_dinasour.png", "describe the image and tell me what is the main object in the image"),
    ]
    image_paths = [x[0] for x in test_data]
    prompts = [x[1] for x in test_data]
    
    
    def test_delete_rope():
        pass
    
    
    # patching
    shuffle_image_order = 1
    delete_llm_pos_embed = 0
    if shuffle_image_order:
        replace_qwen2_5_vl_shuffle_image_token_orders()
    if delete_llm_pos_embed:
        # layer_ids = [i for i in range(model.config.num_hidden_layers)]
        layer_ids = [0,]
        layer_ids = [model.config.num_hidden_layers - i for i in range(1, 27)]
        replace_qwen2_5_vl_delete_llm_pos_embed(layer_ids_to_delete=layer_ids)
    
    # 批量推理
    results = generate_batch_responses(model, processor, image_paths, prompts)
    
    # 分析
    # run_batch_forward(model, processor, image_paths, prompts)