from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

import torch


def check(device="cuda:0"):
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
        model_dir, min_pixels=min_pixels, max_pixels=max_pixels
    )

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/raid_sdd/lyy/Interpretability/lyy/mm/test_figs/test_code.jpg", 
                },
                {"type": "text", "text": "describe the image and tell me what is the main object in the image"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # print("inputs", inputs)
    inputs = inputs.to(device)
    
    # ========================================================================================
    # Inference: Generation of the output
    # outputs = model.generate(**inputs, 
    #                             max_new_tokens=50,
    #                             return_dict_in_generate=True,
    #                             output_hidden_states=True,)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)
    
    
    # forward
    with torch.no_grad():
        outputs = model(**inputs,
                        return_dict=True,
                        output_hidden_states=True,)

    # print("outputs", type(outputs))
    # print(outputs.keys())  # ['logits', 'past_key_values', 'hidden_states', 'rope_deltas']
    # print(type(outputs.hidden_states), len(outputs.hidden_states))
    
    # =======================================================================================
    # decode the input ids
    input_ids = inputs.input_ids
    input_content = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    input_tokens = [processor.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
    print(input_content)
    
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_end_pos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print("vision_start_id", vision_start_id)
    print("vision_end_id", vision_end_id)
    print("image_token_id", image_token_id, model.config.image_token_id)
    
    image_tokens_start_pos= (inputs["input_ids"] == vision_start_id).nonzero().tolist()[0][1]  # 0: idx of True items; 1: position of the first True item
    image_tokens_end_pos= (inputs["input_ids"] == vision_end_id).nonzero().tolist()[0][1]
    im_end_pos= (inputs["input_ids"] == im_end_pos).nonzero().tolist()[-1][1]
    print("image_tokens_start_id", image_tokens_start_pos)
    print("image_tokens_end_id", image_tokens_end_pos)
    print("im_end_id", im_end_pos)
    
    # print("input_tokens:", input_tokens)
    # print("image_tokens:", input_tokens[0][image_tokens_start_pos+1:image_tokens_end_pos])
    print("text_tokens:", input_tokens[0][image_tokens_end_pos+1:im_end_pos])
    
    for i, h in enumerate(outputs.hidden_states):
        # print(h.shape)  # (bsz, len, dim)
        # calculate the average norm of the hidden state over all positions
        avg_norm = torch.mean(torch.norm(h, dim=-1), dim=1).item()
        
        # calculate the average norm of the hidden state over image tokens
        # get image token positions
        avg_norm_img = torch.mean(torch.norm(h[:, image_tokens_start_pos+1:image_tokens_end_pos, :], dim=-1), dim=1).item()
        avg_norm_text = torch.mean(torch.norm(h[:, image_tokens_end_pos+1:, :], dim=-1), dim=1).item()
        
        if i < len(outputs.hidden_states) - 1:
            print(f"Layer {i} hidden states: avg_norm={avg_norm}, avg_norm_img={avg_norm_img}, avg_norm_text={avg_norm_text}")
        else:
            print(f"Layer {i} outputs: avg_norm={avg_norm}, avg_norm_img={avg_norm_img}, avg_norm_text={avg_norm_text}")
        
    
    last_token_h = outputs.hidden_states[-1][:, -1, :]
    max_dim_value, _ = torch.max(last_token_h, dim=-1)
    min_dim_value, _ = torch.min(last_token_h, dim=-1)
    print("last_token_h", last_token_h.shape, last_token_h)
    print("max_dim_value", max_dim_value.item())
    print("min_dim_value", min_dim_value.item())


if __name__ == "__main__":
    check(device="cuda:3")