from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

import sys
sys.path.append(str(root_dir))


model_path = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
)

processor = AutoProcessor.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image")

text = """
Please choose the best decription for the image from the following 4 options:\n
            A: a bird\n
            B: a crocodile\n
            C: a cat\n
            D: a panda\n
            Please select the correct option and return only the letter of the option.\nYour choice is:
"""
image_file = root_dir / "test_figs/test_dinasour.png"

text = "Is there a bus in the image? Please answer with 'yes' or 'no'."
text = "Is there a bus in the image? Whats's your reason(do not mention the two men)?"
text = "Describe this image as detailed as possible, and tell me what type is the car in the background."
text = "What's behind the two men? A bus or a van?"
image_file = root_dir / "test_figs/pope/bad_case/llava_pope_adversarial_Is there a bus in the image?_no/1487.png"

text = "Is there a vase in the image? Please answer with 'yes' or 'no'."
image_file = root_dir / "test_figs/pope_random-cases-llm_layer_25/sample-id_8742-entity_vase-ans_yes/8742.png"

text = "Is the mug to the left of or to the right of the plate?"
text = "Is the mug behind or in front of the plate? Why?"
image_file = root_dir / "test_figs/whatsup/mug_left_of_plate.jpg"

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": text},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


device = "cuda:0"
raw_image = Image.open(image_file).convert("RGB")
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)

# decode the input ids
# input_ids = inputs.input_ids
# input_content = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
# input_tokens = [processor.tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
# print(input_content)

model.to(device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))