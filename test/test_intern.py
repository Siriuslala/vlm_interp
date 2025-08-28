from PIL import Image

import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


device = "cuda:0"

model_dir = "/raid_sdd/lyy/hf/models--InternVL2_5-8B"
model_dir = "OpenGVLab/InternVL2_5-8B"
# model_dir = "/raid_sdd/lyy/hf/huggingface/hub/models--OpenGVLab--InternVL2_5-8B/snapshots/e9e4c0dc1db56bfab10458671519b7fa3dd29463"
model = AutoModel.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    padding_side='left',
    use_fast=True,
    trust_remote_code=True,
)
print(str(type(model)))
import pdb; pdb.set_trace()

# processor = AutoProcessor.from_pretrained(model_dir).from_pretrained(
#     model_dir, 
#     torch_dtype=torch.bfloat16, 
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# )

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_intern(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image")

image_file_0 = "/raid_sdd/lyy/Interpretability/lyy/mm/test_figs/test_dinasour.png"
image_file_1 = "/raid_sdd/lyy/Interpretability/lyy/mm/test_figs/test_code.jpg"


# text = """
# Please choose the best decription for the image from the following 4 options:\n
#             A: a bird\n
#             B: a crocodile\n
#             C: a cat\n
#             D: a panda\n
#             Please select the correct option and return only the letter of the option.\nYour choice is:
# """
# conversation = [
#     {

#       "role": "user",
#       "content": [
#           {"type": "text", "text": text},
#           {"type": "image"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# raw_image = Image.open(image_file).convert("RGB")
# inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device)

# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))



# batch inference, single image per sample (单图批处理)
pixel_values_0 = load_image_intern(image_file_0, max_num=12).to(torch.bfloat16).cuda()
pixel_values_1 = load_image_intern(image_file_1, max_num=12).to(torch.bfloat16).cuda()
num_patches_list = [pixel_values_0.size(0), pixel_values_1.size(0)]
pixel_values = torch.cat((pixel_values_0, pixel_values_1), dim=0)

questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
generation_config = dict(max_new_tokens=1024, do_sample=False)

responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config,
                             )
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')