import re
import jsonlines
import random
from pathlib import Path

# from transformers import (
#     Qwen2_5_VLForConditionalGeneration,
#     Qwen2VLForConditionalGeneration,
#     LlavaForConditionalGeneration,
#     LlavaNextForConditionalGeneration,
#     AutoModel,
#     AutoConfig,
#     AutoTokenizer, 
#     AutoProcessor
# )

# text = "A glass to the left of a armchair"
# pattern = r"(A|an) (.*) to the (left|right) of (a|an) (.*)"

# match = re.match(pattern, text)

# print(match)
# print(match.group(2), match.group(5))


# 2332870, 2354453, 2338056, 2339558, 2369060, 2416755, 2331146, 2332040, 2332720
path = "/raid_sdd/lyy/Interpretability/lyy/mm/figures/seg_with_unembedding_tokens/llava1_5_7b/seg_img_2354453/text_tokens_analysis.jsonl"
path = "/raid_sdd/lyy/Interpretability/lyy/mm/figures/seg_with_unembedding_tokens_delete_pos_embed/llava1_5_7b/seg_img_2354453/text_tokens_analysis.jsonl"

# all_data = []
# with jsonlines.open(path, "r") as f:
#     for line in f:
#         all_data.append(line)
# for line in all_data:
#     data = line["text_tokens"]
#     for layer_id in range(len(data)):
#         print(f"object: {line["object"]}, layer_id: {layer_id}, data: {data[layer_id]} \n")
#     print("-" * 50)


# x = [2, 1, 3, 4, 5, 88, 93]
# random.seed(14)  # Set a seed for reproducibility
# random.shuffle(x)
# print(x)

# import string
# print(len(string.whitespace))
# x = list(string.whitespace)
# print(f"{x}")
# print(list(string.punctuation))


# tokenizer = AutoTokenizer.from_pretrained(
#             "/raid_sdd/lyy/hf/models--LLaVA-1.5-7B",
#             padding_side='left',
#         )
# print(tokenizer.tokenize("skateboard"))


    
# meaningless_tokens = []
# meaningless_tokens.extend(string.punctuation)
# meaningless_tokens.extend(string.whitespace)
# print(f"meaningless_tokens: {meaningless_tokens}")

# for i in range(21):
#     theta = 10000 ** (-2 * i / 40)
#     g = (1 / (theta + 1e-9)) ** 0.1
#     print(f"theta: {theta}, g: {g}, theta * g: {theta * g}")

# import torch
# i_normed = torch.arange(0, 80, 2, dtype=torch.float)
# print(i_normed)

# x = Path(__file__)
# y = Path(__file__).parent
# z = Path(__file__).name
# w = Path(__file__).stem
# a = Path(__file__).suffix
# print(x)
# print(y)
# print(z)
# print(w)
# print(a)

# import torch
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch._C._GLIBCXX_USE_CXX11_ABI)

x = min(3, 1)
print(x)