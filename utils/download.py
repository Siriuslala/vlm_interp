

# from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir="/raid_sdd/lyy/hf")

from huggingface_hub import snapshot_download

from datasets import load_dataset
import subprocess
import jsonlines
import os


# VSR
# text_path = "/raid_sdd/lyy/dataset/visual-spatial-reasoning/data/splits/random/test.jsonl"
# image_path = "/raid_sdd/lyy/dataset/visual-spatial-reasoning/data/images/test"
# with jsonlines.open(text_path, mode='r') as f:
#     for line in f:
#         img_link = line["image_link"]
#         save_path = os.path.join(image_path, line["image"])
#         subprocess.run(["curl", "-C", "-", "-o", save_path, img_link], check=True)
        

# ds = load_dataset("MMVP/MMVP", cache_dir="/raid_sdd/lyy/dataset")
# print(ds)


snapshot_download(repo_id="MMVP/MMVP", repo_type="dataset",
                  local_dir="/raid_sdd/lyy/dataset",
                  local_dir_use_symlinks=False, resume_download=True)

