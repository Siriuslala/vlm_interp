conda create -n mm python=3.12 -y
# pytorch
pip install -i https://mirrors.aliyun.com/pypi/simple modelscope transformers qwen-vl-utils
python download.py

# flashattention: 按教程来 pip install xxx.whl --no-bulid-isolation