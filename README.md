# Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models

<!-- <p align="center">
   <a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vRR3Wl7wsCgHpwUw1_eUXW_fptAPLL3FkhnW_rua0O1Ji_GIVrpTjY5LaKAhwO-WeARjnY_KNw0SYNJ/pubhtml" target="_blank">🌐 Leaderboard (new)</a> | <a href="https://twitter.com/thukeg" target="_blank">🐦 Twitter</a> | <a href="mailto:agentbench@googlegroups.com">✉️ Google Group</a> | <a href="https://arxiv.org/abs/2308.03688" target="_blank">📃 Paper </a>
</p> -->

<p align="center">
   <a href="https://arxiv.org/abs/2509.19191" target="_blank">📃 Paper </a>
</p>


<!-- <p align="center">
👋 Join our <a href="https://join.slack.com/t/agentbenchcol-huw1944/shared_invite/zt-20ixabcuv-31cFLBAkqGQxQkJqrWVEVg" target="_blank">Slack</a>  for <i>Q & A</i> or <i><b>collaboration</b> on next version of AgentBench</i>!
</p> -->

<p align="center">
👋 We encourage everyone to explore deeper on VLMs based on our methods!
</p>

## Introduction
Inspired by the dual-stream hypothesis of human vision, which distinguishes the "what" and "where" pathways, we deconstruct the visual processing in VLMs into two parts:
- object recognition -- the "what" way
- spatial perception -- the "where" way

For object recognition, we convert images into *text token maps* and find that the model's perception of image content unfolds as a two-stage process from shallow to deep layers, beginning with attribute recognition and culminating in semantic disambiguation.

For spatial perception, we theoretically derive and empirically verify the geometric structure underlying the positional representation in VLMs. 

Based on these findings, we introduce *an instruction-agnostic token compression algorithm* based on a plug-and-play visual decoder to improve decoding efficiency, and *a RoPE scaling technique* to enhance spatial reasoning. 
Through rigorous experiments, our work validates these analyses, offering a deeper understanding of VLM internals and providing clear principles for designing more capable future architectures.


## Quick Start
### Environment Setup
We recommend you to create a conda environment first as follows:

```shell
conda create -n mm python=3.12
conda activate mm
```

Then clone our repository and install the required packages:

```shell
git clone https://github.com/Siriuslala/vlm_interp.git
cd vlm_interp
pip install -r requirements.txt
```

Before starting, please create a `.env` file, and finish the configuration of the environment variables following the example in `.env_example`:

```shell
ROOT_DIR=./  # where your code is located
DATA_DIR=/path/to/your/data  # where your data is located
WORK_DIR=/path/to/your/work  # intermediate files (e.g., model checkpoints, log files, etc.)
```

### Directory Structure
```
[vlm_interp/]
├── .env_example
├── eval  # evaluation scripts
│   ├── CHAIR
│   ├── COCO
│   ├── GQA
│   ├── MMBench
│   ├── MME
│   ├── POPE
│   ├── SQA
│   ├── TextVQA
│   ├── VQA
│   ├── VSR
│   ├── WhatsUp
│   │   ├── bboxes  # our annotations for WhatsUp
│   │   └── dataset_zoo
│   ├── custom_data.py
│   └── data_utils.py
├── font
│   └── SimHei.ttf
├── model  # include the visual decoder and RoPE scaling model
│   ├── qwen2_vl_rope_scaling.py
│   └── unembedding.py
├── patch  # monkey patch files
│   ├── intern_2_5_hijack.py
│   ├── internvl2_5_utils
│   ├── llava1_5_vl_hijack.py
│   ├── monkey_patch.py
│   └── qwen2_5_vl_hijack.py
├── requirements.txt
├── test  # simple tests
│   ├── test_intern.py
│   ├── test_llava.py
│   ├── test_pred.py
│   ├── test_pred_batch.py
│   └── test_qwen.py
├── test_direction.py  # test scripts for the exploration of spatial perception
├── test_pos_embed.py  # test scripts for the exploration of positional embedding
├── test_seg_map.py  # test scripts for the exploration of object recognition
├── train_rope_scaling  # scripts for training the RoPE scaling model
│   ├── qwen2_vl_sft_parallel.py
│   ├── qwen2_vl_sft_single_gpu.py
│   ├── qwen_vl_finetune
│   └── scripts
│       ├── sft.sh
│       ├── sft_official.sh
│       └── sft_parellel.sh
├── train_unembedding  # scripts for training the visual decoder for token compression
│   ├── dataset.py
│   ├── draw.py
│   ├── scripts
│   │   └── train.sh
│   └── train.py
└── utils
    ├── __init__.py
    ├── data_process.py  # scripts for processing the dataset
    ├── download.py
    ├── draw_file_tree.py
    ├── env.sh
    ├── font.sh
    └── intern_vl.py
```

### Object Recognition
The scripts for object recognition are located in `test_seg_map.py`.
This script contains the functions corresponding to the experiments in Section 3 of our paper. Specific usage details can be found in the comments of the script (after `if __name__ == '__main__':`).

Some of the important functions in `test_seg_map.py` are:
- `seg_with_unembedding_tokens`: draw the text maps and segmentation maps for LLaVA;
- `seg_with_unembedding_tokens_qwen`: draw the text maps and segmentation maps for QwenVL;


### Spatial Perception
The scripts for spatial perception are located in `test_direction.py`.
This script contains the functions corresponding to the experiments in Section 4 of our paper. Specific usage details can be found in the comments of the script (after `if __name__ == '__main__':`).

Some of the important functions in `test_direction.py` are:
- `explore_1d_pos_embed_visual_geometry`: plot the 1D positional embedding in visual geometry (Section 4.1);
- `get_relation_representations_layerwise`: plot the relation representations layerwise for left of, right of, behind, in front of (Section 4.2);
- `intervene_in_spatial_reasoning`: intervene in the spatial reasoning process (Section 4.2);
- `erase_object_in_llm`: the `erase` test in Section 4.2;


## Citation

```
@article{li2025reading,
  title={Reading Images Like Texts: Sequential Image Understanding in Vision-Language Models},
  author={Li, Yueyan and Zhao, Chenggong and Zang, Zeyuan and Yuan, Caixia and Wang, Xiaojie},
  journal={arXiv preprint arXiv:2509.19191},
  year={2025}
}
```