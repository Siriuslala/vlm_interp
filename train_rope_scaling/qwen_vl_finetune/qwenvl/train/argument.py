import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    # ADDED: LoRA specific arguments
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    lora_r: int = field(default=8, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj",
        metadata={"help": "Comma separated list of module names to apply Lora to."}
    )
    # ADDED: Model Type
    rope_scaling: bool = field(default=False, metadata={"help": "Whether to use RoPE scaling."})
    scaling_type: str = field(default="sigmoid", metadata={"help": "Type of scaling to use for RoPE."})
    poly_alpha: float = field(default=99.0, metadata={"help": "Alpha parameter for polynomial scaling."})
    poly_p: float = field(default=6, metadata={"help": "P parameter for polynomial scaling."})
    sig_alpha: float = field(default=99.0, metadata={"help": "Alpha parameter for sigmoid scaling."})
    sig_mid_point: float = field(default=0.6, metadata={"help": "Midpoint for sigmoid scaling."})
    sig_k: float = field(default=40.0, metadata={"help": "K parameter for sigmoid scaling."})


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # ADDED: Quantization arguments for memory-efficient training with LoRA
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8bit."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4bit."})
    label_names: List[str] = field(default_factory=lambda: ["labels"])