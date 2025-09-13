from matplotlib import pyplot as plt
import os
import jsonlines
from pathlib import Path
import re

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))

def draw_loss_curve(log_paths, draw_val=False):
    fig_dir = root_dir / "figures"
    save_dir = fig_dir / "train_vision_decoder"
    os.makedirs(save_dir, exist_ok=True)
    model_name = "llava" if "llava" in str(log_paths[0]) else "qwen"
    if draw_val:
        save_path = save_dir / f"{model_name}_loss_curve_val.pdf"
    else:
        save_path = save_dir / f"{model_name}_loss_curve.pdf"
    colors = ['blue', 'g', 'gold', 'c']
    plt.figure()
    for i, log_path in enumerate(log_paths):
        with jsonlines.open(log_path, 'r') as f:
            steps = []
            val_steps = []
            losses = []
            val_losses = []
            for item in f:
                steps.append(item['step'])
                losses.append(item['loss'])
                if item.get('eval_loss') is not None:
                    val_losses.append(item['eval_loss'])
                    val_steps.append(item['step'])
        label = log_path.parent.name
        pattern = r'(epoch|bsz|lr|warmup|alpha|temp|patience)(\d+(?:\.\d+)?(?:e-\d+)?)'
        matches = re.findall(pattern, label)
        params = {name: value for name, value in matches}
        # label = f"temperature={params.get('temp', 'N/A')}, alpha={params.get('alpha', 'N/A')}"
        label = f"alpha={params.get('alpha', 'N/A')}, temp={params.get('temp', 'N/A')}"
        if draw_val:
            plt.plot(val_steps, val_losses, label=label, color=colors[i])
        else:
            plt.plot(steps, losses, label=label, color=colors[i])
        plt.xlabel('Steps')
        y_label = "Train Loss" if not draw_val else "Validation Loss"
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(save_path)


if __name__ == "__main__":
    
    log_dirs = [
        # Path(work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr2e-4-warmup1000-alpha0.8-temp2.5-patience10"),
        # Path(work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr2e-4-warmup1000-alpha0.8-temp4.0-patience10"),
        # Path(work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr2e-4-warmup1000-alpha0.8-temp6.0-patience10"),
        Path(work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr2e-4-warmup1000-alpha0.7-temp5.0-patience10"),
        Path(work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr2e-4-warmup1000-alpha0.8-temp5.0-patience10"),
        Path(work_dir / "checkpoints_vision_decoder/qwen2_5_vl-epoch1-bsz32-lr2e-4-warmup1000-alpha0.9-temp5.0-patience10"),
    ] 
    log_paths = [log_dir / "loss_info.jsonl" for log_dir in log_dirs]
    draw_loss_curve(log_paths, draw_val=True)
