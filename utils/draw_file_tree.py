import os
import fnmatch
from pathlib import Path

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))


def should_ignore(path, root, ignore_patterns):
    """判断路径是否在忽略名单中"""
    rel_path = os.path.relpath(path, root)
    if rel_path == ".": return False
    
    # 检查每一级目录名和文件名
    path_parts = rel_path.split(os.sep)
    
    for pattern in ignore_patterns:
        # 去掉结尾的斜杠用于匹配
        p = pattern.rstrip('/')
        # 1. 直接匹配文件名或相对路径
        if fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(os.path.basename(path), p):
            return True
        # 2. 检查路径中的任一环节是否匹配（如忽略 __pycache__）
        if any(fnmatch.fnmatch(part, p) for part in path_parts):
            return True
    return False

def draw_tree(directory, ignore_patterns, prefix="", root=None):
    if root is None:
        root = directory
        print(f"[{os.path.basename(root)}/]")

    # 获取当前目录下未被忽略的项目
    try:
        all_items = sorted(os.listdir(directory))
    except PermissionError:
        return

    filtered_items = []
    for item in all_items:
        full_path = os.path.join(directory, item)
        if item == '.git':
            continue
        if not should_ignore(full_path, root, ignore_patterns):
            filtered_items.append(item)

    # 递归打印
    for i, item in enumerate(filtered_items):
        full_path = os.path.join(directory, item)
        is_last = (i == len(filtered_items) - 1)
        
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{item}")

        if os.path.isdir(full_path):
            extension = "    " if is_last else "│   "
            draw_tree(full_path, ignore_patterns, prefix + extension, root)


if __name__ == "__main__":
    current_dir = root_dir
    rules = []
    gitignore_file = os.path.join(current_dir, ".gitignore")
    if os.path.exists(gitignore_file):
        with open(gitignore_file, 'r') as f:
            rules.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
    draw_tree(current_dir, rules)