import os
import re
from pathlib import Path

# 当前目录
folder = Path(".")
images = list(folder.glob("rgb_*_0000.png"))

# 提取数字并排序
def extract_index(file_name):
    match = re.search(r"rgb_(\d+)_0000\.png", file_name)
    return int(match.group(1)) if match else -1

images.sort(key=lambda x: extract_index(x.name))

# 重命名
for i, img in enumerate(images):
    new_name = f"frame_{i:04d}.png"
    os.rename(img, new_name)
    print(f"Renamed {img} -> {new_name}")