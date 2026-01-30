import json
from pathlib import Path

# ====== 修改为你的 json 路径 ======
json_path = Path(r"D:/zhanlan/segment_coco_out/instances_train.json")

# ==================================
with json_path.open("r", encoding="utf-8") as f:
    coco = json.load(f)

categories = coco.get("categories", [])
if not categories:
    raise RuntimeError("JSON 中没有 categories 字段")

# 按 category_id 排序（保证稳定）
categories = sorted(categories, key=lambda x: x["id"])

class_names = [c["name"] for c in categories]
num_classes = len(class_names)

print("=" * 60)
print(f"JSON 文件: {json_path}")
print(f"类别数量: {num_classes}")
print("类别列表(按 category_id 排序):")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print("\n===== 可直接复制到 MMDetection config =====\n")

print(f"num_classes = {num_classes}\n")

print("metainfo = dict(")
print("    classes=(")
for name in class_names:
    print(f"        '{name}',")
print("    )")
print(")")
print("\n" + "=" * 60)
