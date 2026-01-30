# -*- coding: utf-8 -*-
"""
Labelme -> COCO (instances) + train/val split + copy images
- 递归扫描 DATA_ROOT 下所有图片（多级子目录）
- 读取同名 labelme json（如 a.jpg 对应 a.json）
- split train/val
- 复制图片到 OUT_DIR/train2017 与 OUT_DIR/val2017
- 输出 COCO：OUT_DIR/instances_train.json / OUT_DIR/instances_val.json
"""

import json
import random
import shutil
import hashlib
from pathlib import Path
from PIL import Image


# ===================== 配 置 区 =====================

DATA_ROOT = r"D:/zhanlan/segment_data"      # 根目录（递归扫描）
OUT_DIR = r"D:/zhanlan/segment_coco_out"    # 输出根目录（推荐放这里）

TRAIN_IMG_DIRNAME = "train2017"
VAL_IMG_DIRNAME = "val2017"
TRAIN_JSON_NAME = "instances_train.json"
VAL_JSON_NAME = "instances_val.json"

VAL_RATIO = 0.2
RANDOM_SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LABEL_MAP = {
    # "1.3": "seam",
    # "1.5": "seam",
}

KEEP_FULL_IMAGE_POLYGON = True

# 是否同时复制 labelme 原始 json（可选，方便溯源）
COPY_ORIGINAL_LABELME_JSON = False

# ==================================================


def polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def shape_to_polygon(shape):
    st = shape.get("shape_type", "polygon")
    pts = shape.get("points", [])
    if not pts or len(pts) < 2:
        return None

    if st == "polygon":
        if len(pts) < 3:
            return None
        return [(float(x), float(y)) for x, y in pts]

    if st == "rectangle":
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    return None


def is_full_image_poly(poly, w, h, eps=1.0):
    if not poly or len(poly) < 4:
        return False
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (abs(x_min - 0) <= eps and abs(y_min - 0) <= eps and
            abs(x_max - (w - 1)) <= eps and abs(y_max - (h - 1)) <= eps)


def read_labelme_json(json_path: Path):
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(json_path.read_text(encoding="utf-8-sig"))


def short_hash(s: str, n=8) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def unique_dest_name(src_path: Path, used_names: set):
    """
    生成不冲突的目标文件名：
    - 优先用原始文件名
    - 冲突时加 hash 后缀
    """
    base = src_path.stem
    ext = src_path.suffix.lower()
    name = f"{base}{ext}"
    if name not in used_names:
        used_names.add(name)
        return name

    # 冲突：用相对路径做 hash 保证稳定
    h = short_hash(str(src_path))
    name2 = f"{base}_{h}{ext}"
    while name2 in used_names:
        h = short_hash(str(src_path) + name2)
        name2 = f"{base}_{h}{ext}"
    used_names.add(name2)
    return name2


def build_coco_with_copy(img_paths, root: Path, out_img_dir: Path, used_names: set,
                         starting_img_id=1, starting_ann_id=1, cat_name_to_id=None):
    """
    对一组图片：
    - 复制到 out_img_dir（文件名去重）
    - 生成 COCO dict（file_name 只写复制后的文件名）
    """
    if cat_name_to_id is None:
        cat_name_to_id = {}

    images = []
    annotations = []

    img_id = starting_img_id
    ann_id = starting_ann_id

    out_img_dir.mkdir(parents=True, exist_ok=True)

    for src_img in img_paths:
        src_json = src_img.with_suffix(".json")
        if not src_json.exists():
            continue

        # 复制图片（确保不重名）
        dst_name = unique_dest_name(src_img, used_names)
        dst_img = out_img_dir / dst_name
        shutil.copy2(src_img, dst_img)

        # 可选：复制原始 labelme json 方便溯源
        if COPY_ORIGINAL_LABELME_JSON:
            shutil.copy2(src_json, out_img_dir / f"{Path(dst_name).stem}.labelme.json")

        # 尺寸
        with Image.open(src_img) as im:
            w, h = im.size

        data = read_labelme_json(src_json)

        images.append({
            "id": img_id,
            "file_name": dst_name,   # 关键：仅文件名，配合 data_prefix.img
            "width": w,
            "height": h,
        })

        for shp in data.get("shapes", []):
            label = (shp.get("label") or "").strip()
            if not label:
                continue
            label = LABEL_MAP.get(label, label)

            poly = shape_to_polygon(shp)
            if poly is None:
                continue

            if (not KEEP_FULL_IMAGE_POLYGON) and is_full_image_poly(poly, w, h):
                continue

            if label not in cat_name_to_id:
                cat_name_to_id[label] = len(cat_name_to_id) + 1
            cid = cat_name_to_id[label]

            seg = [[coord for xy in poly for coord in xy]]
            bbox = bbox_from_points(poly)
            area = float(polygon_area(poly))
            if area <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "segmentation": seg,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    # categories（按 id 排序）
    inv = sorted(((cid, name) for name, cid in cat_name_to_id.items()), key=lambda x: x[0])
    categories = [{"id": cid, "name": name, "supercategory": "none"} for cid, name in inv]

    coco = {
        "info": {"description": "labelme2coco", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    return coco, img_id, ann_id, cat_name_to_id


def main():
    root = Path(DATA_ROOT)
    if not root.exists():
        print(f"DATA_ROOT 不存在: {root}")
        return

    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    train_img_dir = out_root / TRAIN_IMG_DIRNAME
    val_img_dir = out_root / VAL_IMG_DIRNAME

    # 递归找所有图片
    all_imgs = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

    # 只保留有同名 json 的图片
    paired = [p for p in all_imgs if p.with_suffix(".json").exists()]
    if not paired:
        print("没有找到任何“图片 + 同名labelme json”的配对文件")
        return

    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(paired)

    val_count = int(len(paired) * VAL_RATIO)
    val_imgs = paired[:val_count]
    train_imgs = paired[val_count:]

    # 共享类别表，保证 train/val category_id 一致
    cat_name_to_id = {}

    # 复制时去重：train/val 各自一个集合（更符合直觉）
    used_train_names = set()
    used_val_names = set()

    train_coco, _, _, cat_name_to_id = build_coco_with_copy(
        train_imgs, root, train_img_dir, used_train_names,
        starting_img_id=1, starting_ann_id=1, cat_name_to_id=cat_name_to_id
    )

    val_coco, _, _, cat_name_to_id = build_coco_with_copy(
        val_imgs, root, val_img_dir, used_val_names,
        starting_img_id=1, starting_ann_id=1, cat_name_to_id=cat_name_to_id
    )

    # 强制两份 categories 一致（用最终 cat_name_to_id 刷一遍）
    inv = sorted(((cid, name) for name, cid in cat_name_to_id.items()), key=lambda x: x[0])
    final_categories = [{"id": cid, "name": name, "supercategory": "none"} for cid, name in inv]
    train_coco["categories"] = final_categories
    val_coco["categories"] = final_categories

    train_json_path = out_root / TRAIN_JSON_NAME
    val_json_path = out_root / VAL_JSON_NAME
    train_json_path.write_text(json.dumps(train_coco, ensure_ascii=False, indent=2), encoding="utf-8")
    val_json_path.write_text(json.dumps(val_coco, ensure_ascii=False, indent=2), encoding="utf-8")

    print("======= 完成：已 split + 已复制图片 =======")
    print(f"总样本: {len(paired)}")
    print(f"train 图片: {len(train_coco['images'])} -> {train_img_dir}")
    print(f"val   图片: {len(val_coco['images'])} -> {val_img_dir}")
    print(f"train 标注: {len(train_coco['annotations'])}")
    print(f"val   标注: {len(val_coco['annotations'])}")
    print(f"类别数: {len(final_categories)}")
    print(f"输出: {train_json_path}")
    print(f"输出: {val_json_path}")


if __name__ == "__main__":
    main()
