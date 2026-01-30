# -*- coding: utf-8 -*-
"""
Batch inference script (config-on-top, no CLI)

Usage:
    python tools/image_demo_batch_config.py

What it does:
- Read images from INPUT (file/folder/glob)
- Run MMDetection DetInferencer in batches
- Save visualization images to OUT_DIR
- (Optional) save predictions to jsonl

Note:
- For your custom dataset/classes, make sure MODEL is your training config
  (the one containing num_classes + metainfo).
"""

import json
from glob import glob
from pathlib import Path
from typing import List

from mmengine.logging import print_log
from mmdet.apis import DetInferencer

# =========================
# ✅ 配置区（只改这里）
# =========================
CONFIG = {
    # 输入：可以是单张图片 / 文件夹 / glob（支持 ** 递归）
    "INPUT": r"D:\zhanlan\segment_coco_out\val2017",  # 例如：r"D:/imgs" 或 r"D:/imgs/**/*.jpg"

    # 输出目录：保存可视化图片 & 预测结果
    "OUT_DIR": r"D:/zhanlan/infer_outputs12",

    # 模型配置：一定要用你训练时的那份 config（包含 num_classes + metainfo）
    "MODEL_CONFIG": r"D:/zhanlanProject/mmdetection/zhanlan/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",

    # 权重：训练得到的 epoch_x.pth
    "WEIGHTS": r"D:/zhanlanProject/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth",

    # 设备
    "DEVICE": "cuda:0",  # 或 "cpu"

    # 推理阈值
    "PRED_SCORE_THR": 0.6,

    # batch size
    "BATCH_SIZE": 4,

    # 是否递归搜图（当 INPUT 是文件夹时有效）
    "RECURSIVE": True,

    # 是否弹窗显示（Windows 会很慢，不建议开）
    "SHOW": False,

    # 是否保存可视化图
    "SAVE_VIS": True,

    # 是否保存原始预测（inferencer 也可能会额外保存json）
    "SAVE_PRED": False,

    # 是否把每张图的预测汇总保存成 jsonl（推荐开，便于后处理）
    "SAVE_JSONL": True,
    "JSONL_NAME": "predictions.jsonl",

    # palette: 'none'/'random'/'coco' 等
    "PALETTE": "random",
}

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(inputs: str, recursive: bool) -> List[str]:
    """Collect image paths from file / folder / glob."""
    p = Path(inputs)

    # 1) single file
    if p.exists() and p.is_file():
        if p.suffix.lower() in IMG_EXTS:
            return [str(p)]
        return []

    # 2) folder
    if p.exists() and p.is_dir():
        pattern = "**/*" if recursive else "*"
        files = [f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in IMG_EXTS]
        return [str(f) for f in sorted(files)]

    # 3) glob
    files = glob(inputs, recursive=True)
    files = [f for f in files if Path(f).is_file() and Path(f).suffix.lower() in IMG_EXTS]
    return sorted(files)


def main():
    cfg = CONFIG

    out_dir = Path(cfg["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    img_list = collect_images(cfg["INPUT"], recursive=cfg["RECURSIVE"])
    if not img_list:
        raise FileNotFoundError(
            f"未找到任何图片：INPUT={cfg['INPUT']}\n"
            f"支持：单文件 / 文件夹 / glob（如 D:/imgs/**/*.jpg）\n"
            f"支持后缀：{sorted(IMG_EXTS)}"
        )

    print_log(f"共找到 {len(img_list)} 张图片，将开始批量推理……")

    inferencer = DetInferencer(
        model=cfg["MODEL_CONFIG"],
        weights=cfg["WEIGHTS"],
        device=cfg["DEVICE"],
        palette=cfg["PALETTE"],
    )

    all_jsonl_rows = []

    bs = int(cfg["BATCH_SIZE"])
    for i in range(0, len(img_list), bs):
        batch_paths = img_list[i:i + bs]

        # 注意：DetInferencer 内部已经支持 list inputs
        results = inferencer(
            inputs=batch_paths,
            out_dir=str(out_dir) if (cfg["SAVE_VIS"] or cfg["SAVE_PRED"]) else "",
            pred_score_thr=float(cfg["PRED_SCORE_THR"]),
            batch_size=len(batch_paths),
            show=bool(cfg["SHOW"]),
            no_save_vis=not bool(cfg["SAVE_VIS"]),
            no_save_pred=not bool(cfg["SAVE_PRED"]),
            print_result=False,
        )

        if cfg["SAVE_JSONL"]:
            preds = results.get("predictions", None) or results.get("preds", None)
            if preds is None:
                # 兜底：直接把整个 results 记下来（不同版本结构可能不同）
                preds = [{"inputs": batch_paths, "results": results}]

            # 补充 img_path 方便后处理
            for j, item in enumerate(preds):
                if isinstance(item, dict) and ("img_path" not in item and "input" not in item):
                    item["img_path"] = batch_paths[min(j, len(batch_paths) - 1)]
            all_jsonl_rows.extend(preds)

        print_log(f"进度：{min(i + bs, len(img_list))}/{len(img_list)}")

    if cfg["SAVE_JSONL"]:
        jsonl_path = out_dir / cfg["JSONL_NAME"]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in all_jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print_log(f"已保存 jsonl：{jsonl_path}")

    if cfg["SAVE_VIS"]:
        print_log(f"已保存可视化结果图到：{out_dir}")

    print_log("批量推理完成 ✅")


if __name__ == "__main__":
    main()
