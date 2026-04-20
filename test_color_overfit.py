from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Detections:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


def list_images(source: str, max_images: int) -> List[Path]:
    p = Path(source)
    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
        return [p]
    if p.is_dir():
        imgs = [x for x in p.rglob("*") if x.suffix.lower() in IMAGE_SUFFIXES]
    else:
        imgs = [x for x in Path().glob(source) if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES]
    imgs = sorted(imgs)
    return imgs[:max_images] if max_images > 0 else imgs


def apply_low_saturation(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] *= 0.2
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_hue_shift(img: np.ndarray, shift: int = 25) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_dark(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=0.78, beta=-28)


def apply_bright(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.18, beta=20)


def apply_warm_cast(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    x[..., 0] *= 0.85  # B
    x[..., 2] *= 1.20  # R
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_gray3(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def predict_detections(model: YOLO, img: np.ndarray, imgsz: int, conf: float, iou: float, device: str) -> Detections:
    result = model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int32),
        )
    return Detections(
        xyxy=result.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
        conf=result.boxes.conf.detach().cpu().numpy().astype(np.float32),
        cls=result.boxes.cls.detach().cpu().numpy().astype(np.int32),
    )


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_ratio(base: Detections, variant: Detections, iou_thres: float = 0.5) -> Tuple[float, float, float, int, int]:
    n_base = len(base.cls)
    n_var = len(variant.cls)

    if n_base == 0:
        if n_var == 0:
            return 1.0, 1.0, 1.0, n_base, n_var
        return 0.0, 0.0, 0.0, n_base, n_var

    used = set()
    matched = 0
    conf_ratios: List[float] = []

    order = np.argsort(-base.conf)
    for bi in order:
        best_j = -1
        best_iou = 0.0
        for j in range(n_var):
            if j in used:
                continue
            if int(variant.cls[j]) != int(base.cls[bi]):
                continue
            iou = iou_xyxy(base.xyxy[bi], variant.xyxy[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_thres:
            used.add(best_j)
            matched += 1
            conf_ratios.append(float(variant.conf[best_j]) / max(float(base.conf[bi]), 1e-6))

    match_r = matched / n_base
    conf_r = float(np.mean(conf_ratios)) if conf_ratios else 0.0
    count_r = n_var / n_base
    return match_r, conf_r, count_r, n_base, n_var


def robust_score(match_r: float, conf_r: float, count_r: float) -> float:
    count_term = max(0.0, 1.0 - abs(1.0 - count_r))
    return 0.6 * match_r + 0.3 * min(conf_r, 1.5) / 1.5 + 0.1 * count_term


def main() -> None:
    parser = argparse.ArgumentParser(description="Color overfitting stress test for YOLO detectors.")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image file, directory, or glob pattern")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--save-dir", type=str, default="runs/color_overfit_test")
    args = parser.parse_args()

    images = list_images(args.source, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found from source: {args.source}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "gray3": apply_gray3,
        "low_saturation": apply_low_saturation,
        "hue_shift": apply_hue_shift,
        "dark": apply_dark,
        "bright": apply_bright,
        "warm_cast": apply_warm_cast,
    }

    rows = []

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        base = predict_detections(model, img, args.imgsz, args.conf, args.iou, args.device)

        for name, fn in transforms.items():
            img_variant = fn(img)
            pred_var = predict_detections(model, img_variant, args.imgsz, args.conf, args.iou, args.device)
            match_r, conf_r, count_r, n_base, n_var = match_ratio(base, pred_var)
            score = robust_score(match_r, conf_r, count_r)
            rows.append(
                {
                    "image": str(img_path),
                    "variant": name,
                    "base_det": n_base,
                    "variant_det": n_var,
                    "match_ratio": round(match_r, 6),
                    "conf_ratio": round(conf_r, 6),
                    "count_ratio": round(count_r, 6),
                    "score": round(score, 6),
                }
            )

        if idx % 20 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)} images")

    if not rows:
        raise RuntimeError("No valid predictions collected. Please check model/source/conf settings.")

    summary: Dict[str, Dict[str, float]] = {}
    for name in transforms:
        part = [r for r in rows if r["variant"] == name]
        summary[name] = {
            "images": len(part),
            "match_ratio_mean": float(np.mean([r["match_ratio"] for r in part])),
            "conf_ratio_mean": float(np.mean([r["conf_ratio"] for r in part])),
            "count_ratio_mean": float(np.mean([r["count_ratio"] for r in part])),
            "score_mean": float(np.mean([r["score"] for r in part])),
        }

    detail_csv = save_dir / "color_overfit_detail.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "variant", "base_det", "variant_det", "match_ratio", "conf_ratio", "count_ratio", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_json = save_dir / "color_overfit_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Color Robustness Summary ===")
    for k, v in sorted(summary.items(), key=lambda x: x[1]["score_mean"]):
        print(
            f"{k:14s} score={v['score_mean']:.4f} "
            f"match={v['match_ratio_mean']:.4f} conf={v['conf_ratio_mean']:.4f} count={v['count_ratio_mean']:.4f}"
        )

    global_score = float(np.mean([v["score_mean"] for v in summary.values()]))
    print(f"\nGlobal color-robust score: {global_score:.4f}")
    print(f"Detail CSV: {detail_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
