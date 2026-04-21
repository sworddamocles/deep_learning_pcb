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


def get_class_name(names: Dict[int, str] | List[str] | None, class_id: int) -> str:
    if names is None:
        return str(class_id)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def add_label_header(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    header_h = 52 if subtitle else 36
    canvas = np.full((img.shape[0] + header_h, img.shape[1], 3), 24, dtype=np.uint8)
    canvas[:header_h] = (32, 32, 32)
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)
    canvas[header_h:] = img
    return canvas


def draw_detections(img: np.ndarray, dets: Detections, names: Dict[int, str] | List[str] | None, box_color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    for xyxy, conf, cls in zip(dets.xyxy, dets.conf, dets.cls):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1 = max(0, min(x1, out.shape[1] - 1))
        y1 = max(0, min(y1, out.shape[0] - 1))
        x2 = max(0, min(x2, out.shape[1] - 1))
        y2 = max(0, min(y2, out.shape[0] - 1))
        label = f"{get_class_name(names, int(cls))} {float(conf):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_keep_aspect(img: np.ndarray, target_height: int) -> np.ndarray:
    scale = target_height / max(1, img.shape[0])
    target_width = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)


def pad_to_width(img: np.ndarray, target_width: int, fill_value: int = 18) -> np.ndarray:
    if img.shape[1] >= target_width:
        return img
    pad = np.full((img.shape[0], target_width - img.shape[1], 3), fill_value, dtype=np.uint8)
    return np.hstack((img, pad))


def build_batch_mosaic(
    items: List[Dict[str, object]],
    names: Dict[int, str] | List[str] | None,
    out_path: Path,
    target_height: int = 360,
    gap: int = 16,
) -> None:
    rows: List[np.ndarray] = []
    for item in items:
        img_path = Path(item["image_path"])
        base_img = draw_detections(
            resize_keep_aspect(item["base_img"], target_height),
            item["base_det"],
            names,
            (46, 204, 113),
        )
        compare_img = draw_detections(
            resize_keep_aspect(item["compare_img"], target_height),
            item["compare_det"],
            names,
            (231, 76, 60),
        )

        base_img = add_label_header(
            base_img,
            f"ORIGINAL | {img_path.name}",
            f"detections: {len(item['base_det'].cls)}",
        )
        compare_img = add_label_header(
            compare_img,
            f"COMPARE | {item['compare_name']}",
            f"score={item['compare_score']:.4f}  match={item['compare_match']:.4f}  conf={item['compare_conf']:.4f}  count={item['compare_count']:.4f}",
        )

        row_height = max(base_img.shape[0], compare_img.shape[0])
        if base_img.shape[0] < row_height:
            base_img = cv2.copyMakeBorder(base_img, 0, row_height - base_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        if compare_img.shape[0] < row_height:
            compare_img = cv2.copyMakeBorder(compare_img, 0, row_height - compare_img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))

        row = np.hstack((base_img, np.full((row_height, gap, 3), 24, dtype=np.uint8), compare_img))
        rows.append(row)

    if not rows:
        return

    max_width = max(row.shape[1] for row in rows)
    padded_rows = [pad_to_width(row, max_width) for row in rows]
    mosaic = padded_rows[0]
    for row in padded_rows[1:]:
        spacer = np.full((gap, max_width, 3), 24, dtype=np.uint8)
        mosaic = np.vstack((mosaic, spacer, row))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mosaic)


def result_to_detections(result: object) -> Detections:
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


def predict_detections(model: YOLO, img: np.ndarray, imgsz: int, conf: float, iou: float, device: str) -> Detections:
    return result_to_detections(model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0])


def predict_detections_many(
    model: YOLO,
    imgs: List[np.ndarray],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> List[Detections]:
    results = model.predict(source=imgs, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    return [result_to_detections(result) for result in results]


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
    parser.add_argument("--batch-size", type=int, default=20, help="Number of images per output batch")
    parser.add_argument(
        "--compare-variant",
        type=str,
        default="worst",
        help="Variant to show in the comparison panel, or 'worst' to use the lowest-scoring transform",
    )
    parser.add_argument("--viz-height", type=int, default=360, help="Visualization tile height for batch mosaics")
    parser.add_argument("--save-dir", type=str, default="runs/color_overfit_test")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.viz_height <= 0:
        raise ValueError("--viz-height must be greater than 0")

    images = list_images(args.source, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found from source: {args.source}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    class_names = getattr(model, "names", None)

    transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "gray3": apply_gray3,
        "low_saturation": apply_low_saturation,
        "hue_shift": apply_hue_shift,
        "dark": apply_dark,
        "bright": apply_bright,
        "warm_cast": apply_warm_cast,
    }

    rows = []
    batch_items: List[Dict[str, object]] = []
    batch_index = 0

    def flush_batch(items: List[Dict[str, object]], current_batch_index: int) -> None:
        if not items:
            return
        batch_score = float(np.mean([float(item["compare_score"]) for item in items]))
        batch_match = float(np.mean([float(item["compare_match"]) for item in items]))
        batch_conf = float(np.mean([float(item["compare_conf"]) for item in items]))
        batch_count = float(np.mean([float(item["compare_count"]) for item in items]))
        mosaic_path = save_dir / f"color_overfit_batch_{current_batch_index:03d}.jpg"
        build_batch_mosaic(items, class_names, mosaic_path, target_height=args.viz_height)
        print(
            f"Batch {current_batch_index:03d} done: images={len(items)} "
            f"score={batch_score:.4f} match={batch_match:.4f} conf={batch_conf:.4f} count={batch_count:.4f}"
        )
        print(f"Batch mosaic: {mosaic_path}")

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        variant_items = list(transforms.items())
        variant_images = [fn(img) for _, fn in variant_items]
        predictions = predict_detections_many(
            model,
            [img, *variant_images],
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )

        base = predictions[0]
        variant_results: List[Dict[str, object]] = []

        for (name, _), img_variant, pred_var in zip(variant_items, variant_images, predictions[1:]):
            match_r, conf_r, count_r, n_base, n_var = match_ratio(base, pred_var)
            score = robust_score(match_r, conf_r, count_r)
            variant_results.append(
                {
                    "name": name,
                    "img": img_variant,
                    "pred": pred_var,
                    "match_r": match_r,
                    "conf_r": conf_r,
                    "count_r": count_r,
                    "score": score,
                }
            )
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

        if args.compare_variant == "worst":
            compare_result = min(variant_results, key=lambda item: float(item["score"]))
        else:
            compare_result = next((item for item in variant_results if item["name"] == args.compare_variant), None)
            if compare_result is None:
                valid = ", ".join(["worst", *transforms.keys()])
                raise ValueError(f"Unknown --compare-variant '{args.compare_variant}'. Valid options: {valid}")

        batch_items.append(
            {
                "image_path": img_path,
                "base_img": img,
                "base_det": base,
                "compare_name": compare_result["name"],
                "compare_img": compare_result["img"],
                "compare_det": compare_result["pred"],
                "compare_score": compare_result["score"],
                "compare_match": compare_result["match_r"],
                "compare_conf": compare_result["conf_r"],
                "compare_count": compare_result["count_r"],
            }
        )

        if len(batch_items) >= args.batch_size:
            batch_index += 1
            flush_batch(batch_items, batch_index)
            batch_items = []

    if batch_items:
        batch_index += 1
        flush_batch(batch_items, batch_index)

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
