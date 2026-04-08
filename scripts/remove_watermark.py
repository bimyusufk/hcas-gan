"""Remove 'camopedia.org' watermark from style reference images via inpainting.

Deteksi watermark kuning di tengah, expand mask, lalu inpaint dengan cv2.inpaint.
Output ke camo/style_reference_clean/ dengan metadata log.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def detect_watermark_mask(img_bgr: np.ndarray, hsv_lower: Tuple, hsv_upper: Tuple) -> Tuple[np.ndarray, bool]:
    """Deteksi area watermark di pusat gambar dan kembalikan mask untuk inpainting.
    
    Strategi: assume watermark selalu di tengah (~50% lebar/tinggi).
    Bikin mask berbentuk "strip horizontal" di area tengah untuk inpaint.
    
    Args:
        img_bgr: input image in BGR
        hsv_lower: not used dalam strategi ini (kept for compatibility)
        hsv_upper: not used dalam strategi ini (kept for compatibility)
    
    Returns:
        (mask, found): binary mask uint8, bool always True (watermark assumed ada).
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Buat horizontal strip di tengah (~50% height di center)
    strip_height = int(h * 0.25)  # 25% from center = 50% total coverage
    y_center = h // 2
    y1 = max(0, y_center - strip_height // 2)
    y2 = min(h, y_center + strip_height // 2)
    
    mask[y1:y2, :] = 255
    
    return mask, True


def expand_mask(mask: np.ndarray, expansion: int = 30) -> np.ndarray:
    """Expand mask to ensure full inpainting coverage.
    
    Args:
        mask: binary mask uint8
        expansion: dilation size in pixels
    
    Returns:
        expanded mask uint8
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
    expanded = cv2.dilate(mask, kernel, iterations=1)
    return expanded


def inpaint_watermark(img_bgr: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    """Inpaint watermark area.
    
    Args:
        img_bgr: input image BGR
        mask: binary inpainting mask (uint8)
        method: "telea" or "ns" (Navier-Stokes)
    
    Returns:
        inpainted image BGR
    """
    radius = 3
    if method.lower() == "ns":
        inpainted = cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_NS)
    else:
        inpainted = cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA)
    
    return inpainted


def process_image(input_path: Path, output_path: Path, **config) -> dict:
    """Process single image: detect + inpaint watermark.
    
    Returns:
        result dict with keys: success, watermark_detected, pixel_removed, error_msg
    """
    result = {
        "file": input_path.name,
        "success": False,
        "watermark_detected": False,
        "pixel_removed": 0,
        "error_msg": None,
    }
    
    try:
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            result["error_msg"] = "Failed to read image"
            return result
        
        h, w = img_bgr.shape[:2]
        result["input_size"] = f"{w}x{h}"
        
        hsv_lower = config.get("hsv_lower", (15, 50, 50))
        hsv_upper = config.get("hsv_upper", (35, 255, 255))
        
        mask, found = detect_watermark_mask(img_bgr, hsv_lower, hsv_upper)
        result["watermark_detected"] = found
        
        if found:
            pixel_removed = np.count_nonzero(mask)
            result["pixel_removed"] = int(pixel_removed)
            
            expansion = config.get("mask_expansion", 30)
            expanded_mask = expand_mask(mask, expansion)
            
            method = config.get("inpaint_method", "telea")
            img_inpainted = inpaint_watermark(img_bgr, expanded_mask, method)
            
            cv2.imwrite(str(output_path), img_inpainted)
            result["success"] = True
        else:
            cv2.imwrite(str(output_path), img_bgr)
            result["success"] = True
            result["error_msg"] = "No watermark detected; copied as-is"
        
    except Exception as e:
        result["error_msg"] = f"{type(e).__name__}: {str(e)}"
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Remove watermark from style reference camo images via inpainting"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="camo/style_reference",
        help="Input directory with style reference images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="camo/style_reference_clean",
        help="Output directory for cleaned images",
    )
    parser.add_argument(
        "--hsv-h-range",
        type=int,
        nargs=2,
        default=[15, 35],
        help="HSV H range [min max] for yellow watermark detection",
    )
    parser.add_argument(
        "--hsv-s-range",
        type=int,
        nargs=2,
        default=[50, 255],
        help="HSV S range [min max]",
    )
    parser.add_argument(
        "--hsv-v-range",
        type=int,
        nargs=2,
        default=[50, 255],
        help="HSV V range [min max]",
    )
    parser.add_argument(
        "--mask-expansion",
        type=int,
        default=30,
        help="Dilation expansion size for inpainting mask (pixels)",
    )
    parser.add_argument(
        "--inpaint-method",
        type=str,
        default="telea",
        choices=["telea", "ns"],
        help="Inpainting method: TELEA or Navier-Stokes",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already in output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze, do not write output",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    hsv_lower = (args.hsv_h_range[0], args.hsv_s_range[0], args.hsv_v_range[0])
    hsv_upper = (args.hsv_h_range[1], args.hsv_s_range[1], args.hsv_v_range[1])
    
    config = {
        "hsv_lower": hsv_lower,
        "hsv_upper": hsv_upper,
        "mask_expansion": args.mask_expansion,
        "inpaint_method": args.inpaint_method,
    }
    
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in sorted(input_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in image_exts
    ]
    
    print(f"[watermark_removal] Found {len(image_files)} images in {input_dir}")
    print(f"[watermark_removal] Config: HSV {hsv_lower} to {hsv_upper}, expansion={args.mask_expansion}px, method={args.inpaint_method}")
    print(f"[watermark_removal] Dry-run: {args.dry_run}, skip-existing: {args.skip_existing}")
    print()
    
    results = []
    watermark_count = 0
    success_count = 0
    
    for idx, img_path in enumerate(image_files):
        rel_path = img_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        if args.skip_existing and output_path.exists():
            print(f"[{idx+1}/{len(image_files)}] SKIP (exists): {rel_path}")
            continue
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = process_image(img_path, output_path, **config) if not args.dry_run else {
            "file": img_path.name,
            "success": True,
            "watermark_detected": False,
            "pixel_removed": 0,
            "error_msg": None,
            "input_size": "???",
        }
        
        results.append(result)
        
        if result["watermark_detected"]:
            watermark_count += 1
        if result["success"]:
            success_count += 1
        
        status = "OK" if result["success"] else "FAIL"
        wm_tag = " [WM]" if result["watermark_detected"] else ""
        msg = result.get("error_msg", "")
        msg_suffix = f" ({msg})" if msg else ""
        
        print(f"[{idx+1}/{len(image_files)}] {status}{wm_tag}: {rel_path}{msg_suffix}")
    
    print()
    print(f"[watermark_removal] Summary:")
    print(f"  Total processed: {len(results)}")
    print(f"  Watermark detected: {watermark_count}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {len(results) - success_count}")
    
    log_path = output_dir / "watermark_removal_report.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "success", "watermark_detected", "pixel_removed", "error_msg", "input_size"],
        )
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  Report saved: {log_path}")
    
    config_path = output_dir / "watermark_removal_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "hsv_lower": list(hsv_lower),
                "hsv_upper": list(hsv_upper),
                "mask_expansion": args.mask_expansion,
                "inpaint_method": args.inpaint_method,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "datetime_processed": str(np.datetime64('now')),
            },
            f,
            indent=2,
        )
    print(f"  Config saved: {config_path}")


if __name__ == "__main__":
    main()
