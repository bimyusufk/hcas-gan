#!/usr/bin/env python3
"""
Watermark removal with center-strip detection + minimal expansion.
Targets only the center region where watermarks typically appear.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import csv
from tqdm import tqdm


def detect_watermark_strip_mask(img_bgr, strip_height_percent=15):
    """
    Create a horizontal strip mask at the center of the image.
    Only covers the region where watermark text typically appears.
    Default: 15% of image height (much smaller than before).
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Compute the strip region
    strip_height = max(int(h * strip_height_percent / 100), 20)  # min 20px
    y_center = h // 2
    y1 = max(0, y_center - strip_height // 2)
    y2 = min(h, y_center + strip_height // 2)
    
    # Create mask for watermark region
    mask[y1:y2, :] = 255
    
    return mask, (y1, y2, strip_height)


def expand_mask(mask, expansion=2):
    """Dilate mask by a small amount."""
    if expansion < 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
    return cv2.dilate(mask, kernel, iterations=1)


def inpaint_watermark(img_bgr, mask, method="telea"):
    """Inpaint the masked region."""
    if method == "telea":
        result = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    else:  # navier-stokes
        result = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_NS)
    return result


def process_image(input_path, output_path, strip_height_percent=15, expansion=2, inpaint_method="telea"):
    """Process single image: detect center watermark strip, inpaint minimally."""
    try:
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            return {"file": input_path.name, "success": False, "error_msg": "Failed to load image"}
        
        img_h, img_w = img_bgr.shape[:2]
        
        # Create strip mask
        strip_mask, (y1, y2, strip_h) = detect_watermark_strip_mask(img_bgr, strip_height_percent=strip_height_percent)
        
        # Expand minimally
        inpaint_mask = expand_mask(strip_mask, expansion=expansion)
        
        pixels_covered = np.sum(inpaint_mask > 0)
        
        # Inpaint
        result = inpaint_watermark(img_bgr, inpaint_mask, method=inpaint_method)
        
        cv2.imwrite(str(output_path), result)
        
        return {
            "file": input_path.name,
            "success": True,
            "watermark_detected": True,
            "strip_region": f"y={y1}-{y2} (h={strip_h})",
            "pixels_covered": int(pixels_covered),
            "input_size": f"{img_w}x{img_h}",
            "error_msg": None
        }
    
    except Exception as e:
        return {
            "file": input_path.name,
            "success": False,
            "error_msg": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Remove watermarks using conservative center-strip detection")
    parser.add_argument("--input-dir", default="camo/style_references", help="Input directory")
    parser.add_argument("--output-dir", default="camo/style_reference_clean", help="Output directory")
    parser.add_argument("--strip-height-percent", type=float, default=15, help="Strip height as % of image height (default: 15%)")
    parser.add_argument("--expansion", type=int, default=2, help="Mask expansion in pixels (default: 2)")
    parser.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea", help="Inpaint method")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no actual processing)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    print(f"[watermark_removal_v3] Found {len(image_files)} images in {input_dir}")
    print(f"[watermark_removal_v3] Config: center-strip {args.strip_height_percent}% + {args.expansion}px expansion, method={args.inpaint_method}")
    print(f"[watermark_removal_v3] Dry-run: {args.dry_run}, skip-existing: {args.skip_existing}\n")
    
    results = []
    success_count = 0
    failed_count = 0
    
    for idx, input_path in enumerate(tqdm(image_files, desc="Processing"), 1):
        output_path = output_dir / input_path.name
        
        if output_path.exists() and args.skip_existing:
            continue
        
        if args.dry_run:
            print(f"[{idx}/{len(image_files)}] DRY-RUN: {input_path.name}")
            continue
        
        result = process_image(input_path, output_path, 
                              strip_height_percent=args.strip_height_percent,
                              expansion=args.expansion, 
                              inpaint_method=args.inpaint_method)
        results.append(result)
        
        if result["success"]:
            success_count += 1
            print(f"[{idx}/{len(image_files)}] OK: {input_path.name}")
        else:
            failed_count += 1
            print(f"[{idx}/{len(image_files)}] FAIL: {input_path.name} - {result.get('error_msg')}")
    
    # Save reports
    report_path = output_dir / "watermark_removal_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "success", "watermark_detected", "strip_region", "pixels_covered", "input_size", "error_msg"])
        writer.writeheader()
        writer.writerows(results)
    
    config_path = output_dir / "watermark_removal_config.json"
    config = {
        "method": "center-strip detection (conservative)",
        "strip_height_percent": args.strip_height_percent,
        "expansion_px": args.expansion,
        "inpaint_method": args.inpaint_method,
        "total_processed": len(results),
        "success": success_count,
        "failed": failed_count
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[watermark_removal_v3] Summary:")
    print(f"  Total processed: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Report saved: {report_path}")
    print(f"  Config saved: {config_path}")


if __name__ == "__main__":
    main()
