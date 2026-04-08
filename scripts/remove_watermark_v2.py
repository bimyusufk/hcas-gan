#!/usr/bin/env python3
"""
Improved watermark removal using color-based detection.
Targets only the yellow watermark pixels, not the entire center strip.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import csv
from tqdm import tqdm


def detect_watermark_bounding_box(img_bgr, hsv_lower=(15, 50, 50), hsv_upper=(35, 255, 255)):
    """
    Detect yellow watermark by color range, return bounding box.
    More targeted than a full horizontal strip.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    # Find contours of yellow regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # Get bounding box of all yellow regions
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    return (x, y, w, h), mask


def expand_bbox(bbox, img_h, img_w, expansion=5):
    """Expand bounding box by fixed pixels."""
    x, y, w, h = bbox
    x = max(0, x - expansion)
    y = max(0, y - expansion)
    w = min(img_w - x, w + 2 * expansion)
    h = min(img_h - y, h + 2 * expansion)
    return (x, y, w, h)


def create_mask_from_bbox(img_h, img_w, bbox):
    """Create inpainting mask from bounding box."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x, y, w, h = bbox
    mask[y:y+h, x:x+w] = 255
    return mask


def inpaint_watermark(img_bgr, mask, method="telea"):
    """Inpaint the masked region."""
    if method == "telea":
        result = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    else:  # navier-stokes
        result = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_NS)
    return result


def process_image(input_path, output_path, expansion=5, inpaint_method="telea"):
    """Process single image: detect yellow watermark, inpaint."""
    try:
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            return {"file": input_path.name, "success": False, "error_msg": "Failed to load image"}
        
        img_h, img_w = img_bgr.shape[:2]
        
        # Detect yellow watermark bounding box
        bbox, color_mask = detect_watermark_bounding_box(img_bgr)
        
        if bbox is None:
            # No watermark detected, just copy
            cv2.imwrite(str(output_path), img_bgr)
            return {
                "file": input_path.name,
                "success": True,
                "watermark_detected": False,
                "pixels_covered": 0,
                "input_size": f"{img_w}x{img_h}",
                "error_msg": None
            }
        
        # Expand the bounding box slightly for safety
        expanded_bbox = expand_bbox(bbox, img_h, img_w, expansion=expansion)
        
        # Create precise mask from expanded bbox
        inpaint_mask = create_mask_from_bbox(img_h, img_w, expanded_bbox)
        pixels_covered = np.sum(inpaint_mask > 0)
        
        # Inpaint
        result = inpaint_watermark(img_bgr, inpaint_mask, method=inpaint_method)
        
        cv2.imwrite(str(output_path), result)
        
        return {
            "file": input_path.name,
            "success": True,
            "watermark_detected": True,
            "pixels_covered": int(pixels_covered),
            "input_size": f"{img_w}x{img_h}",
            "bbox": str(expanded_bbox),
            "error_msg": None
        }
    
    except Exception as e:
        return {
            "file": input_path.name,
            "success": False,
            "error_msg": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Remove watermarks from images using color-based detection")
    parser.add_argument("--input-dir", default="camo/style_references", help="Input directory")
    parser.add_argument("--output-dir", default="camo/style_reference_clean", help="Output directory")
    parser.add_argument("--expansion", type=int, default=5, help="Bounding box expansion in pixels")
    parser.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea", help="Inpaint method")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no actual processing)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    print(f"[watermark_removal_v2] Found {len(image_files)} images in {input_dir}")
    print(f"[watermark_removal_v2] Config: color-based detection, expansion={args.expansion}px, method={args.inpaint_method}")
    print(f"[watermark_removal_v2] Dry-run: {args.dry_run}, skip-existing: {args.skip_existing}\n")
    
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
        
        result = process_image(input_path, output_path, expansion=args.expansion, inpaint_method=args.inpaint_method)
        results.append(result)
        
        if result["success"]:
            success_count += 1
            status = "[WM]" if result.get("watermark_detected") else "[OK]"
            print(f"[{idx}/{len(image_files)}] {status}: {input_path.name}")
        else:
            failed_count += 1
            print(f"[{idx}/{len(image_files)}] FAIL: {input_path.name} - {result.get('error_msg')}")
    
    # Save reports
    report_path = output_dir / "watermark_removal_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "success", "watermark_detected", "pixels_covered", "input_size", "bbox", "error_msg"])
        writer.writeheader()
        writer.writerows(results)
    
    config_path = output_dir / "watermark_removal_config.json"
    config = {
        "method": "color-based detection (HSV yellow)",
        "expansion_px": args.expansion,
        "inpaint_method": args.inpaint_method,
        "total_processed": len(results),
        "success": success_count,
        "failed": failed_count
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[watermark_removal_v2] Summary:")
    print(f"  Total processed: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Report saved: {report_path}")
    print(f"  Config saved: {config_path}")


if __name__ == "__main__":
    main()
