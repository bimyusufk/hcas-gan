#!/usr/bin/env python3
"""
Remove watermarks by CROPPING instead of inpainting.
Removes the center region containing the watermark, preserving image quality.
Much better results than inpainting on patterned images.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import csv
from tqdm import tqdm


def crop_watermark_region(img_bgr, crop_height_percent=12):
    """
    Crop out the center region containing watermark.
    Returns the image with watermark region removed (top and bottom combined).
    
    This preserves image quality by avoiding inpainting blur.
    """
    h, w = img_bgr.shape[:2]
    
    # Height to remove from center
    remove_height = max(int(h * crop_height_percent / 100), 20)
    
    # Crop coordinates
    y1 = h // 2 - remove_height // 2
    y2 = y1 + remove_height
    
    # Create cropped image (remove center region)
    top_part = img_bgr[:y1, :]
    bottom_part = img_bgr[y2:, :]
    
    # Stack back (images without watermark in middle)
    if top_part.shape[0] > 0 and bottom_part.shape[0] > 0:
        result = np.vstack([top_part, bottom_part])
    elif top_part.shape[0] > 0:
        result = top_part
    else:
        result = bottom_part
    
    return result, (y1, y2, remove_height)


def process_image(input_path, output_path, crop_height_percent=12):
    """Process single image: crop out watermark region."""
    try:
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            return {"file": input_path.name, "success": False, "error_msg": "Failed to load image"}
        
        original_h, original_w = img_bgr.shape[:2]
        
        # Crop watermark region
        result, (y1, y2, crop_h) = crop_watermark_region(img_bgr, crop_height_percent=crop_height_percent)
        new_h, new_w = result.shape[:2]
        
        cv2.imwrite(str(output_path), result)
        
        return {
            "file": input_path.name,
            "success": True,
            "original_size": f"{original_w}x{original_h}",
            "cropped_size": f"{new_w}x{new_h}",
            "crop_height_px": crop_h,
            "crop_region": f"y={y1}-{y2}",
            "height_retained_percent": round(100 * new_h / original_h, 1),
            "error_msg": None
        }
    
    except Exception as e:
        return {
            "file": input_path.name,
            "success": False,
            "error_msg": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Remove watermarks by cropping center region")
    parser.add_argument("--input-dir", default="camo/style_references", help="Input directory")
    parser.add_argument("--output-dir", default="camo/style_reference_clean", help="Output directory")
    parser.add_argument("--crop-height-percent", type=float, default=12, help="% of image height to crop (default: 12%)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    print(f"[watermark_removal_crop] Found {len(image_files)} images in {input_dir}")
    print(f"[watermark_removal_crop] Method: CENTER CROP (no inpainting blur!)")
    print(f"[watermark_removal_crop] Removing {args.crop_height_percent}% of image height\n")
    
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
        
        result = process_image(input_path, output_path, crop_height_percent=args.crop_height_percent)
        results.append(result)
        
        if result["success"]:
            success_count += 1
            retained = result.get("height_retained_percent", 0)
            print(f"[{idx}/{len(image_files)}] OK: {input_path.name} → {retained}% height retained")
        else:
            failed_count += 1
            print(f"[{idx}/{len(image_files)}] FAIL: {input_path.name} - {result.get('error_msg')}")
    
    # Save reports
    report_path = output_dir / "watermark_removal_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "success", "original_size", "cropped_size", "crop_height_px", "crop_region", "height_retained_percent", "error_msg"])
        writer.writeheader()
        writer.writerows(results)
    
    config_path = output_dir / "watermark_removal_config.json"
    config = {
        "method": "center crop (preserves image quality, no inpainting blur)",
        "crop_height_percent": args.crop_height_percent,
        "total_processed": len(results),
        "success": success_count,
        "failed": failed_count
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[watermark_removal_crop] Summary:")
    print(f"  Total processed: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Report saved: {report_path}")
    print(f"  Config saved: {config_path}")


if __name__ == "__main__":
    main()
