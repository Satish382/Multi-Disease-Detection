from PIL import Image, ImageStat, ImageFilter
import numpy as np

def validate_medical_image(image):
    # --- 1. Size Check ---
    width, height = image.size
    print(f"Size: {width}x{height}")
    min_dim = 100
    if width < min_dim or height < min_dim:
        print("FAIL: Too small")
        return False, "Image too small"

    # --- 2. Aspect Ratio Check ---
    aspect_ratio = width / height
    print(f"Aspect Ratio: {aspect_ratio}")
    if not (0.5 <= aspect_ratio <= 2.0):
         print("FAIL: Bad aspect ratio")
         return False, "Invalid aspect ratio"

    # --- 3. Color Variance Check (Detect non-grayscale) ---
    # Split into RGB channels
    img_rgb = image.convert('RGB')
    r, g, b = img_rgb.split()
    # Calculate variance between channels (if grayscale, variance is near 0)
    # We take the difference between channels
    stat_diff_rg = ImageStat.Stat(ImageChops.difference(r, g))
    stat_diff_gb = ImageStat.Stat(ImageChops.difference(g, b))
    stat_diff_rb = ImageStat.Stat(ImageChops.difference(r, b))
    
    # Average pixel difference between channels
    avg_diff = (sum(stat_diff_rg.mean) + sum(stat_diff_gb.mean) + sum(stat_diff_rb.mean)) / 3
    print(f"Color Variance (Avg Diff): {avg_diff}")
    
    # Threshold: MRI/CT are grayscale. Even heatmaps are overlay, but raw input should be gray.
    # Allow some tolerance for compression artifact or slight tint
    if avg_diff > 50:  # Adjust threshold as needed (was 50 in app.py)
        print("FAIL: Too colorful")
        return False, "Image validation failed: Image contains too much color for a medical scan."

    # --- 4. Brightness/Contrast Check ---
    grayscale = image.convert('L')
    stat = ImageStat.Stat(grayscale)
    mean_brightness = stat.mean[0]
    std_dev_contrast = stat.stddev[0]
    
    print(f"Brightness: {mean_brightness}")
    print(f"Contrast (StdDev): {std_dev_contrast}")
    
    if mean_brightness < 5 or mean_brightness > 250:
        print("FAIL: Brightness out of range")
        return False, "Image too dark or too bright"
        
    if std_dev_contrast < 10:
        print("FAIL: Contrast too low")
        return False, "Image contrast too low (solid color?)"

    # --- 5. Structure/Edge Check ---
    # Medical scans have complex structure. Random noise has high entropy, flat images have low.
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_mean = edge_stat.mean[0]
    print(f"Edge Density (Mean): {edge_mean}")
    
    if edge_mean < 2: # Very few edges
        print("FAIL: Too few edges")
        return False, "Image lacks structural detail"

    print("PASS: Validation successful")
    
    # --- 6. Corner Analysis (New Idea) ---
    # Brain scans usually have dark backgrounds. Check 4 corners.
    w, h = image.size
    corners = [
        (0, 0, 20, 20),           # Top-left
        (w-20, 0, w, 20),         # Top-right
        (0, h-20, 20, h),         # Bottom-left
        (w-20, h-20, w, h)        # Bottom-right
    ]
    corner_brightness = []
    for box in corners:
        region = grayscale.crop(box)
        stat = ImageStat.Stat(region)
        corner_brightness.append(stat.mean[0])
    
    avg_corner_brightness = sum(corner_brightness) / 4
    print(f"Corner Brightness (Avg): {avg_corner_brightness}")
    if avg_corner_brightness > 40: # Threshold for "dark background"
         print("FAIL: Corners too bright (likely not a brain scan)")
    else:
         print("PASS: Corners dark enough")

    return True, "Valid"

from PIL import ImageChops
import os

target = "images.jpg"
if os.path.exists(target):
    print(f"--- Analyzing {target} ---")
    img = Image.open(target)
    validate_medical_image(img)
else:
    print(f"{target} not found")
