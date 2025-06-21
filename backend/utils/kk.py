import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import shutil

def read_multiband_image(filepath):
    img = tiff.imread(filepath)
    if img is None or img.ndim != 3 or img.shape[0] < 6:
        raise ValueError(f"Invalid image format or not enough bands: {filepath}")
    img = np.transpose(img[:6], (1, 2, 0))  # Convert (C, H, W) → (H, W, C)
    return img.astype(np.float32)

def calculate_depletion_score(img):
    B02, B03, B04, B08, B11, B12 = [img[:, :, i] for i in range(6)]
    eps = 1e-6
    ndvi = (B08 - B04) / (B08 + B04 + eps)
    ndbi = (B11 - B08) / (B11 + B08 + eps)

    ndvi_mean = np.mean(ndvi)
    ndbi_mean = np.mean(ndbi)
    score = ndbi_mean - ndvi_mean

    print(f"📸 NDVI: {ndvi_mean:.4f}, NDBI: {ndbi_mean:.4f}, Score: {score:.4f}")
    return score

def classify_images(base_folder="data", output_folder="classified"):
    os.makedirs(output_folder, exist_ok=True)
    image_scores = []

    print("\n🔍 Reading and scoring images...\n")
    for state in tqdm(os.listdir(base_folder), desc="Processing states"):
        state_path = os.path.join(base_folder, state)
        if not os.path.isdir(state_path):
            continue

        for folder in os.listdir(state_path):
            img_path = os.path.join(state_path, folder, "response.tiff")
            if not os.path.isfile(img_path):
                continue

            try:
                img = read_multiband_image(img_path)
                score = calculate_depletion_score(img)
                image_scores.append((state, folder, img_path, score))
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")

    if not image_scores:
        print("❌ No valid images processed.")
        return

    scores_only = [score for _, _, _, score in image_scores]
    low_thres = np.percentile(scores_only, 33)
    high_thres = np.percentile(scores_only, 66)

    print(f"\n📊 Thresholds:")
    print(f"   🔵 LOW    < {low_thres:.4f}")
    print(f"   🟡 MEDIUM : {low_thres:.4f} – {high_thres:.4f}")
    print(f"   🔴 HIGH   > {high_thres:.4f}\n")

    for state, folder, img_path, score in image_scores:
        label = "low" if score < low_thres else "high" if score > high_thres else "medium"
        out_path = os.path.join(output_folder, label, state)
        os.makedirs(out_path, exist_ok=True)
        dest_path = os.path.join(out_path, f"{state}_{folder}.tiff")
        shutil.copy(img_path, dest_path)

    print(f"\n✅ Classification complete!")
    print(f"📁 Images saved to: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    classify_images()
