"""Generate diverse evaluation test cases for TrustLens accuracy & robustness evaluation."""
import os
import cv2
import numpy as np
import insightface.data

TEST_DIR = os.path.join("demo", "test_cases")
os.makedirs(TEST_DIR, exist_ok=True)

# 1. Frontal High Quality (from public_face_demo.jpg)
base_path = os.path.join("demo", "public_face_demo.jpg")
if not os.path.exists(base_path):
    base_path = os.path.join("demo", "sample_face.jpg")

base_img = cv2.imread(base_path)
cv2.imwrite(os.path.join(TEST_DIR, "01_frontal_hq.jpg"), base_img)

# 2. Slight Pose / Angle Variation (Rotated 12 degrees with border reflection)
h, w = base_img.shape[:2]
M = cv2.getRotationMatrix2D((w/2, h/2), 12, 1.0)
rotated = cv2.warpAffine(base_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
cv2.imwrite(os.path.join(TEST_DIR, "02_pose_angle.jpg"), rotated)

# 3. Low Resolution / Compressed (Resized down and back up)
small = cv2.resize(base_img, (72, 72), interpolation=cv2.INTER_AREA)
low_res = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(TEST_DIR, "03_low_res.jpg"), low_res)

# 4. Lighting / Illumination Variation (Directional light gradient + shadow)
gradient = np.tile(np.linspace(0.45, 1.3, w), (h, 1))
lighting = np.clip(base_img * gradient[:, :, np.newaxis], 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(TEST_DIR, "04_lighting_variation.jpg"), lighting)

# 5. Multi-Face Group Image (using sample_face.jpg or t1.jpg)
try:
    img_t1 = insightface.data.get_image("t1")
    cv2.imwrite(os.path.join(TEST_DIR, "05_multi_face_group.jpg"), img_t1)
except Exception:
    sample_path = os.path.join("demo", "sample_face.jpg")
    if os.path.exists(sample_path):
        cv2.imwrite(os.path.join(TEST_DIR, "05_multi_face_group.jpg"), cv2.imread(sample_path))

# 6. Un-indexed / Private synthetic face (No Web Match case)
# Create a unique procedural synthetic face collage that does not exist anywhere on the web
synth = np.full((320, 320, 3), (220, 220, 220), dtype=np.uint8)
# Base face oval
cv2.ellipse(synth, (160, 160), (75, 105), 0, 0, 360, (170, 195, 230), -1)
# Eyes
cv2.ellipse(synth, (130, 140), (14, 8), 0, 0, 360, (255, 255, 255), -1)
cv2.circle(synth, (130, 140), 5, (80, 50, 30), -1)
cv2.ellipse(synth, (190, 140), (14, 8), 0, 0, 360, (255, 255, 255), -1)
cv2.circle(synth, (190, 140), 5, (80, 50, 30), -1)
# Eyebrows
cv2.line(synth, (115, 125), (145, 128), (50, 40, 30), 3)
cv2.line(synth, (175, 128), (205, 125), (50, 40, 30), 3)
# Nose
cv2.line(synth, (160, 145), (155, 175), (140, 160, 190), 2)
cv2.line(synth, (155, 175), (165, 175), (140, 160, 190), 2)
# Mouth
cv2.ellipse(synth, (160, 210), (25, 10), 0, 0, 180, (100, 110, 200), -1)
# Add realistic skin noise and slight blur
noise = np.random.normal(0, 4, synth.shape).astype(np.int16)
synth = np.clip(synth.astype(np.int16) + noise, 0, 255).astype(np.uint8)
synth = cv2.GaussianBlur(synth, (3, 3), 0)
cv2.imwrite(os.path.join(TEST_DIR, "06_unindexed_nomatch.jpg"), synth)

print("Generated test cases:")
for f in sorted(os.listdir(TEST_DIR)):
    print(f" - {f}")
