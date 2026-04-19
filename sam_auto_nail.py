import numpy as np
import cv2
import torch
import segment_anything
from segment_anything import sam_model_registry

# ===== GLOBAL SINGLETON =====
_sam = None
_mask_generator = None
_device = None


# ===== LOAD MODEL (SAFE) =====
def load_model(model_path: str):
    global _sam, _mask_generator, _device

    if _sam is not None:
        print("⚡ Model already loaded")
        return

    # detect device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        _device = "mps"

    print("🚀 Loading SAM on:", _device)

    # load model
    sam_model = sam_model_registry["vit_b"](checkpoint=model_path)
    sam_model.to(_device)

    if sam_model is None:
        raise Exception("❌ SAM load failed")

    # create mask generator
    _mask_generator = segment_anything.SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=500
    )

    _sam = sam_model

    print("✅ SAM loaded successfully")


# ===== SCORE MASK (CHỌN MÓNG) =====
def _score_mask(mask, h, w):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return -1

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()

    box_w = maxx - minx
    box_h = maxy - miny
    area = len(xs)

    aspect = box_w / (box_h + 1e-5)
    center_y = (miny + maxy) / 2 / h

    score = 0

    # 🎯 móng thường nằm phía trên
    if center_y < 0.6:
        score += 2

    # 🎯 shape móng (không quá dài hoặc quá vuông)
    if 0.5 < aspect < 2.5:
        score += 2

    # 🎯 diện tích hợp lý
    if 800 < area < 50000:
        score += 2

    return score


# ===== MAIN FUNCTION =====
def extract_nail_auto(image_np: np.ndarray):
    global _mask_generator

    if _mask_generator is None:
        raise Exception("❌ Model chưa load. Gọi load_model() trước.")

    if image_np is None:
        raise Exception("❌ Image input is None")

    # convert BGR -> RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    # ===== STEP 1: generate masks =====
    masks = _mask_generator.generate(image_rgb)

    if len(masks) == 0:
        raise Exception("❌ No mask generated")

    # ===== STEP 2: chọn mask tốt nhất =====
    best_score = -1
    best_mask = None

    for m in masks:
        mask = m["segmentation"]
        score = _score_mask(mask, h, w)

        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise Exception("❌ No nail detected")

    # ===== STEP 3: tạo alpha =====
    alpha = (best_mask * 255).astype(np.uint8)

    # làm mịn viền
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    # ===== STEP 4: tạo RGBA =====
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = alpha

    # ===== STEP 5: crop =====
    ys, xs = np.where(best_mask)

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()

    cropped = rgba[miny:maxy, minx:maxx]

    # ===== STEP 6: convert về BGRA (OpenCV chuẩn) =====
    output = cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA)

    return output