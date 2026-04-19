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

        # model=sam_model,
        # points_per_side=8,          # 🔥 giảm mạnh (32 → 8)
        # pred_iou_thresh=0.9,
        # stability_score_thresh=0.95,
        # min_mask_region_area=2000   # bỏ mask nhỏ
    )

    _sam = sam_model

    print("✅ SAM loaded successfully")


# ===== SCORE MASK (CHỌN MÓNG) =====
def score_mask(mask, h, w):
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

    # 🎯 nằm phía trên (móng)
    if center_y < 0.5:
        score += 2

    # 🎯 shape giống móng
    if 0.5 < aspect < 2.0:
        score += 2

    # 🎯 size hợp lý
    if 1000 < area < 20000:
        score += 2

    return score


# ===== MAIN FUNCTION =====
def extract_nail_auto(image_np):
    """
    input: numpy image (BGR từ OpenCV)
    output: RGBA image (numpy)
    """

    # convert sang RGB cho SAM
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    # ===== STEP 1: generate masks =====
    masks = mask_generator.generate(image_rgb)

    best_score = -1
    best_mask = None

    # ===== STEP 2: chọn mask tốt nhất =====
    for m in masks:
        mask = m["segmentation"]
        s = score_mask(mask, h, w)

        if s > best_score:
            best_score = s
            best_mask = mask

    if best_mask is None:
        raise Exception("No nail detected")

    # ===== STEP 3: tạo alpha =====
    alpha = (best_mask * 255).astype(np.uint8)

    # 🎯 làm mịn viền (anti-alias nhẹ)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    # ===== STEP 4: tạo RGBA =====
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # giữ nguyên màu gốc
    rgba[:, :, 0:3] = image_rgb

    # gán alpha
    rgba[:, :, 3] = alpha

    # ===== STEP 5: crop =====
    ys, xs = np.where(best_mask)

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()

    cropped = rgba[miny:maxy, minx:maxx]

    # ===== STEP 6: convert về BGRA để encode đúng màu =====
    output = cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA)

    return output