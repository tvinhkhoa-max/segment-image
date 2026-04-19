import numpy as np
import cv2
import torch
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# load model
sam = None
mask_generator = None

# sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
# sam.to("cpu")
def load_model(model_path):
    global sam, mask_generator
    from segment_anything import SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    # sam = sam_model_registry["vit_b"](checkpoint=model_path)
    # sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=500
    )

    print("Model loaded on:", device)


# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.88,
#     stability_score_thresh=0.92,
#     min_mask_region_area=500
# )

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


def extract_nail_auto_bk(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)

    best_score = -1
    best_mask = None

    for m in masks:
        mask = m["segmentation"]
        score = score_mask(mask, image.shape[:2])

        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise Exception("No nail detected")

    # apply mask
    # result = image.copy()
    # result[~best_mask] = 0
    # tạo ảnh RGBA
    h, w, _ = image.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # copy màu gốc
    rgba[:, :, :3] = image

    # alpha channel
    rgba[:, :, 3] = best_mask.astype(np.uint8) * 255

    # crop
    ys, xs = np.where(best_mask)
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()

    cropped = result[miny:maxy, minx:maxx]

    return cropped

# ===== MAIN FUNCTION =====
def extract_nail_auto(image_np):
    """
    input: numpy image (BGR từ OpenCV)
    output: RGBA image (numpy)
    """
    global mask_generator

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