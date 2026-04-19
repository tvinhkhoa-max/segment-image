import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# load model
# sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
# sam.to("cpu")

# predictor = SamPredictor(sam)

def extract_nail(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    h, w, _ = image.shape

    # 🎯 chọn điểm ở đầu móng (center phía trên)
    input_point = np.array([[w // 2, int(h * 0.2)]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # lấy mask tốt nhất
    mask = masks[np.argmax(scores)]

    # apply mask
    result = image.copy()
    result[~mask] = 0

    return result

# result = extract_nail("tdsq2ir0abkg28pnzdi8j3nh_cropped.png")
# cv2.imwrite("output.png", result)