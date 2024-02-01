from typing import List, Dict
from pathlib import Path
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv
import matplotlib.pyplot as plt


PATH_MODEL_CHECKPOINT = str(Path.joinpath(Path.cwd().parent, "checkpoints", "sam_vit_h_4b8939.pth"))
MODEL_TYPE = "vit_h"

MASK_GENERATOR = SamAutomaticMaskGenerator(
    model=sam_model_registry[MODEL_TYPE](checkpoint=PATH_MODEL_CHECKPOINT),
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)


def segment_with_prompts(image: np.ndarray, prompts: np.ndarray, model_path=PATH_MODEL_CHECKPOINT, model_type=MODEL_TYPE) -> List[Dict]:
    sam = sam_model_registry[model_type](checkpoint=model_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(prompts)
    return masks


def segment(image: np.ndarray, model_path=PATH_MODEL_CHECKPOINT, model_type=MODEL_TYPE) -> List[Dict]:
    sam = sam_model_registry[model_type](checkpoint=model_path)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks


def plot_masks(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def plot_segmentation(image: np.ndarray, masks: List[Dict]):
    plt.imshow(image)
    plot_masks(masks)
    plt.show()


if __name__ == "__main__":
    img_path = str(Path.joinpath(Path.cwd().parent, "data", "30-59_45_ORIGINAL.jpg"))
    image = cv.imread(img_path)[100:]
    masks = segment(image)
