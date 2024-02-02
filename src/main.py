import os
from typing import List, Dict
from pathlib import Path
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv
import matplotlib.pyplot as plt

ROOT_DIR = Path.cwd().parent
PATH_MODEL_CHECKPOINT = str(Path.joinpath(ROOT_DIR, "checkpoints", "sam_vit_h_4b8939.pth"))
MODEL_TYPE = "vit_h"
BBOX_PIPE_DEFAULT = [250, 200, 268, 276]

MASK_GENERATOR = SamAutomaticMaskGenerator(
    model=sam_model_registry[MODEL_TYPE](checkpoint=PATH_MODEL_CHECKPOINT),
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)


def segment_with_prompts(image: np.ndarray, prompts: np.ndarray, model_path=PATH_MODEL_CHECKPOINT,
                         model_type=MODEL_TYPE) -> np.ndarray:
    sam = sam_model_registry[model_type](checkpoint=model_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(prompts)
    return masks


def segment(image: np.ndarray, model_path=PATH_MODEL_CHECKPOINT, model_type=MODEL_TYPE) -> List[Dict]:
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks


def segment_with_custom_mask_generator(image: np.ndarray, model_path=PATH_MODEL_CHECKPOINT, model_type=MODEL_TYPE):
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.95,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500,
    )
    masks = mask_generator.generate(image)
    return masks


def remove_masks_with_area_smaller_than(masks, area_threshold=50000):
    new_masks = []
    for mask in masks:
        if mask["area"] > area_threshold:
            new_masks.append(mask)
    return new_masks


def plot_masks(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)


def plot_segmentation(image: np.ndarray, masks: List[Dict], write=False, filename=""):
    plt.figure()
    plt.imshow(image)
    plot_masks(masks)
    if not write:
        plt.show()
    else:
        plt.savefig(Path.joinpath(ROOT_DIR, "results", filename))
        plt.close()


def compute_iou_bbox(bbox: List, bbox_reference=BBOX_PIPE_DEFAULT) -> float:
    # bboxes in XYWH format
    area_ref_bbox = bbox_reference[2] * bbox_reference[3]
    area_bbox = bbox[2] * bbox[3]

    bbox_intersect = [
        max(bbox_reference[0], bbox[0]),
        max(bbox_reference[1], bbox[1]),
        min(bbox_reference[0] + bbox_reference[2], bbox[0] + bbox[2]),
        min(bbox_reference[1] + bbox_reference[3], bbox[1] + bbox[3]),
    ]  # not WH but abs pixel values

    area_of_overlap = max(0, bbox_intersect[2] - bbox_intersect[0]) * max(0, bbox_intersect[3] - bbox_intersect[1])
    area_of_union = area_ref_bbox + area_bbox - area_of_overlap
    return area_of_overlap / area_of_union


if __name__ == "__main__":
    if not os.path.exists(Path.joinpath(ROOT_DIR, "results")):
        os.makedirs(Path.joinpath(ROOT_DIR, "results"))

    for root, dirs, files in os.walk((Path.joinpath(ROOT_DIR, "data"))):
        for filename in files:
            img_path = str(Path.joinpath(ROOT_DIR, "data", filename))
            image = cv.imread(img_path)[100:]
            masks = segment_with_custom_mask_generator(image)
            masks_filtered = remove_masks_with_area_smaller_than(masks)

            iou_scores = []
            for mask in masks_filtered:
                iou_scores.append(compute_iou(mask["bbox"]))

            plot_segmentation(image, [masks_filtered[np.argmax(iou_scores)]])
