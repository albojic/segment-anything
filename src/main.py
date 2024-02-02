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
PROMPT_DEFAULT = np.array([[384, 338]])
PROMPT_LABEL_DEFAULT = np.asarray([1])


def segment_with_prompts(image: np.ndarray, prompts: np.ndarray = PROMPT_DEFAULT,
                         prompts_labels: np.ndarray = PROMPT_LABEL_DEFAULT,
                         model_path=PATH_MODEL_CHECKPOINT, model_type=MODEL_TYPE) -> np.ndarray:
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, logit = predictor.predict(point_coords=prompts, point_labels=prompts_labels, multimask_output=True)
    return masks, scores


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
        points_per_batch=128,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.95,
    )
    masks = mask_generator.generate(image)
    return masks


def remove_small_masks(masks, area_threshold=50000):
    new_masks = []
    for mask in masks:
        if mask["area"] > area_threshold:
            new_masks.append(mask)
    return new_masks


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_mask(image, mask, write=False, filename=""):
    plt.figure()
    plt.imshow(image)
    show_mask(mask, plt.gca())
    if not write:
        plt.show()
    else:
        plt.savefig(Path.joinpath(ROOT_DIR, "results", filename))
        plt.close()


def show_anns(anns):
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
    show_anns(masks)
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


def compute_iou_masks(mask1: Dict, mask2: Dict) -> float:
    img_mask1 = np.where(mask1['segmentation'], 1, 0)
    img_mask2 = np.where(mask2['segmentation'], 1, 0)
    img_sum = img_mask1 + img_mask2
    area_of_overlap = np.where(img_sum == 2, 1, 0).sum()
    area_of_union = np.where(img_sum != 0, 1, 0).sum()
    return area_of_overlap / area_of_union


def compute_iou(mask, bbox=BBOX_PIPE_DEFAULT):
    img_mask = np.where(mask['segmentation'], 1, 0)
    img_bbox = np.zeros(img_mask.shape)
    img_bbox[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = 1
    img_sum = img_mask + img_bbox
    area_of_overlap = np.where(img_sum == 2, 1, 0).sum()
    area_of_union = np.where(img_sum != 0, 1, 0).sum()
    return area_of_overlap / area_of_union


def remove_partial_segmentation(masks: List[Dict], best_mask: Dict):
    iou_masks = np.asarray([compute_iou_masks(best_mask, masks[i]) for i in range(len(masks))])
    overlapping_masks = np.argwhere(iou_masks > 0.2)
    if len(overlapping_masks) > 1:
        overlapping_masks_areas = [masks[i[0]]["area"] for i in overlapping_masks]
        return masks[overlapping_masks[np.argmax([overlapping_masks_areas])][0]]
    return best_mask


if __name__ == "__main__":
    if not os.path.exists(Path.joinpath(ROOT_DIR, "results")):
        os.makedirs(Path.joinpath(ROOT_DIR, "results"))

    for root, dirs, files in os.walk((Path.joinpath(ROOT_DIR, "data"))):
        for filename in files:
            img_path = str(Path.joinpath(ROOT_DIR, "data", filename))
            image = cv.imread(img_path)[100:]

            masks_prompted, scores = segment_with_prompts(image)
            best_mask = masks_prompted[np.argmax(scores)]
            plot_mask(image, best_mask, True, filename[:-4] + "_pipe_masked.jpg")

            # masks = segment_with_custom_mask_generator(image)
            # masks_filtered = remove_small_masks(masks)
            #
            # iou_scores = [compute_iou(mask) for mask in masks_filtered]
            # best_mask = remove_partial_segmentation(masks=masks_filtered,
            #                                         best_mask=masks_filtered[np.argmax(iou_scores)])

            # plot_segmentation(image, [best_mask], True, filename[:-4] + "_pipe_masked.jpg")
