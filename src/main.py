from pathlib import Path
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv


PATH_MODEL_CHECKPOINT = str(Path.joinpath(Path.cwd().parent, "checkpoints", "sam_vit_h_4b8939.pth"))
MODEL_TYPE = "vit_h"


def segment_with_prompts(image: np.ndarray, prompts: np.ndarray, model_path=PATH_MODEL_CHECKPOINT, model_type=MODEL_TYPE):
    sam = sam_model_registry[model_type](checkpoint=model_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(prompts)


def segment(image: np.ndarray, model_path=PATH_MODEL_CHECKPOINT, model_type=MODEL_TYPE):
    sam = sam_model_registry[model_type](checkpoint=model_path)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)


if __name__ == "__main__":
    img_path = str(Path.joinpath(Path.cwd().parent, "data", "30-59_45_ORIGINAL.jpg"))
    image = cv.imread(img_path)
    segment(image)
