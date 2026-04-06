import os
import random

import torch
import numpy as np
from IPython.display import display
from PIL import ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from config import *
from annotate_paintings_utils import *


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_grounding_model(device, seed=42):
    # make Grounding DINO deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL_ID).to(device)

    return processor, model


def display_annotated_image(image, labels_scores_boxes, show_small_image=False):
    font = ImageFont.truetype("../../config/alata-regular.ttf", 18)
    draw = ImageDraw.Draw(image, "RGBA")

    for label, score, coords in labels_scores_boxes:
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF)) + "80"
        text_position = (coords[0] + 10, coords[1] + 5)

        draw.rectangle(coords, outline=random_color, width=5)
        draw.text(text_position, label + " " + str(round(score, 2)), fill=random_color, font=font)

    if show_small_image:
        display(resize_image(image))
    else:
        display(image)


def detect_objects_parallel(
    image, objects, processor, model, device, object_threshold=0.3, text_threshold=0.3
):

    text = " . ".join(objects) + " ."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        # threshold for filtering object detection predictions (lower -> more bounding boxes)
        threshold=object_threshold,
        # threshold for filtering text detection predictions (lower -> the input text is taken exactly)
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    assert len(results) == 1

    labels = results[0]["text_labels"]
    scores = results[0]["scores"].cpu().numpy()
    box_coordinates = [list(coords) for coords in results[0]["boxes"].cpu().numpy()]
    labels_scores_boxes = sorted(list(zip(labels, scores, box_coordinates)), key=lambda x: x[1])

    if VERBOSE:
        print("\nGROUNDED OBJECTS")
        for label, score, coords in labels_scores_boxes:
            print(label, float(score), [float(coord) for coord in coords])

        display_annotated_image(image, labels_scores_boxes)

    return labels_scores_boxes


def detect_objects(
    image, objects, processor, model, device, object_threshold=0.3, text_threshold=0.3
):

    painting_labels_scores_boxes = []

    for object_name in objects:
        inputs = processor(images=image, text=object_name + " .", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # threshold for filtering object detection predictions (lower -> more bounding boxes)
            threshold=object_threshold,
            # threshold for filtering text detection predictions (lower -> the input text is taken exactly)
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        assert len(results) == 1

        labels = results[0]["text_labels"]
        scores = results[0]["scores"].cpu().numpy()
        box_coordinates = [list(coords) for coords in results[0]["boxes"].cpu().numpy()]
        painting_labels_scores_boxes.extend(
            sorted(list(zip(labels, scores, box_coordinates)), key=lambda x: x[1])
        )

    painting_labels_scores_boxes = sorted(painting_labels_scores_boxes, key=lambda x: x[1])

    if VERBOSE:
        print("\nGROUNDED OBJECTS")
        for label, score, coords in painting_labels_scores_boxes:
            print(label, float(score), [float(coord) for coord in coords])

        display_annotated_image(image, painting_labels_scores_boxes)

    return painting_labels_scores_boxes
