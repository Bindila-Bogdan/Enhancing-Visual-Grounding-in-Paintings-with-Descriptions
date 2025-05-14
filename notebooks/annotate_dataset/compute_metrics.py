import sys
import copy
from pprint import pprint
from difflib import SequenceMatcher

import torch
import Levenshtein
from sentence_transformers import SentenceTransformer
from torchmetrics.detection import MeanAveragePrecision

from config import *
from annotate_paintings_utils import clean_object_name


# compute the detection quality of described objects
def count_tp_fp_fn(predictions, ground_truth, tp_fp_fn):
    predictions_copy = copy.deepcopy(predictions)
    ground_truth_copy = copy.deepcopy(ground_truth)

    for prediction in predictions_copy:
        if prediction in ground_truth_copy:
            tp_fp_fn[0] += 1
            ground_truth_copy.remove(prediction)
        else:
            tp_fp_fn[1] += 1

    tp_fp_fn[2] = len(ground_truth_copy)


def compute_micro_f1(tp_fp_fn, name):
    if tp_fp_fn[0] + tp_fp_fn[1] == 0:
        precision = 0
    else:
        precision = tp_fp_fn[0] / (tp_fp_fn[0] + tp_fp_fn[1])

    if tp_fp_fn[0] + tp_fp_fn[2] == 0:
        recall = 0
    else:
        recall = tp_fp_fn[0] / (tp_fp_fn[0] + tp_fp_fn[2])

    if precision + recall == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2 * (precision * recall) / (precision + recall)

    micro_f1 = round(micro_f1, 2)

    if VERBOSE:
        print(f"Micro F1 for extracting {name}: {micro_f1}")

    return micro_f1


# compute the description span extraction quality
def get_sentence_similarity_model():
    return SentenceTransformer(SENTENCE_SIMILARITY_MODEL)


def assess_span_extraction_quality(ground_truth_span, extracted_span):
    # these metrics are computed at the word level
    ground_truth_words = "".join(
        char for char in ground_truth_span.lower() if char.isalnum() or char.isspace()
    ).split()
    extracted_words = "".join(
        char for char in extracted_span.lower() if char.isalnum() or char.isspace()
    ).split()

    matcher = SequenceMatcher(None, ground_truth_words, extracted_words)
    opcodes = matcher.get_opcodes()

    matches = 0
    deletions = 0
    insertions = 0

    for tag, a0, a1, b0, b1 in opcodes:
        if tag == "delete":
            deletions += a1 - a0
        elif tag == "insert":
            insertions += b1 - b0
        elif tag == "replace":
            deletions += a1 - a0
            insertions += b1 - b0
        elif tag == "equal":
            matches += a1 - a0

    ground_truth_words_no = len(ground_truth_words)
    extracted_words_no = len(extracted_words)

    deletion_pecentage = (
        (deletions / ground_truth_words_no) * 100 if ground_truth_words_no > 0 else 0
    )
    false_positive_percentage = (
        (insertions / extracted_words_no) * 100 if extracted_words_no > 0 else 0
    )
    coverage_percentage = (
        (matches / ground_truth_words_no) * 100 if ground_truth_words_no > 0 else 0
    )

    return deletion_pecentage, false_positive_percentage, coverage_percentage


def compute_levenshtein_distance(ground_truth_span, extracted_span):
    return Levenshtein.distance(ground_truth_span, extracted_span)


def compare_spans(ground_truth_span, extracted_span, model):
    similarity = float(
        model.similarity(model.encode([ground_truth_span]), model.encode([extracted_span]))[0][0]
    )
    levenshtein_distance = compute_levenshtein_distance(ground_truth_span, extracted_span)
    deletion_pecentage, false_positive_percentage, coverage_percentage = (
        assess_span_extraction_quality(ground_truth_span, extracted_span)
    )

    span_extraction_metrics = {
        "cosine similarity": round(similarity, 4),
        "Levenshtein distance": levenshtein_distance,
        "delete percentage": round(deletion_pecentage, 4),
        "false positive percentage": round(false_positive_percentage, 4),
        "coverage percentage": round(coverage_percentage, 4),
    }

    if VERBOSE:
        pprint(span_extraction_metrics)

    return span_extraction_metrics


def compute_spans_quality(
    ground_truth_spans, predicted_spans, span_similarity_metrics, sentence_similarity_model
):
    for object_name in predicted_spans.keys():
        if object_name in ground_truth_spans.keys():
            ground_truth_object_spans = sorted(ground_truth_spans[object_name])
            extracted_object_spans = sorted(predicted_spans[object_name])

            for extracted_object_span in extracted_object_spans:

                min_levenshtein_dist = sys.maxsize
                most_similar_ground_truth_span = None

                for ground_truth_object_span in ground_truth_object_spans:

                    levenshtein_dist = compute_levenshtein_distance(
                        ground_truth_object_span, extracted_object_span
                    )

                    if min_levenshtein_dist > levenshtein_dist:
                        min_levenshtein_dist = levenshtein_dist
                        most_similar_ground_truth_span = ground_truth_object_span

                similarity_metrics = compare_spans(
                    most_similar_ground_truth_span,
                    extracted_object_span,
                    sentence_similarity_model,
                )

                for metric, value in similarity_metrics.items():
                    span_similarity_metrics[metric].append(value)


# compute the grounding quality of the linguistic expressions or nouns
def get_bounding_boxes(
    labels_scores_boxes,
    labels_to_ids,
    ground_truth_bboxes,
    painting_id,
    all_predicted_bboxes,
    all_ground_truth_bboxes,
    device,
):
    # treat the case when the predicted class is unknown
    predicted_bboxes = {
        "boxes": torch.tensor([detection[2] for detection in labels_scores_boxes], device=device),
        "scores": torch.tensor([detection[1] for detection in labels_scores_boxes], device=device),
        "labels": torch.tensor(
            [
                (
                    labels_to_ids[detection[0]]
                    if detection[0] in labels_to_ids.keys()
                    else max(labels_to_ids.values()) + 1
                )
                for detection in labels_scores_boxes
            ],
            device=device,
        ),
    }

    paintings_annotation = [
        annotation for annotation in ground_truth_bboxes if annotation["image_id"] == painting_id
    ]

    if len(paintings_annotation) != 0:
        target_bboxes = {
            "boxes": torch.tensor(
                [annotation["bbox"] for annotation in paintings_annotation], device=device
            ),
            "labels": torch.tensor(
                [
                    labels_to_ids[clean_object_name(annotation["label"])]
                    for annotation in paintings_annotation
                ],
                device=device,
            ),
        }
    else:
        # treat the case when for an image there's no ground truth
        target_bboxes = {
            "boxes": torch.empty((0, 4)).to(device),
            "labels": torch.empty((0,), dtype=torch.int64).to(device),
        }

    all_predicted_bboxes.append(predicted_bboxes)
    all_ground_truth_bboxes.append(target_bboxes)


def compute_mean_average_precision(predictions, targets, device):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True).to(device)

    metric.update(predictions, targets)
    metrics = metric.compute()

    map_50 = float(metrics["map_50"])
    map_50_95 = float(metrics["map"])

    if VERBOSE:
        print(f"mAP@50: {map_50}")
        print(f"mAP@50-95: {map_50_95}")

        # if performance per class is available, show it
        try:
            print(f"mAP per class: {metrics['map_per_class']}")
            print(f"classes: {metrics['classes']}")
        except:
            pass

    return map_50, map_50_95
