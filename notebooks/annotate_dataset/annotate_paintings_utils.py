import io
import copy
import json
import base64

import polars as pl
from PIL import Image
from nltk.corpus import stopwords

RAW_DATA_PATH = "../../data/raw/"
RESULTS_PATH = "../../experiments/prompting/"
ANNOTATIONS_PATH = "../../data/annotations/"
INTERMEDIATE_DATA_PATH = "../../data/intermediate/filtered_paintings/"

STOP_WORDS = stopwords.words("english")
FEW_SHOT_EXAMPLES_IDS = [2156, 2484, 11819, 256, 10748, 3344, 10676]


def load_image(painting_id):
    return Image.open(f"{RAW_DATA_PATH}filtered_paintings/{painting_id}.png")


def image_to_bytes(image):
    # define an in-memory byte stream
    img_byte_array = io.BytesIO()

    # convert the image to a byte representation and store it in the in-memory byte stream
    image.save(img_byte_array, format="PNG")

    # get the byte representation of the image
    img_bytes = img_byte_array.getvalue()

    return img_bytes


def image_to_url(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{image_base64}"

    return image_url


def load_data():
    paintings_data = pl.read_json(f"{INTERMEDIATE_DATA_PATH}filtered_paintings_enhanced_data.json")
    annotations = pl.read_json(ANNOTATIONS_PATH + "manual_annotations.json").with_columns(
        pl.col("object_name").str.replace_all(",", "", literal=True).alias("object_name")
    )

    few_shots_descriptions = (
        paintings_data.filter(pl.col("id").is_in(FEW_SHOT_EXAMPLES_IDS))
        .select("id", "description")
        .rename({"id": "painting_id"})
    )
    few_shot_examples = (
        (
            annotations.filter(pl.col("painting_id").is_in(FEW_SHOT_EXAMPLES_IDS))
            .group_by("painting_id")
            .agg(pl.col("*"))
            .select("painting_id", "object_name", "description_spans", "object_description")
        ).join(few_shots_descriptions, on="painting_id")
    ).to_dicts()

    test_descriptions = (
        paintings_data.filter(~pl.col("id").is_in(FEW_SHOT_EXAMPLES_IDS))
        .select("id", "description")
        .rename({"id": "painting_id"})
    )
    test_paintings = (
        (
            annotations.filter(~pl.col("painting_id").is_in(FEW_SHOT_EXAMPLES_IDS))
            .group_by("painting_id")
            .agg(pl.col("*"))
            .select("painting_id", "object_name", "description_spans")
        ).join(test_descriptions, on="painting_id")
    ).to_dicts()

    return paintings_data, annotations, few_shot_examples, test_paintings


def get_bbox_annotations():
    with open(f"{ANNOTATIONS_PATH}bounding_boxes_annotations.json", "r") as file:
        bounding_boxes_annotations = json.load(file)

    ids_to_labels = {
        label["id"]: label["name"] for label in bounding_boxes_annotations["categories"]
    }
    labels_to_ids = {
        clean_object_name(label["name"]): label["id"]
        for label in bounding_boxes_annotations["categories"]
    }

    painting_id_mapping = {
        painting_info["id"]: painting_info["file_name"].split("-")[-1].split(".")[0]
        for painting_info in bounding_boxes_annotations["images"]
    }
    processed_bounding_box_annotations = []

    for bounding_box_annotation in bounding_boxes_annotations["annotations"]:
        x1, y1, x2, y2 = bounding_box_annotation["bbox"]

        x2 += x1
        y2 += y1

        bbox = [x1, y1, x2, y2]

        processed_bounding_box_annotations.append(
            {
                "image_id": int(painting_id_mapping[bounding_box_annotation["image_id"]]),
                "label": ids_to_labels[bounding_box_annotation["category_id"]],
                "bbox": bbox,
            }
        )

    return processed_bounding_box_annotations, labels_to_ids


def clean_object_name(object_name):
    input_words = object_name.lower().split(" ")
    cleaned_object_name = " ".join(
        [word.replace("\n", "") for word in input_words if word.replace("\n", "") not in STOP_WORDS]
    )

    return cleaned_object_name


def sort_and_clean_output(llm_output, painting):
    sorted(llm_output, key=lambda x: x.object_name)

    llm_output_copy = copy.deepcopy(llm_output)

    for index in range(len(llm_output)):
        if llm_output_copy[index].object_name not in painting["description"]:
            del llm_output_copy[index]
            continue

        spans = llm_output_copy[index].description_spans
        kept_spans = []

        for span in spans:
            if span in painting["description"]:
                kept_spans.append(span)
            else:
                print("not there")

        llm_output_copy[index].description_spans = kept_spans

    return llm_output_copy


def process_objects(llm_output, painting, all_predicted_objects, all_ground_truth_objects, verbose):
    predicted_objects = sorted(
        [
            clean_object_name(object_name)
            for object_name in [annotation.object_name for annotation in llm_output]
        ]
    )
    all_predicted_objects.append(predicted_objects)

    ground_truth_objects = sorted(
        [clean_object_name(object_name) for object_name in copy.deepcopy(painting["object_name"])]
    )
    all_ground_truth_objects.append(ground_truth_objects)

    if verbose:
        print(predicted_objects, ground_truth_objects)

    return predicted_objects, ground_truth_objects


def process_spans(llm_output, painting):
    predicted_spans_per_object = {
        clean_object_name(annotation.object_name): annotation.description_spans
        for annotation in llm_output
    }
    ground_truth_spans_per_object = dict(
        zip(
            [clean_object_name(object_name) for object_name in painting["object_name"]],
            painting["description_spans"],
        )
    )

    print("ground truth spans per object", ground_truth_spans_per_object)

    predicted_spans = []
    for annotation in llm_output:
        predicted_spans.extend(annotation.description_spans)

    ground_truth_spans = []
    for spans in painting["description_spans"]:
        ground_truth_spans.extend(spans)

    return (
        predicted_spans_per_object,
        ground_truth_spans_per_object,
        predicted_spans,
        ground_truth_spans,
    )


def get_object_descriptions(llm_output, all_predicted_object_descriptions):
    predicted_object_descriptions = []

    for object_data in llm_output:
        predicted_object_descriptions.append(object_data.__dict__["object_description"])

    all_predicted_object_descriptions.append(predicted_object_descriptions)


def store_results(prompt_type, observations, results_values, metrics):
    results_file_name = f"{RESULTS_PATH}prompting_results.json"

    try:
        with open(results_file_name, "r") as file:
            all_results = json.load(file)
    except:
        all_results = None

    results = {
        "prompt_type": prompt_type,
        "observations": observations,
        "total_token_count": metrics["total_token_count"],
        "total_token_count_judge": metrics["total_token_count_judge"],
        "unprocessed_painting_ids": metrics["unprocessed_painting_ids"],
        "micro_f1_objects": metrics["micro_f1_objects"],
        "micro_f1_spans": metrics["micro_f1_spans"],
        "span_similarity_metrics": metrics["span_similarity_metrics"],
        "object_description_metrics": metrics["object_description_metrics"],
        "map_50": metrics["map_50"],
        "map_50_95": metrics["map_50_95"],
        "results": results_values,
    }

    if all_results is None:
        all_results = [results]
    else:
        all_results.append(results)

    with open(results_file_name, "w") as file:
        json.dump(all_results, file, indent=4)
