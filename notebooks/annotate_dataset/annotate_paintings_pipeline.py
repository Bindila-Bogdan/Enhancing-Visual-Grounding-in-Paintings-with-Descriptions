import sys
from tqdm import tqdm

from config import *
from call_llm import *
from ground_objects import *
from compute_metrics import *
from judge_annotations import *
from annotate_paintings_utils import *


def extract_objects(
    llm_client,
    backup_llm_client,
    judge_client,
    few_shot_examples,
    image,
    painting,
    total_token_count,
    total_token_count_judge,
):
    feedback = None
    object_extraction_trials_no = 1
    unprocessed_painting_id = None
    painting_id_wo_objects = None
    judge_metrics = None
    judgement = None

    while object_extraction_trials_no > 0:
        llm_output, token_count, _, _ = generate(
            llm_client,
            backup_llm_client,
            few_shot_examples,
            image,
            painting["description"],
            None,
            f"{PROMPT_TYPE}_object_extraction",
            feedback,
        )
        total_token_count += token_count

        if llm_output is None:
            object_and_spans = None
            unprocessed_painting_id = painting["painting_id"]
            break

        llm_output = sort_and_clean_output(llm_output, painting)

        if VERBOSE:
            pprint(llm_output)

        if len(llm_output) == 0:
            painting_id_wo_objects = painting["painting_id"]
            object_and_spans = None
            break

        object_and_spans = {"object_names": [], "descriptions_spans": []}

        for llm_output_ in llm_output:
            object_and_spans["object_names"].append(llm_output_.object_name)
            object_and_spans["descriptions_spans"].append(llm_output_.description_spans)

        if not JUDGE_OUTPUT:
            object_extraction_trials_no = 0
            break

        _, judge_metrics, token_count_judge, judgement = judge_objects_extractions(
            judge_client, image, painting["description"], object_and_spans, None
        )
        total_token_count_judge += token_count_judge
        object_extraction_trials_no -= 1

    return (
        llm_output,
        object_and_spans,
        painting_id_wo_objects,
        unprocessed_painting_id,
        total_token_count,
        total_token_count_judge,
        judge_metrics,
        judgement,
    )


def compose_object_description(
    llm_client,
    backup_llm_client,
    judge_client,
    few_shot_examples,
    object_and_spans,
    painting_id,
    total_token_count,
    total_token_count_judge,
):
    feedback = None
    object_description_trials_no = 1
    unprocessed_painting_id = None
    painting_id_to_check = None
    judgement = None

    while object_description_trials_no > 0:
        llm_output, token_count, _, _ = generate(
            llm_client,
            backup_llm_client,
            few_shot_examples,
            None,
            None,
            object_and_spans,
            f"{PROMPT_TYPE}_create_description",
            feedback,
        )
        total_token_count += token_count

        if VERBOSE:
            pprint(llm_output)

        if llm_output is None:
            unprocessed_painting_id = painting_id
            break

        if len(object_and_spans["object_names"]) != len(llm_output):
            if VERBOSE:
                print("The annotator didn't provide the number of expected descriptions.")
            unprocessed_painting_id = painting_id
            break

        if not JUDGE_OUTPUT:
            object_description_trials_no = 0
            break

        object_and_spans["objects_description"] = [
            annotation.object_description for annotation in llm_output
        ]
        _, token_count_judge, inconsistent_scoring, judgement = judge_objects_descriptions(
            judge_client, object_and_spans, None
        )
        total_token_count_judge += token_count_judge
        object_description_trials_no -= 1

        if inconsistent_scoring:
            painting_id_to_check = painting_id

    return (
        llm_output,
        unprocessed_painting_id,
        painting_id_to_check,
        total_token_count,
        total_token_count_judge,
        judgement,
    )


def annotate_paintings_batch(start_index, stop_index, store_freq=25):
    # get data
    paintings_data, _, few_shot_examples, _, _ = load_data()

    # get device type
    device = get_device()

    # load models
    llm_client = get_llm_client()
    backup_llm_client = get_llm_client(location="us-central1")
    judge_client = get_judge_llm_client()
    grounding_processor, grounding_model = get_grounding_model(device)

    # other tracked info
    total_token_count = 0
    total_token_count_judge = 0
    painting_ids = []
    paintings_ids_to_check = []
    paintings_ids_unprocessed = []
    paintings_ids_wo_objects = []

    paintings_annotations = []
    last_in_batch_index = 0

    for in_batch_index, painting in enumerate(tqdm(paintings_data[start_index:stop_index])):
        resized_image, image = load_image(painting["painting_id"])
        if VERBOSE:
            print(f"PAINTING DESCRIPTION\n{painting['description']}")

        # extract described objects
        (
            llm_output_objects,
            object_and_spans,
            painting_id_wo_objects,
            unprocessed_painting_id,
            total_token_count,
            total_token_count_judge,
            extraction_judge_metrics,
            extraction_judgement,
        ) = extract_objects(
            llm_client,
            backup_llm_client,
            judge_client,
            few_shot_examples,
            resized_image,
            painting,
            total_token_count,
            total_token_count_judge,
        )

        if unprocessed_painting_id is not None:
            paintings_ids_unprocessed.append(unprocessed_painting_id)
            continue

        if painting_id_wo_objects is not None:
            paintings_ids_wo_objects.append(painting_id_wo_objects)
            continue

        # create description per object
        (
            llm_output_descriptions,
            unprocessed_painting_id,
            painting_id_to_check,
            total_token_count,
            total_token_count_judge,
            description_judgement,
        ) = compose_object_description(
            llm_client,
            backup_llm_client,
            judge_client,
            few_shot_examples,
            copy.deepcopy(object_and_spans),
            painting["painting_id"],
            total_token_count,
            total_token_count_judge,
        )

        if unprocessed_painting_id is not None:
            paintings_ids_unprocessed.append(unprocessed_painting_id)
            continue

        if painting_id_to_check is not None:
            paintings_ids_to_check.append(painting_id_to_check)

        # handle objects
        predicted_objects, _ = process_objects(llm_output_objects, painting, None, None)

        # handle spans
        predicted_spans_per_object, _, _, _ = process_spans(llm_output_objects, painting)

        # handle descriptions
        objects_descriptions = [
            object_name.object_description for object_name in llm_output_descriptions
        ]

        # ground objects
        labels_scores_boxes, _ = detect_objects(
            image,
            predicted_objects,
            grounding_processor,
            grounding_model,
            device,
            object_threshold=0.34,
            text_threshold=0.32,
        )

        # if the else branch has been reached, the full image annotation process is complete
        if len(predicted_spans_per_object.keys()) != len(objects_descriptions):
            paintings_ids_unprocessed.append(painting["painting_id"])
            continue
        else:
            painting_ids.append(painting["painting_id"])

        for index, object_name in enumerate(predicted_spans_per_object.keys()):
            predicted_spans_per_object[object_name] = [
                predicted_spans_per_object[object_name],
                objects_descriptions[index],
            ]

        paintings_annotations.append(
            {
                "painting_id": painting["painting_id"],
                "objects": predicted_spans_per_object,
                "bounding_boxes": [
                    [bbox[0], float(bbox[1]), [float(bbox_coord) for bbox_coord in bbox[2]]]
                    for bbox in labels_scores_boxes
                ],
                "extraction_judgement": {**extraction_judgement, **extraction_judge_metrics},
                "description_judgement": description_judgement,
            }
        )

        if in_batch_index % store_freq == 0 and in_batch_index > 0:
            print("\nStoring results...")
            store_annotations(
                total_token_count,
                total_token_count_judge,
                paintings_ids_unprocessed,
                paintings_ids_to_check,
                paintings_ids_wo_objects,
                paintings_annotations,
                start_index,
                in_batch_index + start_index,
                last_in_batch_index + start_index,
            )
            last_in_batch_index = in_batch_index

    print("\nStoring results...")
    store_annotations(
        total_token_count,
        total_token_count_judge,
        paintings_ids_unprocessed,
        paintings_ids_to_check,
        paintings_ids_wo_objects,
        paintings_annotations,
        start_index,
        in_batch_index + start_index,
        last_in_batch_index + start_index,
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python annotate_paintings_pipeline.py <integer1> <integer2>")
        sys.exit(1)

    try:
        start_index = int(sys.argv[1])
        stop_index = int(sys.argv[2])
    except ValueError:
        print("Error: Both arguments must be valid integers.")
        sys.exit(1)

    annotate_paintings_batch(start_index=start_index, stop_index=stop_index)


if __name__ == "__main__":
    main()
