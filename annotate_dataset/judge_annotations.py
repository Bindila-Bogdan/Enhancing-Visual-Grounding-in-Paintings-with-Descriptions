import os
import json
from pprint import pprint

from openai import OpenAI
from pydantic import BaseModel

from annotate_paintings_utils import *


from config import *


def get_judge_llm_client():
    with open("../../config/keys.json", "r") as file:
        os.environ["OPENAI_API_KEY"] = json.load(file)["openai_api_key"]

    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def call_judge(client, system_prompt, user_message, response_format):
    response = client.beta.chat.completions.parse(
        model=OPEN_AI_MODEL,
        seed=0,
        temperature=0,
        messages=[{"role": "system", "content": system_prompt}, user_message],
        response_format=response_format,
    )

    output = response.choices[0].message.parsed

    if output:
        output = dict(output)
    else:
        raise Exception("Error while judging the object extraction")

    prompt_tokens_count = response.usage.prompt_tokens
    output_tokens_count = response.usage.completion_tokens
    total_token_count = prompt_tokens_count + output_tokens_count

    if VERBOSE:
        print(
            f"\nJUDGE OUTPUT:\nPrompt tokens count: {prompt_tokens_count} Output tokens count: {output_tokens_count}"
        )

    return output, total_token_count


# judge object extraction
def get_object_extraction_fp_fn(extraction_evaluation, description, error_type):
    objects_with_wrong_spans = list(
        set(
            [entry.object_name for entry in extraction_evaluation[f"false_{error_type}_spans"]]
        ).difference(set(extraction_evaluation[f"false_{error_type}_objects"]))
    )

    spans = ""
    objects_with_spans = ""

    for entry in extraction_evaluation[f"false_{error_type}_spans"]:
        if entry.object_name not in description:
            continue

        spans_with_issue = []

        for span in entry.spans_with_issue:
            if span in description and span != "":
                spans_with_issue.append(span)

        if entry.object_name in objects_with_wrong_spans and len(spans_with_issue) != 0:
            spans += f"- only spans ['{"', '".join(spans_with_issue)}'] for object '{entry.object_name}'\n"
        else:
            if len(spans_with_issue) == 0:
                objects_with_spans += f"- object '{entry.object_name}'\n"
            else:
                objects_with_spans += f"- object '{entry.object_name}' together with all its spans ['{"', '".join(spans_with_issue)}']\n"

    objects_null_spans = list(
        set(extraction_evaluation[f"false_{error_type}_objects"]).difference(
            set([entry.object_name for entry in extraction_evaluation[f"false_{error_type}_spans"]])
        )
    )

    for object_null_spans in objects_null_spans:
        if object_null_spans not in description:
            continue

        objects_with_spans += f"- object '{object_null_spans}' and its empty description span\n"

    return spans, objects_with_spans


def compute_metrics_judge_object_extraction(output, object_and_spans, description):
    fn_objects_no = len(
        set([obj for obj in output["false_negative_objects"] if obj in description])
    )
    tp_objects_no = len(
        set(object_and_spans["object_names"]).difference(set(output["false_positive_objects"]))
    )

    if tp_objects_no + fn_objects_no == 0:
        objects_recall = 1
    else:
        objects_recall = tp_objects_no / (tp_objects_no + fn_objects_no)

    fn_spans_no = sum(
        [
            len(
                [
                    span
                    for span in false_negative_spans.spans_with_issue
                    if span in description and span != ""
                ]
            )
            for false_negative_spans in output["false_negative_spans"]
        ]
    )

    fp_spans = []
    tp_spans = []

    for fp_entry in output["false_positive_spans"]:
        fp_spans.extend(
            [span for span in fp_entry.spans_with_issue if span in description and span != ""]
        )

    for spans in object_and_spans["descriptions_spans"]:
        tp_spans.extend([span for span in spans if span in description])

    tp_spans_no = len(set(tp_spans).difference(set(fp_spans)))

    if tp_spans_no + fn_spans_no == 0:
        spans_recall = 1
    else:
        spans_recall = tp_spans_no / (tp_spans_no + fn_spans_no)

    fp_objects_no = len(
        set(output["false_positive_objects"]).intersection(set(object_and_spans["object_names"]))
    )

    if tp_objects_no + fp_objects_no == 0:
        objects_precision = 1
    else:
        objects_precision = tp_objects_no / (tp_objects_no + fp_objects_no)

    fp_spans_no = len(fp_spans)

    if tp_spans_no + fp_spans_no == 0:
        spans_precision = 1
    else:
        spans_precision = tp_spans_no / (tp_spans_no + fp_spans_no)

    judge_object_extraction_metrics = {
        "objects_recall": objects_recall,
        "spans_recall": spans_recall,
        "objects_precision": objects_precision,
        "spans_precision": spans_precision,
    }

    if VERBOSE:
        pprint(judge_object_extraction_metrics)

    return judge_object_extraction_metrics


def get_object_extraction_suggestions(description, output, metrics):
    major_issues_found = False
    judge_suggestions = JUDGE_OBJECT_EXTRACTION_ENHANCED_SUGGESTIONS[0]

    fp_spans, fp_objects_with_spans = get_object_extraction_fp_fn(output, description, "positive")

    if metrics["spans_precision"] < PRECISION_RECALL_THRESHOLD and len(fp_spans) > 0:
        if VERBOSE:
            print("fp_spans\n", fp_spans)
        major_issues_found = True
        judge_suggestions += f"False positives spans (spans that were extracted but should have not been extracted, although the extracted object is correct):\n{fp_spans}\n"

    if metrics["objects_precision"] < PRECISION_RECALL_THRESHOLD and len(fp_objects_with_spans) > 0:
        if VERBOSE:
            print("fp_objects_with_spans\n", fp_objects_with_spans)
        major_issues_found = True
        judge_suggestions += f"False positives objects (objects and all their spans that were extracted but should have not been extracted):\n{fp_objects_with_spans}\n"

    fn_spans, fn_objects_with_spans = get_object_extraction_fp_fn(output, description, "negative")

    if metrics["objects_recall"] < PRECISION_RECALL_THRESHOLD and len(fn_objects_with_spans) > 0:
        if VERBOSE:
            print("fn_objects_with_spans\n", fn_objects_with_spans)
        major_issues_found = True
        judge_suggestions += f"False negative objects (objects together with their spans that were not extracted but should be considered):\n{fn_objects_with_spans}\n"

    if metrics["spans_recall"] < PRECISION_RECALL_THRESHOLD and len(fn_spans) > 0:
        if VERBOSE:
            print("fn_spans\n", fn_spans)
        major_issues_found = True
        judge_suggestions += f"False negative spans (spans that were not extracted, but should have been extracted):\n{fn_spans}\n"

    judge_suggestions += JUDGE_OBJECT_EXTRACTION_ENHANCED_SUGGESTIONS[1]

    if major_issues_found:
        return judge_suggestions
    else:
        return ""


def judge_objects_extractions(
    client, image, description, object_and_spans, object_extraction_metrics
):
    class DescriptionObjectIssues(BaseModel):
        object_name: str
        explanation: str

    class IssueDescription(BaseModel):
        span: str
        explanation: str

    class DescriptionSpanIssues(BaseModel):
        object_name: str
        spans_with_issue: list[IssueDescription]

    class DescriptionExtractionEvaluation(BaseModel):
        false_positive_objects: list[DescriptionObjectIssues]
        false_negative_objects: list[DescriptionObjectIssues]
        false_positive_objects: list[DescriptionSpanIssues]
        false_negative_objects: list[DescriptionSpanIssues]

    class SpanIssues(BaseModel):
        object_name: str
        spans_with_issue: list[str]

    class ExtractionEvaluation(BaseModel):
        false_positive_objects: list[str]
        false_negative_objects: list[str]
        false_positive_spans: list[SpanIssues]
        false_negative_spans: list[SpanIssues]

    system_prompt = JUDGE_OBJECT_EXTRACTION_ENHANCED_SYSTEM_PROMPT

    object_and_spans_text = "- " + "\n- ".join(
        [
            f"{object_name}: {str(description_spans)}"
            for object_name, description_spans in zip(
                object_and_spans["object_names"], object_and_spans["descriptions_spans"]
            )
        ]
    )

    user_prompt = f"""Painting description:\n{description}\n\nExtracted objects together with their description spans:\n{object_and_spans_text}"""

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_to_url(image_to_bytes(image))},
            },
            {"type": "text", "text": user_prompt},
        ],
    }

    output, total_token_count = call_judge(
        client, system_prompt, user_message, ExtractionEvaluation
    )

    if VERBOSE:
        pprint(output)

    metrics = compute_metrics_judge_object_extraction(output, object_and_spans, description)
    judge_suggestions = get_object_extraction_suggestions(description, output, metrics)

    if BENCHMARKING:
        object_extraction_metrics["objects_recall"].append(metrics["objects_recall"])
        object_extraction_metrics["spans_recall"].append(metrics["spans_recall"])
        object_extraction_metrics["objects_precision"].append(metrics["objects_precision"])
        object_extraction_metrics["spans_precision"].append(metrics["spans_precision"])
    else:
        output["false_positive_spans"] = [
            [suggestion.object_name, suggestion.spans_with_issue]
            for suggestion in output["false_positive_spans"]
        ]
        output["false_negative_spans"] = [
            [suggestion.object_name, suggestion.spans_with_issue]
            for suggestion in output["false_negative_spans"]
        ]

    return judge_suggestions, metrics, total_token_count, output


# judge object description
def judge_object_description(client, object_and_spans):
    class ScoreExplanation(BaseModel):
        score: int
        explanation: str

    class DescriptionScoreEvaluation(BaseModel):
        factual_accuracy: list[ScoreExplanation]
        coherence: list[ScoreExplanation]
        grounding_potential: list[ScoreExplanation]
        completeness: list[ScoreExplanation]

    class DescriptionScore(BaseModel):
        factual_accuracy: list[int]
        coherence: list[int]
        grounding_potential: list[int]
        completeness: list[int]

    system_prompt = JUDGE_OBJECT_DESCRIPTION_ENHANCED_SYSTEM_PROMPT

    user_prompt = ""

    for index, _ in enumerate(object_and_spans["objects_description"]):
        user_prompt += f"""Object Name: {object_and_spans["object_names"][index]}\nOriginal Description Spans:\n{object_and_spans["descriptions_spans"][index]}\nGenerated Description: {object_and_spans["objects_description"][index]}"\n\n"""

    user_message = {"role": "user", "content": user_prompt}
    output, total_token_count = call_judge(client, system_prompt, user_message, DescriptionScore)

    return output, total_token_count


def get_object_description_suggestions(object_and_spans, desc_judgements):
    judge_suggestions = JUDGE_OBJECT_DESCRIPTION_ENHANCED_SUGGESTIONS[0]

    any_issues_found = False

    for index, _ in enumerate(desc_judgements["factual_accuracy"]):
        factual_accuracy = desc_judgements["factual_accuracy"][index]
        coherence = desc_judgements["coherence"][index]
        completeness = desc_judgements["completeness"][index]

        issues_found = False
        current_judge_suggestions = ""

        if factual_accuracy < DESCRIPTION_METRIC_THRESHOLD:
            issues_found = True
            current_judge_suggestions += f" {factual_accuracy}/5 for factual accuracy"

        if coherence < DESCRIPTION_METRIC_THRESHOLD:
            issues_found = True
            current_judge_suggestions += f" {coherence}/5 for coherence"

        if completeness < DESCRIPTION_METRIC_THRESHOLD:
            issues_found = True
            current_judge_suggestions += f" {completeness}/5 for completeness"

        if issues_found:
            any_issues_found = True
            object_description = object_and_spans["object_names"][index]
            object_name = object_and_spans["objects_description"][index]
            current_judge_suggestions_complete = (
                f"- description '{object_description}' of object '{object_name}' received the scores:"
                + current_judge_suggestions
                + "\n"
            )
            judge_suggestions += current_judge_suggestions_complete

    judge_suggestions += JUDGE_OBJECT_DESCRIPTION_ENHANCED_SUGGESTIONS[1]

    if any_issues_found:
        return judge_suggestions
    else:
        return ""


def judge_objects_descriptions(client, object_and_spans, object_desc_metrics):
    object_and_spans_filtered = {
        "object_names": [],
        "descriptions_spans": [],
        "objects_description": [],
    }

    for index, _ in enumerate(object_and_spans["objects_description"]):
        description_spans = object_and_spans["descriptions_spans"][index]
        object_description = object_and_spans["objects_description"][index]

        if len(description_spans) == 0 or len(object_description) == 0:
            continue

        object_and_spans_filtered["object_names"].append(object_and_spans["object_names"][index])
        object_and_spans_filtered["descriptions_spans"].append(
            "- " + "\n- ".join(description_spans)
        )
        object_and_spans_filtered["objects_description"].append(object_description)

    if len(object_and_spans_filtered["object_names"]) == 0:
        return (
            "",
            0,
            False,
            {
                "factual_accuracy": [],
                "coherence": [],
                "grounding_potential": [],
                "completeness": [],
            },
        )

    desc_judgements, total_token_count = judge_object_description(client, object_and_spans_filtered)

    if VERBOSE:
        print(desc_judgements)

    if not (
        len(desc_judgements["factual_accuracy"])
        == len(desc_judgements["coherence"])
        == len(desc_judgements["grounding_potential"])
        == len(desc_judgements["completeness"])
        == len(object_and_spans_filtered["objects_description"])
    ):
        if VERBOSE:
            print("The judge didn't provide all description scores.")

        return "", total_token_count, True, desc_judgements

    if BENCHMARKING:
        object_desc_metrics["factual_accuracy"].extend(desc_judgements["factual_accuracy"])
        object_desc_metrics["coherence"].extend(desc_judgements["coherence"])
        object_desc_metrics["grounding_potential"].extend(desc_judgements["grounding_potential"])
        object_desc_metrics["completeness"].extend(desc_judgements["completeness"])

        judge_suggestions = get_object_description_suggestions(
            object_and_spans_filtered, desc_judgements
        )
    else:
        judge_suggestions = ""

    return judge_suggestions, total_token_count, False, desc_judgements
