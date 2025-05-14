import os
import copy
import json
import time

from google import genai
from pydantic import BaseModel
from google.genai import types

from annotate_paintings_utils import load_image, image_to_bytes


def get_llm_client():
    with open("../../config/keys.json", "r") as file:
        os.environ["GEMINI_API_KEY"] = json.load(file)["gemini_api_key"]

    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def show_available_models(client):
    existing_models = [
        model.name.split("/")[1]
        for model in client.models.list()
        if "generateContent" in model.supported_actions
    ]

    print(existing_models)


def get_prompt(examples, image, description, additional_data, prompt_type):
    if prompt_type == "basic_object_extraction":
        return get_basic_object_extraction(examples, image, description)

    elif prompt_type == "basic_create_description":
        return get_basic_object_description(examples, additional_data)

    else:
        raise "Unknown prompt type"


def get_basic_object_extraction(examples, image, description):
    class Annotation(BaseModel):
        object_name: str
        description_spans: list[str]

    prompt_parts = []
    prompt_parts.append(
        types.Content(role="user", parts=[types.Part.from_text(text="\nHere are some examples:")])
    )

    for example in examples:
        example_description = example["description"]
        example_image = load_image(example["painting_id"])

        prompt_parts.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type="image/png", data=image_to_bytes(example_image)
                    ),
                    types.Part.from_text(text=f'Description: """{example_description}"""'),
                ],
            )
        )

        example_detected_objects = json.dumps(
            [
                Annotation(object_name=object_name, description_spans=description_spans).__dict__
                for object_name, description_spans in zip(
                    example["object_name"], example["description_spans"]
                )
            ],
        )

        prompt_parts.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=f"{example_detected_objects}")],
            )
        )
        prompt_parts.append(types.Content(role="user", parts=[types.Part.from_text(text="---")]))

    prompt_parts.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type="image/png", data=image_to_bytes(image)),
                types.Part.from_text(text=f'Description: """{description}"""'),
            ],
        )
    )

    system_prompt_text = (
        "You are an expert in art who can identify objects present in both a painting and its textual description."
        + "After identifying them, you return the objects together with their description spans extracted from the painting description in a JSON format following the provided template."
    )

    return prompt_parts, system_prompt_text, Annotation


def get_basic_object_description(examples, additional_data):
    class Annotation(BaseModel):
        object_description: str

    object_names = additional_data["object_names"]
    descriptions_spans = additional_data["descriptions_spans"]

    prompt_parts = []
    prompt_parts.append(
        types.Content(role="user", parts=[types.Part.from_text(text="\nHere are some examples:\n")])
    )

    for example_painting in examples:
        examples_per_object = list(
            zip(
                example_painting["object_name"],
                example_painting["description_spans"],
                example_painting["object_description"],
            )
        )

        formatted_input_example = ""
        formatted_output_example = []

        for example in examples_per_object:
            formatted_input_example += (
                f"Object name:{example[0]}\nDescription spans:{str(example[1])}\n\n"
            )
            formatted_output_example.append(Annotation(object_description=example[2]).__dict__)

        prompt_parts.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=formatted_input_example),
                ],
            )
        )

        prompt_parts.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=json.dumps(formatted_output_example))],
            )
        )
        prompt_parts.append(types.Content(role="user", parts=[types.Part.from_text(text="---")]))

    input_text = ""
    for object_name, description_spans in zip(object_names, descriptions_spans):
        input_text += f"Object name:{object_name}\nDescription spans:{str(description_spans)}\n\n"

    prompt_parts.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        )
    )

    system_prompt_text = (
        "You are given the name of objects and several short description spans about each of them. "
        + "Your task is to combine these spans into one coherent description paragraph per object that starts with the object name and which is based solely on the provided information. "
        + "In each description, you have to included all the provided details from the associated description spans and nothing more. "
        + """\n**Constraints:\n**
Do not add any details about the object that are not explicitly mentioned in the provided description spans.
Do not infer the object's material, purpose, or origin unless it is directly stated in the text.
Focus on combining and rephrasing the given information, not on creating new information.
Do not assume anything about the object's cultural significance or symbolism unless the provided spans mention it."""
    )

    return prompt_parts, system_prompt_text, Annotation


def generate(
    client,
    examples,
    image,
    description,
    additional_data,
    prompt_type,
    model_name,
    feedback,
    verbose,
):

    print("input description")
    prompt_parts, system_prompt_text, format_class = get_prompt(
        examples, image, copy.deepcopy(description), additional_data, prompt_type
    )

    if feedback:
        print("adding feedback")
        prompt_parts = feedback[0]
        prompt_parts.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=feedback[1])],
            )
        )
        prompt_parts.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=feedback[2])],
            )
        )

    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text=system_prompt_text),
        ],
        response_schema=list[format_class],
    )

    called = False
    trials = 3
    prompt_tokens_count = 0
    output_tokens_count = 0
    total_token_count = 0

    while not called or trials != 0:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_parts,
                config=generate_content_config,
            )
            output = response.parsed

            prompt_tokens_count += response.usage_metadata.prompt_token_count
            output_tokens_count += response.usage_metadata.candidates_token_count
            total_token_count += prompt_tokens_count + output_tokens_count

            if output is None:
                print("Output is None, try again...")
                trials -= 1
            else:
                break

            called = True

        except:
            print("Model is not available, try again...")
            time.sleep(5)

    if verbose:
        print(
            f"Prompt tokens count: {prompt_tokens_count}\nOutput tokens count: {output_tokens_count}"
        )
        print(f"Response:\n{output}\n")

    print("generated response !!!", response.text)

    return output, total_token_count, prompt_parts, response.text
