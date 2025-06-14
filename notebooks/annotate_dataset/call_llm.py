import os
import copy
import json
import time
from pprint import pprint

from google import genai
from pydantic import BaseModel
from google.genai import types

from config import *
from annotate_paintings_utils import load_image, image_to_bytes


def get_llm_client(location="global"):
    if USE_VERTEX:
        return genai.Client(project="enhancing-visual-grounding", vertexai=True, location=location)
    else:
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
        types.Content(
            role="user", parts=[types.Part.from_text(text="\n### Here are some examples ###\n")]
        )
    )

    for example in examples:
        example_description = example["description"]
        example_image, _ = load_image(example["painting_id"])

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
                parts=[types.Part.from_text(text=example_detected_objects)],
            )
        )

    prompt_parts.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="\n---\n### End of examples ###\n\nNow process in a similar way the following data:\n"
                )
            ],
        )
    )

    prompt_parts.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type="image/png", data=image_to_bytes(image)),
                types.Part.from_text(text=f'Description: """{description}"""'),
            ],
        )
    )

    return prompt_parts, OBJECT_EXTRACTION_ENHANCED_SYSTEM_PROMPT, Annotation


def get_basic_object_description(examples, additional_data):
    class Annotation(BaseModel):
        object_description: str

    object_names = additional_data["object_names"]
    descriptions_spans = additional_data["descriptions_spans"]

    prompt_parts = []
    prompt_parts.append(
        types.Content(
            role="user", parts=[types.Part.from_text(text="\n### Here are some examples ###\n")]
        )
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

    prompt_parts.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="\n---\n### End of examples ###\n\nNow process in a similar way the following data:\n"
                )
            ],
        )
    )

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

    return prompt_parts, OBJECT_DESCRIPTION_ENHANCED_SYSTEM_PROMPT, Annotation


def generate(
    client,
    backup_client,
    examples,
    image,
    description,
    additional_data,
    prompt_type,
    feedback,
):
    prompt_parts, system_prompt_text, format_class = get_prompt(
        examples, image, copy.deepcopy(description), additional_data, prompt_type
    )

    if feedback:
        if VERBOSE:
            print("ADDING FEEDBACK")

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
        max_output_tokens=3072,
        response_schema=list[format_class],
    )

    generate_content_config_thinking = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text=system_prompt_text),
        ],
        max_output_tokens=3072,
        response_schema=list[format_class],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    called = False
    trials = 2
    prompt_tokens_count = 0
    output_tokens_count = 0
    total_token_count = 0

    while not called or trials != 0:
        try:
            if trials == 2:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt_parts,
                    config=generate_content_config,
                )
            else:
                response = backup_client.models.generate_content(
                    # if GEMINI_MODEL_BACKUP is replaced by GEMINI_MODEL, change the config too
                    model=GEMINI_MODEL_BACKUP,
                    contents=prompt_parts,
                    config=generate_content_config_thinking,
                )

            output = response.parsed

            try:
                output_tokens_count += response.usage_metadata.thoughts_token_count
            except:
                pass

            try:
                prompt_tokens_count += response.usage_metadata.prompt_token_count
                output_tokens_count += response.usage_metadata.candidates_token_count
                total_token_count += prompt_tokens_count + output_tokens_count
            except:
                pass

            if output is None:
                if VERBOSE:
                    print("Output is None, try again...")
                    pprint(response)
                trials -= 1
            else:
                break

            called = True

        except Exception as e:
            if VERBOSE:
                print("Model is not available, try again...")
            time.sleep(5)

    if VERBOSE:
        print(
            f"\nANNOTATOR OUTPUT:\nPrompt tokens count: {prompt_tokens_count} Output tokens count: {output_tokens_count}"
        )

    return output, total_token_count, prompt_parts, response.text
