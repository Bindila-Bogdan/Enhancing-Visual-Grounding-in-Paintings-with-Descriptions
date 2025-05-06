import io
import os
import copy
import json
import time

from google import genai
from pydantic import BaseModel
from google.genai import types

from annotate_paintings_utils import load_image


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


def image_to_bytes(image):
    # define an in-memory byte stream
    img_byte_array = io.BytesIO()

    # convert the image to a byte representation and store it in the in-memory byte stream
    image.save(img_byte_array, format="PNG")

    # get the byte representation of the image
    img_bytes = img_byte_array.getvalue()

    return img_bytes


def get_prompt(examples, image, description, prompt_type):
    if prompt_type == "basic":
        return get_basic_prompt(examples, image, description)

    elif prompt_type == "basic_with_spans":
        return get_basic_with_spans_prompt(examples, image, description)

    elif prompt_type == "basic_complete":
        return get_basic_complete(examples, image, description)

    else:
        raise "Unknown prompt type"


def get_basic_prompt(examples, image, description):
    class Annotation(BaseModel):
        object_name: str

    prompt_parts = []
    prompt_parts.append(
        types.Content(role="user", parts=[types.Part.from_text(text="\nHere are some examples:")])
    )

    for example in examples:
        example_painting_id = example["painting_id"]
        example_description = example["description"]
        example_image = load_image(example_painting_id)

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
                Annotation(object_name=object_name).__dict__
                for object_name in example["object_name"]
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
                types.Part.from_text(
                    text=f'Description: """{description}"""\n\n'
                    + "Return **ONLY** the objects (lowercased) described in the given painting that also appear in the textual description."
                ),
            ],
        )
    )

    system_prompt_text = (
        "You are an expert in art who can identify objects present in both a painting and its textual description."
        + "After identifying them, you return the objects in a JSON format following the provided template."
    )

    return prompt_parts, system_prompt_text, Annotation


def get_basic_with_spans_prompt(examples, image, description):
    class Annotation(BaseModel):
        object_name: str
        description_spans: list[str]

    prompt_parts = []
    prompt_parts.append(
        types.Content(role="user", parts=[types.Part.from_text(text="\nHere are some examples:")])
    )

    for example in examples:
        example_painting_id = example["painting_id"]
        example_description = example["description"]
        example_image = load_image(example_painting_id)

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


def get_basic_complete(examples, image, description):
    class Annotation(BaseModel):
        object_name: str
        description_spans: list[str]
        object_description: str

    prompt_parts = []
    prompt_parts.append(
        types.Content(role="user", parts=[types.Part.from_text(text="\nHere are some examples:")])
    )

    for example in examples:
        example_painting_id = example["painting_id"]
        example_description = example["description"]
        example_image = load_image(example_painting_id)

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
                Annotation(
                    object_name=object_name,
                    description_spans=description_spans,
                    object_description=object_description,
                ).__dict__
                for object_name, description_spans, object_description in zip(
                    example["object_name"],
                    example["description_spans"],
                    example["object_description"],
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
        + "After identifying them, extact for each object the description spans from the painting descriptions. "
        + "Finally, create a single, coherent description paragraph that starts with the object name of the object based solely on the provided information."
        + "In this description, you have to included all the provided details from the description spans"
        + """**Constraints:**
Do not add any details about the object that are not explicitly mentioned in the provided description spans.
Do not infer the object's material, purpose, or origin unless it is directly stated in the text.
Focus on combining and rephrasing the given information, not on creating new information.
Do not assume anything about the object's cultural significance or symbolism unless the provided spans mention it."""
    )

    return prompt_parts, system_prompt_text, Annotation


def generate(client, examples, image, description, prompt_type, model_name, verbose):
    prompt_parts, system_prompt_text, format_class = get_prompt(
        examples, image, copy.deepcopy(description), prompt_type
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

    while not called:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_parts,
                config=generate_content_config,
            )
            output = response.parsed
            called = True
        except:
            print("Try again...")
            time.sleep(5)

    prompt_tokens_count = response.usage_metadata.prompt_token_count
    output_tokens_count = response.usage_metadata.candidates_token_count
    total_token_count = prompt_tokens_count + output_tokens_count

    if verbose:
        print(
            f"Prompt tokens count: {prompt_tokens_count}\nOutput tokens count: {output_tokens_count}"
        )
        print(f"Response:\n{output}\n")

    return output, total_token_count
