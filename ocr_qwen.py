import base64
import json
import pymupdf  # PyMuPDF
import torch

from io import BytesIO
from os import path
from typing import Any, List, Tuple
from tqdm import tqdm
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download


# download model
model_dir = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct")

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="cuda"
)


# default processor
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_dir,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)


# def decodeBytes(input) -> bytes:
#     string_base64 = input.strip("=")
#     data_len = len(string_base64) % 4
#     if data_len == 1:
#         string_base64 = string_base64[:-1]
#     elif data_len:
#         string_base64 += data_len * "="
#     return base64.b64decode(string_base64)


def inference(
    image,  # Ensure this is not None
    prompt="",
    sys_prompt="You are a helpful assistant.",
    max_new_tokens=4096,
    return_input=False,
):
    if image is None:
        raise ValueError("Image bytes cannot be None.")

    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print("text:", text)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]


def extract_images_from_pdf(pdf_path) -> List[Tuple[int, Any]]:
    images = []
    pdf_document = pymupdf.open(pdf_path)
    for page_num, page in enumerate(pdf_document):  # type: ignore
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (400, 600), pix.samples)

        # Append to the list
        images.append((page_num + 1, img))
    return images


def recognize_images(images: List[Tuple[int, Any]]) -> list[dict[str, str]]:
    results = []
    with tqdm(total=len(images), desc="Recognizing images") as pbar:
        for page_number, image in images:
            recognized_text = inference(
                image,
                prompt="Recognize all text from the image, including titles, chapter names, and body text. If it is considering as a title or chapter name, enclose it in angle brackets. otherwise just output the text content from the image without any additional descriptions or formatting.",
                # prompt="If it is a title or chapter name, enclose it in angle brackets, otherwise just output the text content from the image without any additional descriptions or formatting.",
            )
            results.append({"page": page_number, "content": recognized_text})
            pbar.update(1)
    return results


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def main():
    pdf_path = "data/厌恶及其他mini.pdf"  # Path to the PDF file
    output_path = f"{path.splitext(pdf_path)[0]}_output.json"  # Output JSON file name

    images = extract_images_from_pdf(pdf_path)
    # print(images)
    recognized_data = recognize_images(images)
    save_to_json(recognized_data, output_path)


if __name__ == "__main__":
    main()
