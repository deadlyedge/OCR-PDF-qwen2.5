import base64
import json
import tempfile
import torch

from os import path, remove
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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


def encode_image(image_path):
    """
    Encodes an image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def inference(
    image,  # Ensure this is not None
    image_path,
    prompt="",
    sys_prompt="You are a helpful assistant.",
    max_new_tokens=4096,
    return_input=False,
):
    if image is None or image_path is None:
        raise ValueError("Image bytes cannot be None.")

    image_bytes = encode_image(image_path)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # "min_pixels": min_pixels,
                    # "max_pixels": max_pixels,
                    "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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


def process_pdf(pdf_path, output_path):
    """
    Converts each page of the PDF to an image, extracts text using OpenAI API, and writes to a TXT file.
    """
    try:
        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return

    extracted_text = []

    with tqdm(total=len(images), desc="Recognizing images") as pbar:
        for page_number, image in enumerate(images, start=1):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
                image_path = temp_image.name
                image.save(image_path, "JPEG")
                recognized_text = inference(
                    image,
                    image_path,
                    prompt="Please output all text content from the image without any additional descriptions or formatting.",
                )
                extracted_text.append({"page": page_number, "content": recognized_text})

            # Remove the temporary image file
            remove(image_path)

            pbar.update(1)

    # Write all extracted text to the output TXT file
    try:
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(extracted_text, json_file, ensure_ascii=False, indent=4)
        print(f"\nText extraction complete. Output saved to: {output_path}")
    except Exception as e:
        print(f"\nError writing to TXT file: {e}")


def main():
    """
    Main function to execute the PDF to TXT conversion.
    """
    print("\n********************************")
    print("*** General PDF to TXT Converter ***")
    print("********************************\n")

    # Prompt user for the PDF file path
    pdf_path = "data/厌恶及其他mini.pdf"
    # pdf_path = input("Enter the full path to the PDF file: ").strip()

    if not path.isfile(pdf_path):
        print(f'The path "{pdf_path}" does not exist or is not a file.')
        return

    if not pdf_path.lower().endswith(".pdf"):
        print("The provided file is not a PDF.")
        return

    # Define the output TXT file path
    base_name = path.splitext(path.basename(pdf_path))[0]
    output_dir = path.dirname(pdf_path)
    output_path = path.join(output_dir, f"{base_name}_openai.json")

    # Process the PDF and extract text
    process_pdf(pdf_path, output_path)


if __name__ == "__main__":
    main()
