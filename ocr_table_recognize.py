import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
from tqdm import tqdm
import base64
from io import BytesIO
from openai import OpenAI
import json
import time


def list_image_files(folder_path: str) -> List[str]:
    """List jpg and png files in the given folder (non-recursive)."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f.lower())[1] in valid_extensions
    ]
    return files


def recognize_image(image_path: str) -> str:
    """Load image, resize to 50%, and run OCR inference using OpenAI API via OpenRouter."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")

    image = Image.open(image_path).convert("RGB")
    # Resize image to 50% of original size
    new_size = (image.width // 2, image.height // 2)
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = (
        "Identify table contents from the image without missing any columns or rows. "
        "Output the table contents in plain text format, with rows separated by line breaks "
        "and columns separated by tabs. Do not add additional descriptions or formatting."
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=messages,
        max_completion_tokens=8 * 1024,
        temperature=0,
    )

    recognized_text = ""
    if response.choices and len(response.choices) > 0:
        recognized_text = response.choices[0].message.content
    if recognized_text is None:
        recognized_text = ""

    return recognized_text


def parse_table_text(text: str) -> List[List[str]]:
    """Parse OCR text output into a 2D list representing the table."""
    rows = text.strip().split("\n")
    table = [row.split("\t") for row in rows]
    return table


def save_tables_to_excel(tables: List[Tuple[str, List[List[str]]]], output_path: str):
    """Save multiple tables to an Excel file, each table in a separate sheet."""
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, table_data in tables:
            df = pd.DataFrame(table_data)
            # Optional: set first row as header if appropriate
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False, header=False)


def main():
    folder_path = "data/老陆交易流水"  # Folder containing images
    output_excel = f"{folder_path}/recognized_tables.xlsx"
    temp_json = f"{folder_path}/recognized_tables_temp.json"

    image_files = list_image_files(folder_path)
    tables = []

    # Load temp data if exists
    if os.path.exists(temp_json):
        with open(temp_json, "r", encoding="utf-8") as f:
            tables = json.load(f)

    processed_files = {name for name, _ in tables}

    for image_file in tqdm(image_files, desc="Processing images"):
        if os.path.splitext(image_file)[0] in processed_files:
            continue  # Skip already processed

        image_path = os.path.join(folder_path, image_file)

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                recognized_text = recognize_image(image_path)
                break
            except Exception as e:
                print(f"Error recognizing {image_file} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    recognized_text = ""
        table_data = parse_table_text(recognized_text)
        tables.append((os.path.splitext(image_file)[0], table_data))

        # Save temp after each image
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)

    save_tables_to_excel(tables, output_excel)
    print(f"Recognition complete. Results saved to {output_excel}")


if __name__ == "__main__":
    main()
