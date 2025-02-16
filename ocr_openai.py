import os
import sys
import base64
from openai import OpenAI
from pdf2image import convert_from_path
import tempfile

# Set your OpenAI API key
client = OpenAI(api_key="api_key")


def update_progress(progress):
    """
    Displays a simple progress bar in the console.
    """
    bar_length = 50
    block = int(round(bar_length * progress))
    text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {round(progress * 100, 2)}%"
    sys.stdout.write(text)
    sys.stdout.flush()


def encode_image(image_path):
    """
    Encodes an image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text_from_openai_api(image_path):
    """
    Sends the base64-encoded image to the OpenAI API and retrieves the extracted text.
    """
    base64_image = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Recognize all text from the image, including titles, chapter names, and body text. If it is considering as a title or chapter name, enclose it in angle brackets. otherwise just output the text content from the image without any additional descriptions or formatting.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        print(response.choices[0].message)
        return response.choices[0].message.content
        # return response.choices[0].message["content"]
    except Exception as e:
        print(f"\nError extracting text from image {image_path}: {e}")
        return ""


def process_pdf(pdf_path, output_txt_path):
    """
    Converts each page of the PDF to an image, extracts text using OpenAI API, and writes to a TXT file.
    """
    try:
        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path)
        total_pages = len(images)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return

    extracted_text = []

    for idx, image in enumerate(images, start=1):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            image_path = temp_image.name
            image.save(image_path, "JPEG")

        # Extract text from image using OpenAI API
        text = extract_text_from_openai_api(image_path)
        extracted_text.append(text)

        # Remove the temporary image file
        os.remove(image_path)

        # Update progress
        update_progress(idx / total_pages)

    # Write all extracted text to the output TXT file
    try:
        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(extracted_text))
        print(f"\nText extraction complete. Output saved to: {output_txt_path}")
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

    if not os.path.isfile(pdf_path):
        print(f'The path "{pdf_path}" does not exist or is not a file.')
        return

    if not pdf_path.lower().endswith(".pdf"):
        print("The provided file is not a PDF.")
        return

    # Define the output TXT file path
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.dirname(pdf_path)
    output_txt_path = os.path.join(output_dir, f"{base_name}_openai.txt")

    # Process the PDF and extract text
    process_pdf(pdf_path, output_txt_path)


if __name__ == "__main__":
    main()
