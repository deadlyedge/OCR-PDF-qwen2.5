import os
import sys
import tempfile
from google.generativeai.client import configure
from google.generativeai.files import upload_file
from google.generativeai.generative_models import GenerativeModel
from pdf2image import convert_from_path
from PIL import Image
import time
from datetime import datetime
# import glob

# Set Gemini API key directly
configure(api_key="api_key")


def countdown_timer(seconds):
    """
    Display a countdown timer.
    """
    for remaining in range(seconds, 0, -1):
        sys.stdout.write(f"\rWaiting for {remaining} seconds...  ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rWait complete!            \n")
    sys.stdout.flush()


def update_progress(progress):
    """
    Displays a simple progress bar in the console.
    """
    bar_length = 50
    block = int(round(bar_length * progress))
    text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {round(progress * 100, 2)}%"
    sys.stdout.write(text)
    sys.stdout.flush()


def extract_text_from_gemini_api(image_path, page_num):
    """
    Sends the image to the Gemini API and retrieves the extracted text.
    Added detailed logging and error information.
    """
    try:
        print(f"\nProcessing page {page_num}:")
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Uploading image to Gemini API..."
        )

        myfile = upload_file(image_path)
        model = GenerativeModel("gemini-1.5-flash")
        # model = genai.GenerativeModel("gemini-1.5-pro")

        # Add safety settings to reduce false positives
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Sending request to Gemini API..."
        )

        # Generate content with modified prompt and safety settings
        result = model.generate_content(
            [
                myfile,
                "\n\n",
                "Recognize all text from the image, including titles, chapter names, and body text. If it is considering as a title or chapter name, enclose it in angle brackets. otherwise just output the text content from the image without any additional descriptions or formatting.",
            ],
            safety_settings=safety_settings,
        )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Response received from Gemini API"
        )

        # Check if response has content
        if hasattr(result, "text"):
            print("Successfully extracted text from image")
            return result.text
        elif hasattr(result, "candidates"):
            # Try to get text from candidates
            for candidate in result.candidates:
                if hasattr(candidate, "content"):
                    print("Successfully extracted text from candidates")
                    return candidate.content.text

        print("Warning: No text content found in API response")
        return "No text could be extracted from this image."

    except Exception as e:
        error_message = f"\nError processing page {page_num}:\n"
        error_message += f"Error Type: {type(e).__name__}\n"
        error_message += f"Error Message: {str(e)}\n"

        if hasattr(e, "status_code"):
            error_message += f"Status Code: {e.status_code}\n"
        if hasattr(e, "response"):
            error_message += f"Response: {e.response}\n"
        if hasattr(e, "details"):
            error_message += f"Details: {e.details}\n"

        print(error_message)
        return f"[ERROR ON PAGE {page_num}]: {error_message}"


def process_pdf(pdf_path, output_txt_path):
    """
    Converts each page of the PDF to an image, extracts text using Gemini API,
    and writes it to a TXT file. Includes a 70-second delay between API calls.
    """
    try:
        print("\nInitializing PDF processing...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Converting PDF to images...")
        images = convert_from_path(pdf_path)
        total_pages = len(images)
        print(f"Total pages detected: {total_pages}")
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return

    extracted_text = []
    for idx, image in enumerate(images, start=1):
        print(f"\n{'=' * 50}")
        print(f"Processing page {idx} of {total_pages}")
        print(f"{'=' * 50}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            image_path = temp_image.name

        # Save image
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving temporary image...")
        image.save(image_path, "JPEG")
        print(f"Temporary image saved to: {image_path}")

        # Extract text
        text = extract_text_from_gemini_api(image_path, idx)
        extracted_text.append(text)

        # Remove temporary file immediately after getting the response
        try:
            os.remove(image_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Temporary image removed")
        except Exception as e:
            print(f"\nWarning: Could not remove temporary file {image_path}: {e}")

        # Update progress
        update_progress(idx / total_pages)

        # Add 70-second delay unless I tweak this between API calls if there are more pages to process
        if idx < total_pages:
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting 70-second cooldown period..."
            )
            countdown_timer(5)

    # Write all extracted text to the output TXT file
    try:
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Writing extracted text to file..."
        )
        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n\n".join(extracted_text))
        print(f"Text extraction complete. Output saved to: {output_txt_path}")
    except Exception as e:
        print(f"\nError writing to TXT file: {e}")


def main():
    """
    Main function to execute the PDF to TXT conversion.
    """
    print("\n********************************")
    print("*** General PDF to TXT Converter ***")
    print("********************************\n")

    # Get list of PDFs in current directory
    pdf_files = ["data/厌恶及其他mini.pdf"]
    # pdf_files = glob.glob("*.pdf")

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    print("Available PDF files:")
    print("-" * 50)
    for idx, pdf in enumerate(pdf_files, 1):
        print(f"{idx}. {pdf}")
    print("-" * 50)

    # Get user selection
    while True:
        try:
            selection = input(
                "\nEnter the number of the PDF you want to process (or 'q' to quit): "
            )

            if selection.lower() == "q":
                print("Exiting program.")
                return

            selection = int(selection)
            if 1 <= selection <= len(pdf_files):
                break
            else:
                print(f"Please enter a number between 1 and {len(pdf_files)}")
        except ValueError:
            print("Please enter a valid number")

    # Get the selected PDF file
    pdf_path = pdf_files[selection - 1]
    print(f"\nSelected: {pdf_path}")

    # Define the output TXT file path
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_txt_path = f"{base_name}.txt"

    # Process the PDF and extract text
    process_pdf(pdf_path, output_txt_path)


if __name__ == "__main__":
    main()
