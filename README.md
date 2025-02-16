# OCR-PDF Project

## Overview
The OCR-PDF project is designed to extract text from PDF documents using Optical Character Recognition (OCR) techniques. This project leverages various OCR models to provide accurate and efficient text extraction.

## Features
- Support for multiple OCR models including:
  - Gemini
  - OpenAI
  - Qwen-2.5-vl(locale model)
- Ability to convert extracted text into Word documents.
- Easy integration with existing workflows.

## Compared porformance

after testing the performance of the models, the results are as follows:
| Model Name         | Accuracy | Speed     |
| ------------------ | -------- | --------- |
| Gemini             | Low      | Fast      |
| OpenAI             | Low      | Fast      |
| Qwen(Locale Model) | High     | Very Slow |

(I use nvidia 4080 for testing)

## Installation
To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
To extract text from a PDF file, use the following command:

```bash
python main.py <path_to_pdf>
```

Replace `<path_to_pdf>` with the path to your PDF file.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
