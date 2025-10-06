# Face Detection (OpenCV, Python)

A minimal face detection project using OpenCV Haar cascades.

## Setup

Install dependencies into your Python environment. Example (macOS / zsh):

```bash
python -m pip install -r requirements.txt
```

## Run

Detect faces in an image and save annotated copy:

```bash
python -m src.detect image path/to/input.jpg --output annotated.jpg
```

Run webcam (press 'q' to quit):

```bash
python -m src.detect webcam --source 0
```

## Tests

Run the small unit test:

```bash
python -m pytest -q
```
