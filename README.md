# Car Detection API

This project exposes a FastAPI service that detects whether a car is present in an uploaded image using a custom YOLO model, then optionally returns the image with bounding boxes and labels.

## What’s in the repo

- `app/main.py` - FastAPI application and image-processing pipeline
- `app/car_detection_model.pt` - model artifact
- `app/car_detection_model_v2.pt` - model artifact used by the API
- `car_detection_notebook.ipynb` - experimentation / notebook workflow
- `sample_vehicle_image.jpg` - sample input image
- `output_valid_car_filtered_final.jpg` - example annotated output
- `requirements.txt` - Python dependencies

## Features

- Detects cars in uploaded images with YOLO
- Filters detections by confidence and bounding-box area
- Returns either JSON status only or the annotated image
- Enables CORS for browser-based clients

## Requirements

- Python 3.10+ recommended
- `pip`
- A working OpenCV-compatible environment
- The model file `app/car_detection_model_v2.pt`

## Install

Create a virtual environment if you want to keep dependencies isolated, then install:

```bash
pip install -r requirements.txt
```

## Run the API

The app loads the model using a relative path, so run it from the `app/` directory:

```bash
cd app
uvicorn main:app --reload
```

Or, to run the built-in Uvicorn entry point:

```bash
cd app
python main.py
```

The server starts on `http://0.0.0.0:8000`.

## API Endpoints

### `GET /`

Returns a simple service welcome message.

Example response:

```json
{
  "message": "Welcome to car detection service",
  "version": "version 1.0.0",
  "status": "operational"
}
```

### `GET /health`

Health check endpoint.

Example response:

```json
{
  "status": "healthy",
  "timestamp": "..."
}
```

### `POST /status_only/`

Uploads an image and returns detection status as JSON.

Form field:

- `file` - image file (`jpg`, `jpeg`, or `png`)

Example response:

```json
{
  "status": "Detected",
  "message": "Car detected and passed filtering.",
  "confidence": 93,
  "filter_applied": true,
  "processing_time_ms": 128.4
}
```

### `POST /status_and_image/`

Uploads an image and returns the annotated image as JPEG bytes.

Response headers:

- `X-Detection-Status`
- `X-Detection-Confidence`
- `X-Detection-Message`

The response body is the processed JPEG image.

## Example Usage

Using `curl` for the JSON-only endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/status_only/" \
  -F "file=@sample_vehicle_image.jpg"
```

Using `curl` for the annotated-image endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/status_and_image/" \
  -F "file=@sample_vehicle_image.jpg" \
  --output annotated.jpg
```

## Detection Logic

The API does more than just run YOLO inference. It also:

- keeps detections above a confidence threshold of `0.85`
- filters out small bounding boxes using a minimum normalized area ratio of `0.1`
- reports the largest remaining detection as the main car

## Notes

- Allowed upload formats are `jpg`, `jpeg`, and `png`
- If the model cannot be loaded, the API returns a server-side error until the path is corrected
- The notebook appears to be an experimentation workflow rather than a production entry point

## License

No license file is included in the repository yet.
