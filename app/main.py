from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import io
from ultralytics import YOLO
import supervision as sv # Needed for Detections object
import uvicorn

import time

from typing import Dict, Any, List

# --- CONFIGURATION ---
MODEL_PATH = "car_detection_model_v2.pt" 
TARGET_SIZE = 640 
CONFIDENCE_THRESHOLD = 0.85 
MIN_NORMALIZED_AREA_RATIO = 0.1
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


app = FastAPI(
    title="Car Detection API",
    version="1.0.0",
    description="Service to detect subject car presence using YOLO with post-filtering and image visualization.",
)

# Allow CORS for front-end communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZATION ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"ERROR: Could not load model from {MODEL_PATH}. {e}")
    model = None 

# Pydantic model for the response structure
class DetectionStatus(BaseModel):
    status: str
    message: str
    confidence: int # Percentage value (0-100)
    filter_applied: bool
    processing_time_ms: float

# --- CORE PROCESSING FUNCTION ---
def process_image_and_filter(image_original: np.ndarray):
    """
    Performs YOLO inference and applies combined post-filtering logic.
    Returns: detections (filtered sv.Detections object) and original image dimensions.
    """
    H, W, _ = image_original.shape
    TOTAL_IMAGE_AREA = H * W

    # Run Inference
    results = model(image_original, imgsz=TARGET_SIZE, conf=0.25, verbose=False)[0] 
    detections = sv.Detections.from_ultralytics(results)
    
    # --- 1. Filter by Confidence Score ---
    detections = detections[detections.confidence >= CONFIDENCE_THRESHOLD]
    
    # --- 2. Filter by Bounding Box Size (Normalized Area) ---
    if len(detections) > 0:
        xyxy = detections.xyxy
        box_widths = xyxy[:, 2] - xyxy[:, 0]
        box_heights = xyxy[:, 3] - xyxy[:, 1]
        box_areas = box_widths * box_heights
        
        normalized_areas = box_areas / TOTAL_IMAGE_AREA
        area_filter = normalized_areas >= MIN_NORMALIZED_AREA_RATIO
        detections = detections[area_filter]

    return detections, H, W

# --- ANNOTATION FUNCTION (Based on your working OpenCV code) ---
def annotate_image(image_original: np.ndarray, detections: sv.Detections):
    """
    Annotates the image with bounding box and multi-line label for all valid detections.
    """
    annotated_image = image_original.copy()
    
    # Find the largest car to ensure we report its confidence
    final_xyxy = detections.xyxy
    final_box_widths = final_xyxy[:, 2] - final_xyxy[:, 0]
    final_box_heights = final_xyxy[:, 3] - final_xyxy[:, 1]
    final_box_areas = final_box_widths * final_box_heights
    largest_index = np.argmax(final_box_areas)
    
    # Use only the largest car for the confidence report
    main_car_detection = detections[largest_index:largest_index+1]
    
    # Draw boxes and labels for ALL remaining valid detections
    for box, conf in zip(detections.xyxy, detections.confidence):

        x1, y1, x2, y2 = map(int, box)

        # Bounding box (RED)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0,0,255), 2)

        # Label lines with the requested format
        line1 = f"Detected = car"
        line2 = f"confidence = {conf*100:.0f}%"
        
        # --- Multi-Line Text Placement ---
        TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
        TEXT_SCALE = 0.7
        TEXT_THICKNESS = 2
        TEXT_PAD = 5
        LINE_SPACING = 5 

        (w1, h1), _ = cv2.getTextSize(line1, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
        (w2, h2), _ = cv2.getTextSize(line2, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
        
        max_w = max(w1, w2)
        total_h = h1 + h2 + LINE_SPACING 
        
        # Place the text block at the top-left corner of the bounding box
        text_x = x1
        text_y = y1 - total_h - TEXT_PAD 

        # Fallback if label goes off the top edge of the image
        if text_y < 0:
            text_y = y1 + h1 + TEXT_PAD 

        # Background Rectangle (BLACK)
        cv2.rectangle(annotated_image, 
                      (text_x, text_y - h1 - TEXT_PAD), 
                      (text_x + max_w + TEXT_PAD*2, text_y + total_h + TEXT_PAD*2), 
                      (0,0,0), -1)

        # Plot Line 1 (White text)
        cv2.putText(annotated_image, line1, 
                    (text_x + TEXT_PAD, text_y - TEXT_PAD), 
                    TEXT_FONT, TEXT_SCALE,
                    (255,255,255), TEXT_THICKNESS, cv2.LINE_AA) 

        # Plot Line 2 (White text)
        cv2.putText(annotated_image, line2, 
                    (text_x + TEXT_PAD, text_y + h2 + LINE_SPACING), 
                    TEXT_FONT, TEXT_SCALE,
                    (255,255,255), TEXT_THICKNESS, cv2.LINE_AA) 

    # Return the annotated image (OpenCV BGR format)
    return annotated_image

# --- INPUT VALIDATION AND PREPARATION ---
def validate_and_prepare_image(file: UploadFile):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized. Check server logs.")
    
    # 1. Image Extension Validation
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # 2. Image Decoding
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_original is None:
            raise ValueError("Could not decode image. File may be corrupted.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading or decoding image: {e}")

    return image_original

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with system information."""
    return {
        "message": "Welcome to car detection service",
        "version": "version 1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": str(time.time())}

# --- ENDPOINT 1: STATUS ONLY (Existing functionality, updated status) ---

@app.post("/status_only/", response_model=DetectionStatus)
async def get_detection_status(file: UploadFile = File(...)):
    """
    Returns only the detection status and confidence in JSON format.
    """
    # 1. Start timer
    start_time = time.time()
    
    image_original = validate_and_prepare_image(file)
    detections, _, _ = process_image_and_filter(image_original)
    
    highest_confidence = 0
    
    if len(detections) > 0:
        # Find the highest confidence among the remaining valid detections
        box_areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        largest_car_index = np.argmax(box_areas) 
        highest_confidence = int(detections.confidence[largest_car_index] * 100)
        
        status_str = "Detected"
        message_str = "Car detected and passed filtering."
    else:
        status_str = "Not Detected"
        message_str = "No valid subject car found."
    
    # 2. End timer and calculate duration in milliseconds
    end_time = time.time()
    processing_time_ms = round((end_time - start_time) * 1000, 2)
    
    return {
        "status": status_str,
        "message": message_str,
        "confidence": highest_confidence,
        "filter_applied": True,
        "processing_time_ms": processing_time_ms
    }


# --- ENDPOINT 2: STATUS + ANNOTATED IMAGE (New functionality) ---

@app.post("/status_and_image/")
async def get_detection_and_image(file: UploadFile = File(...)):
    """
    Returns the detection status in a header and the image with 
    bounding boxes/labels as binary data (JPEG format).
    """
    # 1. Start timer
    start_time = time.time()
    
    image_original = validate_and_prepare_image(file)
    detections, _, _ = process_image_and_filter(image_original)
    
    highest_confidence = 0.0
    
    if len(detections) > 0:
        # Annotation only proceeds if a valid car is found
        annotated_image = annotate_image(image_original, detections)
        
        # Calculate final status and confidence
        box_areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        largest_car_index = np.argmax(box_areas) 
        highest_confidence = detections.confidence[largest_car_index]

        status_str = "Detected"
        message_str = "Car detected and passed filtering."
    else:
        # If no car is detected, return the original image unannotated
        annotated_image = image_original 
        status_str = "Not Detected"
        message_str = "No valid subject car found."
        
    # 2. End timer and calculate duration in milliseconds
    end_time = time.time()
    processing_time_ms = round((end_time - start_time) * 1000, 2)

    # Prepare JSON status data for the response header
    response_data = {
        "status": status_str,
        "confidence": int(highest_confidence * 100),
        "message": message_str,
        "processing_time_ms": processing_time_ms # NEW FIELD
    }
    
    # Encode the image (OpenCV BGR format) to JPEG binary data
    is_success, buffer = cv2.imencode(".jpg", annotated_image)
    if not is_success:
        raise HTTPException(status_code=500, detail="Could not encode annotated image to JPEG.")

    # Return the image binary data with custom headers
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Detection-Status": status_str, 
                 "X-Detection-Confidence": str(int(highest_confidence * 100)),
                 "X-Detection-Message": message_str,
                 "Access-Control-Expose-Headers": "X-Detection-Status, X-Detection-Confidence, X-Detection-Message" # Necessary for client-side JS/mobile access
                }
    )

# --- OPTIONAL: Run with Uvicorn ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)