from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional

app = FastAPI(title="Edge Detection API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image_format(filename: str) -> bool:
    """Validate that the uploaded file is a JPG or PNG."""
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def validate_image_quality(image_bgr: np.ndarray) -> list[dict]:
    """
    Check whether an image is suitable for edge detection.

    Accepts low-light and bright-light images as long as they have sufficient
    contrast and sharpness. Rejects only images where edge detection cannot
    produce meaningful results.

    Returns a list of issues. Empty list means the image is acceptable.
    Each issue has:
        - 'code': machine-readable key
        - 'message': human-readable explanation
        - 'metric': the measured value
        - 'threshold': the limit that was violated
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_dev = float(np.std(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    issues = []

    # Pitch black — no signal at all (low light images still have some variation)
    if mean_brightness < 8 and std_dev < 8:
        issues.append({
            "code": "TOO_DARK",
            "message": "Image is too dark with no detectable features. Increase exposure or use a brighter light source.",
            "metric": round(mean_brightness, 2),
            "threshold": 8,
        })
        # No point checking further — the image is essentially empty
        return issues

    # Fully overexposed — all detail washed out
    if mean_brightness > 247 and std_dev < 8:
        issues.append({
            "code": "OVEREXPOSED",
            "message": "Image is fully overexposed with no detectable features. Reduce exposure or avoid direct strong light.",
            "metric": round(mean_brightness, 2),
            "threshold": 247,
        })
        return issues

    # Low contrast — flat or single-colour image with no edges worth detecting.
    # Skip the blur check if this triggers: blur is meaningless on a uniform image.
    if std_dev < 12:
        issues.append({
            "code": "LOW_CONTRAST",
            "message": "Image has insufficient contrast for edge detection. Ensure the scene has visible variation in brightness.",
            "metric": round(std_dev, 2),
            "threshold": 12,
        })
        return issues

    # Too blurry — edges are smeared beyond detection.
    # Note: noisy low-light images tend to have *high* Laplacian variance, so this
    # threshold does not falsely reject dark-but-sharp images.
    if laplacian_variance < 50:
        issues.append({
            "code": "TOO_BLURRY",
            "message": "Image is too blurry for edge detection. Reduce motion blur or improve focus.",
            "metric": round(laplacian_variance, 2),
            "threshold": 50,
        })

    return issues


def perform_edge_detection(image_array: np.ndarray, method: str = "canny") -> np.ndarray:
    """
    Perform edge detection on an image.
    
    Args:
        image_array: Input image as numpy array
        method: Edge detection method ('canny' or 'sobel')
    
    Returns:
        Edge-detected image as numpy array
    """
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    
    if method == "canny":
        # Canny edge detection - good for general purpose
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny edge detection with adaptive thresholds
        edges = cv2.Canny(blurred, 50, 150)
    elif method == "sobel":
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(np.clip(edges, 0, 255))
    else:
        # Default to Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
    
    return edges


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Edge Detection API is running", "status": "healthy"}


@app.post("/detect-edges")
async def detect_edges(
    file: UploadFile = File(...),
    method: Optional[str] = "canny"
):
    """
    Accept an image file and return both original and edge-detected versions.
    
    Args:
        file: Image file (JPG or PNG)
        method: Edge detection method - 'canny' (default) or 'sobel'
    
    Returns:
        JSON response with both original and edge-detected images as base64 encoded strings
    """
    # Validate file format
    if not validate_image_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only JPG and PNG images are allowed."
        )
    
    # Validate method
    if method not in ["canny", "sobel"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid method. Use 'canny' or 'sobel'."
        )
    
    try:
        # Read image file
        contents = await file.read()
        
        # Store original image bytes for response
        original_image_bytes = contents
        
        # Convert to PIL Image for format handling
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed (handles RGBA, L, etc.)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV (OpenCV uses BGR format)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Reject images that are unsuitable for edge detection
        quality_issues = validate_image_quality(image_bgr)
        if quality_issues:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Image rejected: not suitable for edge detection.",
                    "issues": quality_issues,
                },
            )

        # Perform edge detection
        edges = perform_edge_detection(image_bgr, method)
        
        # Convert back to RGB for response
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        result_image = Image.fromarray(edges_rgb)
        
        # Save edge-detected image to bytes buffer
        edges_buffer = io.BytesIO()
        result_image.save(edges_buffer, format="PNG")
        edges_buffer.seek(0)
        edges_bytes = edges_buffer.getvalue()
        
        # Encode both images as base64 strings
        original_base64 = base64.b64encode(original_image_bytes).decode('utf-8')
        edges_base64 = base64.b64encode(edges_bytes).decode('utf-8')
        
        # Determine original image format for proper MIME type
        original_format = "image/png" if file.filename.lower().endswith('.png') else "image/jpeg"
        
        return JSONResponse(
            content={
                "original_image": f"data:{original_format};base64,{original_base64}",
                "edge_detected_image": f"data:image/png;base64,{edges_base64}",
                "method": method,
                "filename": file.filename
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

