# Edge Detection API

A simple FastAPI backend for edge detection on images (RGB and thermal).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- `GET /` - Returns API status

### Edge Detection
- `POST /detect-edges` - Accepts an image file and returns both original and edge-detected versions
  - **Parameters:**
    - `file`: Image file (JPG or PNG only)
    - `method`: Edge detection method - `canny` (default) or `sobel`
  - **Returns:** JSON response with:
    - `original_image`: Base64 encoded original image (data URI format)
    - `edge_detected_image`: Base64 encoded edge-detected image (PNG, data URI format)
    - `method`: Edge detection method used
    - `filename`: Original filename

## Example Usage

```bash
curl -X POST "http://localhost:8000/detect-edges?method=canny" \
  -F "file=@your_image.jpg"
```

### Response Format

```json
{
  "original_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "edge_detected_image": "data:image/png;base64,iVBORw0KGgo...",
  "method": "canny",
  "filename": "your_image.jpg"
}
```

Both images are returned as data URIs that can be directly used in HTML `<img>` tags or displayed in the frontend.

## Supported Formats
- JPG/JPEG
- PNG

## Edge Detection Methods
- **Canny**: Default method, good for general purpose edge detection
- **Sobel**: Alternative method using gradient-based edge detection

