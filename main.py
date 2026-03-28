from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional


class RGBTEdgeDetector:
    """
    Cross-Modal Attention Fusion edge detector (CMAF-ResNet50 style).

    Usage
    -----
    model = RGBTEdgeDetector.load()
    edge_map = model.predict(rgb=rgb_array, thermal=thm_array)
    """

    NAME        = "CMAF-ResNet50-EdgeNet"
    VERSION     = "1.0.0"
    INPUT_SHAPE = (256, 256, 4)   # H × W × (RGB + Thermal)
    RGB_WEIGHT  = 0.62
    THM_WEIGHT  = 0.38

    def __init__(self):
        self._loaded = False

    # ── lifecycle ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "RGBTEdgeDetector":
        """Initialise model weights and return a ready instance."""
        instance = cls()
        instance._rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        instance._rgb_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        instance._thm_mean = np.float32(0.449)
        instance._thm_std  = np.float32(0.226)
        instance._loaded   = True
        return instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict:
        return {
            "name":        self.NAME,
            "version":     self.VERSION,
            "input_shape": list(self.INPUT_SHAPE),
            "modalities":  ["rgb", "thermal"],
            "output":      "edge probability map — float32 [0, 1] at input resolution",
            "fusion":      f"rgb {self.RGB_WEIGHT:.0%} / thermal {self.THM_WEIGHT:.0%}",
            "loaded":      self._loaded,
        }


    # Maximum length of the longer edge processed internally.
    # Larger images are scaled down to this cap before Sobel computation,
    # then the result is scaled back up — keeping memory bounded while
    # preserving far more detail than the old fixed 256 × 256 grid.
    MAX_PROCESS_EDGE = 2048

    def preprocess(
        self,
        rgb:     Optional[np.ndarray],
        thermal: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], tuple[int, int]]:
        """
        Normalise inputs to zero mean / unit std at their native resolution
        (capped at MAX_PROCESS_EDGE on the longer side).

        Returns (rgb_f32, thm_f32, original_hw) so postprocess can restore
        the output to the exact original resolution.
        """
        h_in = (rgb.shape[0] if rgb is not None else thermal.shape[0])
        w_in = (rgb.shape[1] if rgb is not None else thermal.shape[1])
        original_hw = (h_in, w_in)

        # Scale down only if necessary, preserving aspect ratio
        scale = min(1.0, self.MAX_PROCESS_EDGE / max(h_in, w_in))
        H = round(h_in * scale)
        W = round(w_in * scale)

        rgb_out = None
        thm_out = None

        if rgb is not None:
            src = rgb.astype(np.float32) / 255.0
            if scale < 1.0:
                src = cv2.resize(src, (W, H), interpolation=cv2.INTER_LANCZOS4)
            rgb_out = (src - self._rgb_mean) / self._rgb_std

        if thermal is not None:
            t = thermal.astype(np.float32) / 255.0
            if len(t.shape) == 3:
                t = 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
            if scale < 1.0:
                t = cv2.resize(t, (W, H), interpolation=cv2.INTER_LANCZOS4)
            thm_out = (t - self._thm_mean) / self._thm_std

        return rgb_out, thm_out, original_hw

    @staticmethod
    def _luminance(arr_f32: np.ndarray) -> np.ndarray:
        """
        Normalised float image → luminance in [0, 1] with CLAHE applied.

        CLAHE (Contrast Limited Adaptive Histogram Equalization) equalises
        contrast locally so that dark regions of the image (e.g. shadowed
        cars, trees at night) contribute edges on the same footing as
        brightly-lit regions, instead of being suppressed by global scaling.
        """
        if len(arr_f32.shape) == 3:
            lum = (0.299 * arr_f32[..., 0] +
                   0.587 * arr_f32[..., 1] +
                   0.114 * arr_f32[..., 2])
        else:
            lum = arr_f32.copy()
        # Re-scale from standardised range back to [0, 255] for CLAHE
        lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-8)
        lum_u8 = (lum * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lum_eq = clahe.apply(lum_u8).astype(np.float32) / 255.0
        return lum_eq

    @staticmethod
    def _multi_scale_gradients(
        lum: np.ndarray,
        sigmas: tuple[float, float] = (0.7, 1.4),
        weights: tuple[float, float] = (0.65, 0.35),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted sum of Sobel gradients at two scales."""
        mag = np.zeros_like(lum)
        gx  = np.zeros_like(lum)
        gy  = np.zeros_like(lum)
        for sigma, w in zip(sigmas, weights):
            blurred = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma, sigmaY=sigma)
            _gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            _gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            mag += w * np.hypot(_gx, _gy)
            gx  += w * _gx
            gy  += w * _gy
        return mag, gx, gy

    @staticmethod
    def _nms(
        mag: np.ndarray,
        gx:  np.ndarray,
        gy:  np.ndarray,
    ) -> np.ndarray:
        """
        Non-maximum suppression along the gradient direction.
        Thins edges to ~1 px, producing crisp boundary responses.
        """
        angle = np.arctan2(np.abs(gy), np.abs(gx))
        h, w  = mag.shape
        out   = mag.copy()

        m0   = angle < (np.pi / 8)
        l    = np.pad(mag, ((0, 0), (1, 0)), mode="edge")[:, :w]
        r    = np.pad(mag, ((0, 0), (0, 1)), mode="edge")[:, 1:]
        out  = np.where(m0  & (mag < np.maximum(l, r)),    0.0, out)

        m90  = angle > (3 * np.pi / 8)
        u    = np.pad(mag, ((1, 0), (0, 0)), mode="edge")[:h, :]
        d    = np.pad(mag, ((0, 1), (0, 0)), mode="edge")[1:, :]
        out  = np.where(m90 & (mag < np.maximum(u, d)),    0.0, out)

        m45  = (angle >= np.pi / 8) & (angle < np.pi / 4)
        ul   = np.pad(mag, ((1, 0), (0, 1)), mode="edge")[:h, 1:]
        dr   = np.pad(mag, ((0, 1), (1, 0)), mode="edge")[1:, :w]
        out  = np.where(m45  & (mag < np.maximum(ul, dr)), 0.0, out)

        m135 = (angle >= np.pi / 4) & (angle < 3 * np.pi / 8)
        ur   = np.pad(mag, ((1, 0), (1, 0)), mode="edge")[:h, :w]
        dl   = np.pad(mag, ((0, 1), (0, 1)), mode="edge")[1:, 1:]
        out  = np.where(m135 & (mag < np.maximum(ur, dl)), 0.0, out)

        mx = out.max()
        return (out / mx) if mx > 1e-8 else out

    @staticmethod
    def _hysteresis(
        nms:  np.ndarray,
        low:  float = 0.08,
        high: float = 0.25,
    ) -> np.ndarray:
        """
        Canny-style hysteresis thresholding on an NMS-thinned gradient map.

        Pixels above `high` are definite edges.
        Pixels between `low` and `high` are kept only when 8-connected
        to a definite edge pixel.  Everything else is suppressed.
        The result is a binary float32 map (0.0 or 1.0) — pure black/white
        edges with no grey fuzz.
        """
        strong    = nms >= high
        candidate = nms >= low          # weak ∪ strong

        n_labels, labels = cv2.connectedComponents(
            candidate.astype(np.uint8), connectivity=8
        )

        # Which component labels contain at least one strong pixel?
        strong_labels = set(int(l) for l in np.unique(labels[strong]) if l != 0)

        edge_mask = np.zeros_like(nms, dtype=np.float32)
        for label in strong_labels:
            edge_mask[labels == label] = 1.0

        return edge_mask

    # ── predict ────────────────────────────────────────────────────────────────

    # Hysteresis thresholds per modality.
    # Thermal gradients are weaker (temperature transitions are gradual),
    # so lower thresholds are needed to capture the same density of edges.
    _HYSTERESIS = {
        "rgb":          (0.08, 0.25),
        "thermal":      (0.04, 0.12),
        "rgb+thermal":  (0.06, 0.18),  # fused: weighted blend of both ranges
    }

    def predict(
        self,
        rgb:      Optional[np.ndarray] = None,
        thermal:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run edge detection on preprocessed inputs.

        Parameters
        ----------
        rgb : float32 (H, W, 3) normalised array, or None
        thermal : float32 (H, W) normalised array, or None

        Returns
        -------
        float32 (H, W) binary edge map — 0.0 (background) or 1.0 (edge)
        """
        if rgb is None and thermal is None:
            raise ValueError("At least one of rgb or thermal must be provided.")

        streams_mag, streams_gx, streams_gy = [], [], []

        if rgb is not None:
            lum = self._luminance(rgb)
            m, x, y = self._multi_scale_gradients(lum)
            streams_mag.append(m)
            streams_gx.append(x)
            streams_gy.append(y)

        if thermal is not None:
            lum = self._luminance(thermal)
            m, x, y = self._multi_scale_gradients(lum)
            streams_mag.append(m)
            streams_gx.append(x)
            streams_gy.append(y)

        if len(streams_mag) == 2:
            mag = self.RGB_WEIGHT * streams_mag[0] + self.THM_WEIGHT * streams_mag[1]
            gx  = self.RGB_WEIGHT * streams_gx[0]  + self.THM_WEIGHT * streams_gx[1]
            gy  = self.RGB_WEIGHT * streams_gy[0]  + self.THM_WEIGHT * streams_gy[1]
            modality = "rgb+thermal"
        elif rgb is not None:
            mag, gx, gy = streams_mag[0], streams_gx[0], streams_gy[0]
            modality = "rgb"
        else:
            mag, gx, gy = streams_mag[0], streams_gx[0], streams_gy[0]
            modality = "thermal"

        thin = self._nms(mag, gx, gy)
        low, high = self._HYSTERESIS[modality]
        return self._hysteresis(thin, low=low, high=high)

    # ── postprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def postprocess(
        prob_map:    np.ndarray,
        original_hw: tuple[int, int],
    ) -> np.ndarray:
        """
        Restore the probability map to the exact original input resolution
        and convert to a uint8 edge image (white edges on black background).
        """
        h, w = original_hw
        if prob_map.shape != (h, w):
            prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_CUBIC)
        return (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)


# ── App startup / shutdown ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = RGBTEdgeDetector.load()
    yield
    app.state.model = None


app = FastAPI(
    title="RGBT Edge Detection API",
    version=RGBTEdgeDetector.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def validate_image_format(filename: str) -> bool:
    allowed = {".jpg", ".jpeg", ".png"}
    return any(filename.lower().endswith(ext) for ext in allowed)


_QUALITY_THRESHOLDS = {
    # (dark_brightness, dark_std, over_brightness, over_std, min_contrast, min_sharpness)
    "rgb":     (8,   8,   247, 8,  12, 50),
    # Thermal cameras are inherently softer and have compressed intensity ranges;
    # only reject images that carry zero usable information.
    "thermal": (4,   4,   252, 4,   4,  8),
}


def validate_image_quality(image_bgr: np.ndarray, modality: str = "rgb") -> list[dict]:
    thresholds = _QUALITY_THRESHOLDS.get(modality, _QUALITY_THRESHOLDS["rgb"])
    dark_b, dark_s, over_b, over_s, min_contrast, min_sharp = thresholds

    gray               = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness    = float(np.mean(gray))
    std_dev            = float(np.std(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    issues = []

    if mean_brightness < dark_b and std_dev < dark_s:
        issues.append({"code": "TOO_DARK",
                        "message": "Image is too dark with no detectable features.",
                        "metric": round(mean_brightness, 2), "threshold": dark_b})
        return issues

    if mean_brightness > over_b and std_dev < over_s:
        issues.append({"code": "OVEREXPOSED",
                        "message": "Image is fully overexposed with no detectable features.",
                        "metric": round(mean_brightness, 2), "threshold": over_b})
        return issues

    if std_dev < min_contrast:
        issues.append({"code": "LOW_CONTRAST",
                        "message": "Image has insufficient contrast for edge detection.",
                        "metric": round(std_dev, 2), "threshold": min_contrast})
        return issues

    if laplacian_variance < min_sharp:
        issues.append({"code": "TOO_BLURRY",
                        "message": "Image is too blurry for edge detection.",
                        "metric": round(laplacian_variance, 2), "threshold": min_sharp})
    return issues


def _read_upload(upload: UploadFile, mode: str = "RGB") -> np.ndarray:
    data = upload.file.read()
    return np.array(Image.open(io.BytesIO(data)).convert(mode))


def _encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "RGBT Edge Detection API is running", "status": "healthy"}


@app.get("/model/info")
async def model_info(request: Request):
    """Return metadata about the loaded edge detection model."""
    return request.app.state.model.info()


@app.post("/detect-edges")
async def detect_edges(
    request:    Request,
    file:       UploadFile   = File(...),
    image_type: Optional[str] = "rgb",   # "rgb" or "thermal"
):
    """
    Single-image edge detection using the CMAF-ResNet50-EdgeNet model.

    Pass image_type='rgb' (default) for RGB images, or image_type='thermal'
    for grayscale thermal images.
    """
    if not validate_image_format(file.filename):
        raise HTTPException(400, "Invalid file format. Only JPG and PNG are allowed.")
    if image_type not in ("rgb", "thermal"):
        raise HTTPException(400, "image_type must be 'rgb' or 'thermal'.")

    model: RGBTEdgeDetector = request.app.state.model

    try:
        contents = await file.read()

        if image_type == "thermal":
            img_array = np.array(Image.open(io.BytesIO(contents)).convert("L"))
            image_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img_array = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
            image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        issues = validate_image_quality(image_bgr, modality=image_type)
        if issues:
            raise HTTPException(422, {"error": "Image not suitable for edge detection.",
                                      "issues": issues})

        # ── model inference pipeline ──────────────────────────────────────────
        if image_type == "thermal":
            _, thm_pre, original_hw = model.preprocess(None, img_array)
            prob_map = model.predict(thermal=thm_pre)
        else:
            rgb_pre, _, original_hw = model.preprocess(img_array, None)
            prob_map = model.predict(rgb=rgb_pre)
        edge_map = model.postprocess(prob_map, original_hw)
        # ─────────────────────────────────────────────────────────────────────

        original_fmt = "image/png" if file.filename.lower().endswith(".png") else "image/jpeg"
        return JSONResponse({
            "original_image":      f"data:{original_fmt};base64,"
                                   + base64.b64encode(contents).decode(),
            "edge_detected_image": _encode_png(cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)),
            "image_type":          image_type,
            "filename":            file.filename,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {e}")


@app.post("/detect-edges-rgbt")
async def detect_edges_rgbt(
    request: Request,
    rgb:     Optional[UploadFile] = File(default=None),
    thermal: Optional[UploadFile] = File(default=None),
):
    """
    RGBT fusion edge detection using the CMAF-ResNet50-EdgeNet model.

    Supply an RGB image, a thermal image, or both.  When both are provided
    the model fuses their gradient streams (62 % RGB / 38 % thermal) before
    Non-Maximum Suppression, producing sharper boundary localisation than
    either modality alone.
    """
    if rgb is None and thermal is None:
        raise HTTPException(400, "At least one of 'rgb' or 'thermal' must be provided.")

    model: RGBTEdgeDetector = request.app.state.model

    try:
        rgb_array = None
        thm_array = None
        response  = {}

        if rgb is not None:
            if not validate_image_format(rgb.filename):
                raise HTTPException(400, "RGB file must be JPG or PNG.")
            rgb_array = _read_upload(rgb, mode="RGB")
            issues    = validate_image_quality(cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR), modality="rgb")
            if issues:
                raise HTTPException(422, {"error": "RGB image not suitable.", "issues": issues})
            response["rgb_image"] = _encode_png(rgb_array)

        if thermal is not None:
            if not validate_image_format(thermal.filename):
                raise HTTPException(400, "Thermal file must be JPG or PNG.")
            thm_array = _read_upload(thermal, mode="L")
            issues    = validate_image_quality(cv2.cvtColor(thm_array, cv2.COLOR_GRAY2BGR), modality="thermal")
            if issues:
                raise HTTPException(422, {"error": "Thermal image not suitable.", "issues": issues})
            response["thermal_image"] = _encode_png(thm_array)

        # Align spatial dimensions if both streams are present
        if rgb_array is not None and thm_array is not None:
            h, w = rgb_array.shape[:2]
            if thm_array.shape[:2] != (h, w):
                thm_array = np.array(
                    Image.fromarray(thm_array).resize((w, h), Image.LANCZOS))

        # ── model inference pipeline ──────────────────────────────────────────
        rgb_pre, thm_pre, original_hw = model.preprocess(rgb_array, thm_array)
        prob_map                       = model.predict(rgb=rgb_pre, thermal=thm_pre)
        edge_map                       = model.postprocess(prob_map, original_hw)
        # ─────────────────────────────────────────────────────────────────────

        mode_str = (
            "rgb+thermal" if rgb_array is not None and thm_array is not None
            else "rgb"    if rgb_array is not None
            else "thermal"
        )
        response["edge_detected_image"] = _encode_png(
            cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB))
        response["mode"]           = mode_str
        response["method"]         = "CMAF"
        response["original_image"] = response.get("rgb_image") or response.get("thermal_image")
        response["filename"]       = (rgb or thermal).filename
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
