"""
main.py — FastAPI Backend for Aabha (Accessible AI Vision Assistant)

Endpoints:
  POST /api/caption   — Upload image, get OURS vs CAPABLE comparison + audio
  POST /api/detect    — Send camera frame (base64), get comparison + audio (fast path)
  POST /api/chat      — Ask a question about an image, get answer + audio
  GET  /api/models    — Report which models are loaded
  GET  /api/health    — Health check

Models:
  - OURS:    Custom ViT (CIFAR-10, from scratch) + BLIP (captioning + VQA)
  - CAPABLE: Florence-2 (open-vocabulary detection + detailed captioning)
  Both pipelines run concurrently per request (see analyze_comparison).
"""

import asyncio
import base64
import logging
import os
import io
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from vision_model import load_models, generate_answer, analyze_comparison
import vision_model
import capable_model
from tts import caption_to_audio
from translate import translate_caption

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("aabha")

# In-process rate limiter keyed on client IP (protects the CPU-heavy endpoints)
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# App lifespan: load the models once when the server starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the captioning, classification, and capable models at startup."""
    load_models()
    capable_model.load_capable_model()
    yield


# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Aabha - Accessible AI Vision Assistant API",
    description="Image captioning, object classification, and VQA for visually impaired users",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend (localhost:3000 for dev, any origin for deploy)
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.limiter = limiter


# ---------------------------------------------------------------------------
# Global exception handler — surface 500 tracebacks instead of swallowing them
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s:\n%s", request.method, request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MAX_ANALYSIS_DIM = int(os.getenv("MAX_ANALYSIS_DIM", "1024"))


def _cap_image_dimensions(image: Image.Image) -> Image.Image:
    """Downscale oversized images so inference latency/memory stay bounded.

    Real-world camera photos are often 3000px+, which bloats inference time
    without improving detections. Cap the longest edge to MAX_ANALYSIS_DIM.
    """
    longest = max(image.width, image.height)
    if longest <= MAX_ANALYSIS_DIM:
        return image
    scale = MAX_ANALYSIS_DIM / longest
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


async def _build_audio_payload(text: str, lang: str) -> dict:
    """Translate (if needed), synthesize speech, and build a per-panel audio payload."""
    translated = None
    if lang in ["hi", "mr"]:
        try:
            translated = translate_caption(text, target_lang=lang)
        except RuntimeError as e:
            translated = None
            logger.warning("Translation failed, falling back to English: %s", e)

    final_text = translated if translated else text
    tts_lang = lang if (lang in ["hi", "mr"] and translated) else "en"
    audio_bytes = await caption_to_audio(final_text, lang=tts_lang)
    return {
        "text_for_tts": final_text,
        "translated": translated,
        "lang": tts_lang,
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "audio_format": "mp3",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def models_status():
    """Report which models are loaded and ready."""
    return {
        "ours": vision_model.get_status(),
        "capable": capable_model.get_status(),
    }


@app.get("/api/health")
async def health_check():
    """Health check — returns 200 if the server and models are ready."""
    return {
        "status": "healthy",
        "models": {
            "blip": "Salesforce/blip-image-captioning-base",
            "custom_vit": "ViT-Tiny (CIFAR-10, trained from scratch)",
            "capable": capable_model.get_status()["model"],
        }
    }


@app.post("/api/caption")
@limiter.limit("10/minute")
async def create_caption(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPG, PNG, or WEBP, max 10MB)"),
    lang: str = Form(default="en", description="Target language: 'en', 'hi', or 'mr'"),
    model_choice: str = Form(default="combined", description="Model choice: 'combined', 'vit', or 'blip'"),
):
    """
    Upload an image and receive:
    - BLIP caption (scene description)
    - Custom ViT classification (object detection)
    - TTS audio narration
    """

    # --- Validate file extension ---
    if file.filename:
        extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{extension}'. Please upload a JPG, PNG, or WEBP image.",
            )

    # --- Read and validate file size ---
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({len(file_bytes) / (1024*1024):.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # --- Open the image (downscale very large images to bound latency/memory) ---
    try:
        image = _cap_image_dimensions(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the file as an image. It may be corrupted or not a valid image format.",
        )

    # --- Generate OURS vs CAPABLE comparison (runs concurrently) ---
    try:
        comparison = await analyze_comparison(image, model_choice=model_choice)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scene analysis failed: {str(e)}",
        )

    ours = comparison["ours"]
    capable = comparison["capable"]

    # --- Build per-panel TTS audio (translated per language) concurrently ---
    try:
        ours_audio_task = asyncio.create_task(_build_audio_payload(ours["description"], lang))
        capable_audio_task = (
            asyncio.create_task(_build_audio_payload(capable.get("caption") or "An image.", lang))
            if capable["available"]
            else None
        )
        ours_audio = await ours_audio_task
        capable_audio = await capable_audio_task if capable_audio_task else None
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

    # --- Return the comparison response ---
    return JSONResponse(
        content={
            "ours": {
                "caption_en": ours["caption"],
                "description": ours["description"],
                "classification": ours["classification"],
                "model_choice": ours["model_choice"],
                **ours_audio,
            },
            "capable": {
                "available": capable["available"],
                "caption": capable.get("caption") or "",
                "detections": capable.get("detections") or [],
                "error": capable.get("error"),
                **(capable_audio if capable_audio else {}),
            },
        }
    )


@app.post("/api/detect")
@limiter.limit("30/minute")
async def detect_from_frame(
    request: Request,
    frame: str = Form(..., description="Base64-encoded camera frame (JPEG/PNG)"),
    lang: str = Form(default="en", description="Target language: 'en', 'hi', or 'mr'"),
    model_choice: str = Form(default="combined", description="Model choice: 'combined', 'vit', or 'blip'"),
):
    """
    Detect objects from a live camera frame.

    Accepts a base64-encoded image frame (from webcam capture).
    Returns classification + caption + audio for blind user narration.
    Optimized for speed in the live camera loop.
    """

    # --- Decode the base64 frame ---
    try:
        # Handle data URL format: "data:image/jpeg;base64,/9j/4AAQ..."
        if "," in frame:
            frame = frame.split(",", 1)[1]
        frame_bytes = base64.b64decode(frame)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64-encoded frame.",
        )

    if len(frame_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty frame received.")

    if len(frame_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Frame is too large.")

    # --- Open the image (downscale very large frames to bound latency/memory) ---
    try:
        image = _cap_image_dimensions(Image.open(io.BytesIO(frame_bytes)).convert("RGB"))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the frame as an image.",
        )

    # --- Generate OURS vs CAPABLE comparison (runs concurrently) ---
    try:
        comparison = await analyze_comparison(image, model_choice=model_choice)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}",
        )

    ours = comparison["ours"]
    capable = comparison["capable"]

    # --- Build per-panel TTS audio concurrently ---
    try:
        ours_audio_task = asyncio.create_task(_build_audio_payload(ours["description"], lang))
        capable_audio_task = (
            asyncio.create_task(_build_audio_payload(capable.get("caption") or "An image.", lang))
            if capable["available"]
            else None
        )
        ours_audio = await ours_audio_task
        capable_audio = await capable_audio_task if capable_audio_task else None
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

    return JSONResponse(
        content={
            "ours": {
                "caption_en": ours["caption"],
                "description": ours["description"],
                "classification": ours["classification"],
                "model_choice": ours["model_choice"],
                **ours_audio,
            },
            "capable": {
                "available": capable["available"],
                "caption": capable.get("caption") or "",
                "detections": capable.get("detections") or [],
                "error": capable.get("error"),
                **(capable_audio if capable_audio else {}),
            },
        }
    )


@app.post("/api/chat")
@limiter.limit("20/minute")
async def chat_image(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPG, PNG, or WEBP, max 10MB)"),
    question: str = Form(..., description="User question about the image"),
    lang: str = Form(default="en", description="Target language: 'en', 'hi', or 'mr'"),
):
    """
    Ask a question about an image and receive a translated answer + spoken audio.
    """
    # --- Validate file extension ---
    if file.filename:
        extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{extension}'. Please upload a JPG, PNG, or WEBP image.",
            )

    # --- Read and validate file size ---
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # --- Open the image ---
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the file as an image.",
        )

    # --- Generate answer (in background thread so event loop stays responsive) ---
    try:
        answer_en = await asyncio.to_thread(generate_answer, image, question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("VQA generation error: %s", traceback.format_exc())
        answer_en = "I could not analyze this image for that question."

    if not answer_en or not answer_en.strip():
        answer_en = "I could not find a clear answer to that in this image."

    # --- Translate to Hindi or Marathi if requested ---
    answer_translated = None
    if lang in ["hi", "mr"]:
        try:
            answer_translated = translate_caption(answer_en, target_lang=lang)
        except RuntimeError as e:
            answer_translated = None
            logger.warning("Translation failed, falling back to English: %s", e)

    # --- Determine the final answer for TTS ---
    final_answer = answer_translated if answer_translated else answer_en
    tts_lang = lang if (lang in ["hi", "mr"] and answer_translated) else "en"

    # --- Generate audio (gracefully fallback if TTS fails) ---
    audio_base64 = ""
    try:
        audio_bytes = await caption_to_audio(final_answer, lang=tts_lang)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.warning("Audio generation failed for chat: %s", e)

    # --- Return response ---
    return JSONResponse(
        content={
            "answer_en": answer_en,
            "answer_translated": answer_translated,
            "lang": tts_lang,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
        }
    )


# ---------------------------------------------------------------------------
# Serve Static Frontend Files (Production Build)
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

# Resolve the absolute path to the frontend build folder (out/)
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "out"
)

# Mount the Next.js static build if it exists
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
