"""
main.py — FastAPI Backend for Aabha (Accessible AI Vision Assistant)

Endpoints:
  POST /api/caption   — Upload image, get caption + classification + audio
  POST /api/detect    — Send camera frame (base64), get detection + audio (fast path)
  POST /api/chat      — Ask a question about an image, get answer + audio
  GET  /api/health    — Health check

Models:
  - BLIP (Salesforce/blip-image-captioning-base) for captioning + VQA
  - Custom ViT (trained from scratch on CIFAR-10) for classification
"""

import base64
import os
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from vision_model import load_models, generate_caption, classify_image, generate_answer, describe_scene
from tts import caption_to_audio
from translate import translate_caption


# ---------------------------------------------------------------------------
# App lifespan: load the models once when the server starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the captioning and classification models at startup."""
    load_models()
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check — returns 200 if the server and models are ready."""
    return {
        "status": "healthy",
        "models": {
            "blip": "Salesforce/blip-image-captioning-base",
            "custom_vit": "ViT-Tiny (CIFAR-10, trained from scratch)",
        }
    }


@app.post("/api/caption")
async def create_caption(
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

    # --- Open the image ---
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the file as an image. It may be corrupted or not a valid image format.",
        )

    # --- Generate scene description (BLIP caption + ViT classification) ---
    try:
        scene = describe_scene(image, model_choice=model_choice)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scene analysis failed: {str(e)}",
        )

    # --- Translate if Hindi or Marathi is requested ---
    caption_translated = None
    description_translated = None
    if lang in ["hi", "mr"]:
        try:
            caption_translated = translate_caption(scene["caption"], target_lang=lang)
            description_translated = translate_caption(scene["description"], target_lang=lang)
        except RuntimeError as e:
            caption_translated = None
            description_translated = None
            print(f"Translation warning: {e}")

    # --- Determine the final text for TTS ---
    final_text = description_translated if description_translated else scene["description"]
    tts_lang = lang if (lang in ["hi", "mr"] and description_translated) else "en"

    # --- Generate audio ---
    try:
        audio_bytes = await caption_to_audio(final_text, lang=tts_lang)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

    # --- Return the response ---
    return JSONResponse(
        content={
            "caption_en": scene["caption"],
            "description": scene["description"],
            "classification": scene["classification"],
            "caption_translated": caption_translated,
            "lang": tts_lang,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
        }
    )


@app.post("/api/detect")
async def detect_from_frame(
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

    # --- Open the image ---
    try:
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the frame as an image.",
        )

    # --- Generate scene description ---
    try:
        scene = describe_scene(image, model_choice=model_choice)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}",
        )

    # --- Translate if needed ---
    description_for_tts = scene["description"]
    tts_lang = "en"
    if lang in ["hi", "mr"]:
        try:
            description_for_tts = translate_caption(scene["description"], target_lang=lang)
            tts_lang = lang
        except RuntimeError:
            pass  # Fall back to English

    # --- Generate audio ---
    try:
        audio_bytes = await caption_to_audio(description_for_tts, lang=tts_lang)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

    return JSONResponse(
        content={
            "caption": scene["caption"],
            "classification": scene["classification"],
            "description": scene["description"],
            "lang": tts_lang,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
        }
    )


@app.post("/api/chat")
async def chat_image(
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

    # --- Generate answer ---
    try:
        answer_en = generate_answer(image, question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visual question answering failed: {str(e)}",
        )

    # --- Translate to Hindi or Marathi if requested ---
    answer_translated = None
    if lang in ["hi", "mr"]:
        try:
            answer_translated = translate_caption(answer_en, target_lang=lang)
        except RuntimeError as e:
            answer_translated = None
            print(f"Translation warning: {e}")

    # --- Determine the final answer for TTS ---
    final_answer = answer_translated if answer_translated else answer_en
    tts_lang = lang if (lang in ["hi", "mr"] and answer_translated) else "en"

    # --- Generate audio ---
    try:
        audio_bytes = await caption_to_audio(final_answer, lang=tts_lang)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

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
