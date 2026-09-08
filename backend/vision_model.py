"""
vision_model.py — Unified Vision Model Loader (Custom ViT + BLIP)

Loads two models:
  1. Custom ViT (trained from scratch on CIFAR-10) — for object classification
  2. Salesforce BLIP — for image captioning and Visual Question Answering

This replaces the old multi_caption_model.py which loaded 3 separate models.

The "capable" comparison model (Florence-2) is loaded separately via
capable_model.py and orchestrated by analyze_comparison() — both pipelines run
concurrently so total latency tracks the slowest model, not the sum.
"""

import os
import sys
import asyncio
import time
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

import capable_model

# Add the vision_transformer_scratch directory to path
_VIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vision_transformer_scratch",
)
sys.path.insert(0, _VIT_DIR)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level globals for lazy/cached loading
_models_loaded = False

# BLIP
_blip_processor = None
_blip_model = None

# Custom ViT
_vit_model = None
_vit_available = False

# BLIP load status
_blip_available = False

# Dedicated VQA Model (Salesforce/blip-vqa-base)
_vqa_processor = None
_vqa_model = None
_vqa_available = False

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Default checkpoint path
_VIT_CHECKPOINT = os.path.join(_VIT_DIR, "checkpoints", "vit_cifar10_best.pth")


def load_models():
    """Load BLIP and the custom ViT model at startup."""
    global _models_loaded
    global _blip_processor, _blip_model
    global _vit_model, _vit_available
    global _blip_available

    if _models_loaded:
        print("Models already loaded in cache.")
        return

    print(f"Loading models onto device: {DEVICE}...")
    start_time = time.time()

    # ── Load BLIP (captioning + VQA) ──
    print("Loading BLIP (captioning + VQA)...")
    _blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    _blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True,
    ).to(DEVICE)
    _blip_model.eval()
    _blip_available = True
    print("  BLIP loaded successfully.")

    # ── Load Dedicated VQA Model (Salesforce/blip-vqa-base) ──
    try:
        load_vqa_model()
    except Exception as e:
        print(f"  Note: VQA model will load on-demand: {e}")

    # ── Load Custom ViT (classification) ──
    if os.path.exists(_VIT_CHECKPOINT):
        print(f"Loading custom ViT from: {_VIT_CHECKPOINT}")
        try:
            from vit import VisionTransformer
            _vit_model = VisionTransformer.load_trained(_VIT_CHECKPOINT)
            _vit_model.to(DEVICE)
            _vit_model.eval()
            _vit_available = True
            print("  Custom ViT loaded successfully.")
        except Exception as e:
            print(f"  Warning: Failed to load custom ViT: {e}")
            print("  Classification will not be available.")
            _vit_available = False
    else:
        print(f"  Custom ViT checkpoint not found at: {_VIT_CHECKPOINT}")
        print("  Run 'python vision_transformer_scratch/train.py' to train the model.")
        print("  Classification will not be available until training is complete.")
        _vit_available = False

    _models_loaded = True
    elapsed = time.time() - start_time
    print(f"\nAll models loaded in {elapsed:.2f} seconds.")
    print(f"  BLIP:       [OK] ready")
    print(f"  Custom ViT: {'[OK] ready' if _vit_available else '[X] not trained yet'}")


def generate_caption(image: Image.Image) -> str:
    """Generate a caption for the image using BLIP.

    Args:
        image: PIL Image to caption.

    Returns:
        English caption string.
    """
    if not _models_loaded:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize tiny images to avoid processor errors
    if image.width < 32 or image.height < 32:
        image = image.resize((224, 224), Image.Resampling.BILINEAR)

    inputs = _blip_processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = _blip_model.generate(**inputs, max_new_tokens=50)
    caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True)

    return caption.strip()


def classify_image(image: Image.Image) -> dict:
    """Classify the image using the custom ViT.

    Args:
        image: PIL Image to classify.

    Returns:
        Dict with 'class_name', 'confidence', 'available', and 'top3'.
        If the ViT is not trained yet, returns a placeholder response.
    """
    if not _models_loaded:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    if not _vit_available:
        return {
            "class_name": "unknown",
            "confidence": 0.0,
            "available": False,
            "top3": [],
            "message": "Custom ViT not trained yet. Run train.py first.",
        }

    if image.mode != "RGB":
        image = image.convert("RGB")

    result = _vit_model.predict(image)

    return {
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "available": True,
        "top3": result["all_probabilities"][:3],
    }


def load_vqa_model():
    """Load Salesforce/blip-vqa-base for true Visual Question Answering."""
    global _vqa_processor, _vqa_model, _vqa_available
    if _vqa_available and _vqa_model is not None:
        return _vqa_processor, _vqa_model
    try:
        print("[VQA] Loading dedicated Salesforce/blip-vqa-base...")
        _vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        _vqa_model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base",
            use_safetensors=True,
        ).to(DEVICE)
        _vqa_model.eval()
        _vqa_available = True
        print("[VQA] Salesforce/blip-vqa-base loaded successfully.")
        return _vqa_processor, _vqa_model
    except Exception as e:
        print(f"[VQA] Note: Salesforce/blip-vqa-base not ready yet: {e}")
        _vqa_available = False
        return None, None


def generate_answer(image: Image.Image, question: str) -> str:
    """Answer a question about the image with true visual analysis.

    Args:
        image: PIL Image.
        question: Question about the image.

    Returns:
        Exact, concise answer string based on visual analysis.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    q = question.strip().lower()

    # 1. Primary: Dedicated VQA model (Salesforce/blip-vqa-base)
    # Specifically fine-tuned on visual question answering (VQA v2) to answer questions accurately
    vqa_proc, vqa_mdl = load_vqa_model()
    if vqa_mdl is not None and vqa_proc is not None:
        try:
            inputs = vqa_proc(image, question.strip(), return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output_ids = vqa_mdl.generate(**inputs, max_new_tokens=35)
            ans = vqa_proc.decode(output_ids[0], skip_special_tokens=True).strip()
            if ans and len(ans) > 0 and ans.lower() not in ["unanswerable"]:
                return ans[0].upper() + ans[1:] if len(ans) > 1 else ans.upper()
        except Exception as e:
            print(f"[VQA] Error with blip-vqa-base: {e}")

    # 2. Text/OCR Reading: If the question asks to read words, text, or letters on the image
    if any(w in q for w in ["read", "written", "text", "say", "word", "letter", "title", "heading", "name"]):
        try:
            ocr_text = capable_model.extract_text(image)
            if ocr_text:
                return f"The text reads: {ocr_text}"
        except Exception as e:
            print(f"[VQA] OCR extraction error: {e}")

    # 3. Deep visual inspection via Florence-2 scene analysis
    try:
        scene_desc = capable_model.generate_detailed_caption(image)
        if scene_desc:
            # If the user asks about color, extract the relevant color observation
            if "colour" in q or "color" in q:
                color_words = ["white", "black", "blue", "red", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "grey"]
                sentences = [s.strip() for s in scene_desc.split(".") if s.strip()]
                q_nouns = [w for w in q.replace("?", "").split() if len(w) > 2 and w not in ["what", "whats", "the", "colour", "color", "of", "is", "are", "this", "that"]]
                for s in sentences:
                    s_lower = s.lower()
                    if any(c in s_lower for c in color_words):
                        if any(n in s_lower for n in q_nouns):
                            return s + "."
                for s in sentences:
                    if any(c in s.lower() for c in color_words):
                        return s + "."
            return scene_desc
    except Exception as e:
        print(f"[VQA] Deep caption error: {e}")

    # 4. Fallback: Base image caption
    try:
        caption = generate_caption(image)
        if caption:
            return f"This image shows {caption}."
    except Exception:
        pass

    return "I analyzed the image, but could not determine a specific answer to that question."


def describe_scene(image: Image.Image, model_choice: str = "combined") -> dict:
    """Generate scene description based on selected model choice.

    Args:
        image: PIL Image.
        model_choice: Choice of model logic ('combined', 'vit', 'blip').

    Returns:
        Dict with 'caption', 'classification', 'description', and 'model_choice'.
    """
    caption = ""
    classification = {"class_name": "", "confidence": 0.0, "available": False, "top3": []}

    if model_choice in ["combined", "blip"]:
        caption = generate_caption(image)

    if model_choice in ["combined", "vit"]:
        classification = classify_image(image)

    if model_choice == "vit":
        if classification["available"] and classification["confidence"] > 0.1:
            class_name = classification["class_name"]
            conf_pct = int(classification["confidence"] * 100)
            description = f"Detected object: {class_name} with {conf_pct} percent confidence."
        elif not classification["available"]:
            description = "Custom ViT model is not trained yet."
        else:
            description = "Uncertain object detection."
    elif model_choice == "blip":
        description = caption + "." if caption else "An image."
    else:  # 'combined'
        if classification["available"] and classification["confidence"] > 0.3:
            class_name = classification["class_name"]
            confidence = classification["confidence"]
            if confidence > 0.7:
                description = f"I can see a {class_name}. {caption}."
            else:
                description = f"This might be a {class_name}. {caption}."
        else:
            description = caption + "." if caption else "An image."

    # Clean up double periods
    description = description.replace("..", ".")

    return {
        "caption": caption,
        "classification": classification,
        "description": description,
        "model_choice": model_choice,
    }


def get_status() -> dict:
    """Report availability of the OURS pipeline (BLIP + custom ViT)."""
    return {
        "blip": _blip_available,
        "custom_vit": _vit_available,
        "device": DEVICE,
    }


# =============================================================================
# Hybrid Comparison Pipeline (Ours vs Capable)
# =============================================================================

def _capable_summary(image: Image.Image) -> dict:
    """Run the capable model (Florence-2) caption + detection.

    Runs inside a worker thread; inference ops release the GIL so PyTorch
    work from both pipelines genuinely overlaps across CPU cores.
    """
    try:
        if not capable_model.get_status()["available"]:
            return {"available": False, "caption": "", "detections": [], "error": "Capable model not loaded."}

        caption = capable_model.generate_caption(image)
        detections = capable_model.detect_objects(image)
        return {
            "available": True,
            "caption": caption,
            "detections": detections,
            "error": None,
        }
    except Exception as e:
        return {"available": False, "caption": "", "detections": [], "error": str(e)}


async def analyze_comparison(image: Image.Image, model_choice: str = "combined") -> dict:
    """Analyze an image with BOTH pipelines concurrently.

    - "ours":    Custom ViT classification + BLIP caption (existing pipeline)
    - "capable": Florence-2 open-vocabulary detection + detailed caption

    Returns:
        {
            "ours":    <describe_scene result>,
            "capable": {"available", "caption", "detections", "error"},
        }
    """
    ours_task = asyncio.to_thread(describe_scene, image, model_choice)
    capable_task = asyncio.to_thread(_capable_summary, image)

    ours, capable = await asyncio.gather(ours_task, capable_task)
    return {"ours": ours, "capable": capable}


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING VISION MODEL MODULE")
    print("=" * 60)

    load_models()

    # Test with a dummy image
    dummy = Image.new("RGB", (224, 224), color="blue")

    print("\n--- Caption Test ---")
    caption = generate_caption(dummy)
    print(f"Caption: {caption}")

    print("\n--- Classification Test ---")
    result = classify_image(dummy)
    print(f"Classification: {result}")

    print("\n--- Scene Description Test ---")
    scene = describe_scene(dummy)
    print(f"Description: {scene['description']}")

    print("\n--- VQA Test ---")
    answer = generate_answer(dummy, "What color is this?")
    print(f"Answer: {answer}")
