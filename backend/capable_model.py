"""
capable_model.py — Modern Capable Vision Model (Florence-2) Loader

Loads microsoft/Florence-2 (a unified open-vocabulary vision-language model)
to provide the "capable" side of the hybrid head-to-head comparison against the
homegrown ViT-Tiny (10-class CIFAR-10 classifier).

Why Florence-2 on CPU?
    BLIP-2 / large vision-language models are impractical on CPU-only hardware
    (many seconds per frame). Florence-2-base (0.23B params) runs on CPU in
    ~2-4s per image while supporting BOTH:
        - <OD>          open-vocabulary object detection (labels + bounding boxes)
        - <CAPTION>     natural-language scene description
    This makes it the direct apples-to-apples rival to our ViT (detection)
    and BLIP (captioning).

Tasks exposed:
    detect_objects(image)  ->  [{label, score, box:[x1,y1,x2,y2]}]  (pixel coords)
    generate_caption(image)->  English caption string
"""

import os
import torch
from PIL import Image

# Optional DeviceVision settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Allow overriding the model id via env (e.g. Florence-2-large)
MASL_MODEL_ID = os.getenv(
    "CAPABLE_MODEL_ID", "microsoft/Florence-2-base"
)

# Generation constants (mirrors the official Florence-2 examples)
_GEN_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": False,
    "num_beams": 3,
    "num_return_sequences": 1,
}
# End-of-chunk tokens we strip from generated decoder output
_ENDING = "<|endofchunk|>"

# Module-level cache (lazy load)
_loaded = False
_processor = None
_model = None
_available = False
_load_error = None


def load_capable_model():
    """Load Florence-2 into memory. Safe to retry; no-ops if already loaded."""
    global _loaded, _processor, _model, _available, _load_error

    if _loaded:
        return

    from transformers import AutoModelForCausalLM, AutoProcessor

    _loaded = True
    try:
        print(f"[capable] Loading {MASL_MODEL_ID} onto device: {DEVICE}...")
        _processor = AutoProcessor.from_pretrained(MASL_MODEL_ID, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MASL_MODEL_ID, trust_remote_code=True
        ).to(DEVICE)

        # Florence-2 remote code predates the `_supports_sdpa` attribute that
        # newer transformers expects on PreTrainedModel. Add it if missing.
        try:
            if not hasattr(_model, "_supports_sdpa"):
                _model._supports_sdpa = False
        except Exception:
            pass

        _model.eval()
        _available = True
        print("[capable] Florence-2 loaded successfully.")
    except Exception as e:  # network / memory / missing weights
        _load_error = e
        _available = False
        print(f"[capable] WARNING: Failed to load Florence-2: {e}")
        print("[capable] Comparison will be unavailable (ours-only mode).")


def _require_model():
    if not _available or _processor is None or _model is None:
        raise RuntimeError("Capable model is not available. Call load_capable_model first.")
    return _processor, _model


def generate_caption(image: Image.Image) -> str:
    """Produce a natural-language scene description (DETAILED caption)."""
    proc, model = _require_model()
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = proc(text="<DETAILED_CAPTION>", images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    generated_text = proc.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(
        generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height)
    )

    parsed_text = parsed["<DETAILED_CAPTION>"] if isinstance(parsed, dict) else parsed
    # Strip any residual end-of-chunk tokens
    return str(parsed_text).replace(_ENDING, "").strip()


def detect_objects(image: Image.Image, max_objects: int = 20) -> list:
    """Run open-vocabulary object detection.

    Returns a list of dicts:
        {"label": str, "score": float|None, "box": [x1, y1, x2, y2] (pixels)}
    Boxes are resolved to pixel coordinates via the processor.
    `score` is left as None because constrained beam decoding does not expose a
    calibrated per-box probability; the frontend renders it as "--".
    """
    proc, model = _require_model()
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = proc(text="<OD>", images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    generated_text = proc.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(image.width, image.height),
    )

    detections = parsed.get("<OD>") or []
    # Normalize field names: HF returns {"label", "bbox"}
    out = []
    for det in detections[:max_objects]:
        label = det.get("label", "object")
        bbox = det.get("bbox", [0, 0, 0, 0])
        out.append({
            "label": label,
            "score": None,
            "box": [int(v) for v in bbox],
        })
    return out


def get_status() -> dict:
    """Return availability status of the capable model (for /api/models)."""
    return {"available": _available, "model": MASL_MODEL_ID, "device": DEVICE}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the Florence-2 capable model")
    parser.add_argument("--image", default=None, help="Path to test image")
    args = parser.parse_args()

    load_capable_model()
    print("Status:", get_status())

    img = None
    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("RGB")
    else:
        img = Image.new("RGB", (224, 224), color=(70, 130, 180))
        print("Using synthetic blue test image.")

    print("\n--- Caption ---")
    print(generate_caption(img))
    print("\n--- Detection ---")
    for d in detect_objects(img):
        print(d)