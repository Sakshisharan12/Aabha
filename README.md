---
title: Aabha - Accessible AI Vision Assistant for Blind Users
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Aabha: Accessible AI Vision Assistant

Aabha is a web application designed to empower visually impaired individuals — especially blind children — by providing **real-time object detection** and **image-to-audio narration**. The application combines a **custom Vision Transformer (ViT) built from scratch** with a pre-trained BLIP model to detect objects, describe scenes, and speak the results aloud.

Aabha features a modern, fully accessible React **Next.js (App Router)** frontend and a high-performance **FastAPI** backend.

---

## 🌟 Key Features

1. **Custom ViT Object Classifier**: A Vision Transformer built from scratch (Patch Embedding → CLS Token → Positional Embedding → Multi-Head Self-Attention → MLP → Encoder Blocks → Classification Head), trained on CIFAR-10 for object recognition.
2. **AI Scene Captioning**: Uses the pre-trained `Salesforce/blip-image-captioning-base` for natural language scene descriptions.
3. **🆚 Hybrid Head-to-Head Comparison**: Every image is analyzed by **two pipelines running in parallel** and shown side-by-side in the UI:
   - **Ours** — the from-scratch custom ViT (single 10-class label) + BLIP caption.
   - **Capable** — `microsoft/Florence-2` for open-vocabulary object detection (with bounding boxes overlaid on the image) + detailed captioning.
4. **Live Camera Detection**: Real-time object detection from webcam with auto-speak — captures frames, runs both comparison pipelines, and narrates aloud automatically.
5. **Audio Narration**: Converts descriptions into natural-sounding speech using Microsoft Edge Neural TTS (`edge-tts`).
6. **Multilingual Support**: Supports English, Hindi, and Marathi (powered by `googletrans`).
7. **Visual Question Answering (VQA)**: Ask questions about uploaded images and receive spoken answers.
8. **WCAG-Compliant Frontend**: High contrast, screen-reader friendly (ARIA labels), fully keyboard navigable.

---

## 🧠 Custom Vision Transformer (Built From Scratch)

The `vision_transformer_scratch/` directory contains a complete ViT implementation built from the ground up — every layer is hand-coded with detailed mathematical documentation:

| File | Component | Description |
|------|-----------|-------------|
| `patch_embedding.py` | Patch Embedding | Splits image into patches, projects to embeddings |
| `positional_embedding.py` | Positional Embedding | Adds learnable position information to tokens |
| `cls_token.py` | CLS Token | Prepends learnable classification token |
| `attention.py` | Multi-Head Self-Attention | Q-K-V attention mechanism with multiple heads |
| `mlp.py` | Feed-Forward Network | Two-layer MLP with GELU activation |
| `encoder_block.py` | Encoder Block | Pre-Norm Transformer block (MHSA + MLP + Residuals) |
| `vit.py` | Full ViT Assembly | Connects all components into a complete classifier |
| `train.py` | Training Pipeline | CIFAR-10 training with AdamW + cosine annealing |

**ViT-Tiny Configuration**: 32×32 input, patch_size=4, embed_dim=192, 3 heads, 6 encoder blocks, ~1.2M parameters.

---

## 🛠️ Tech Stack

- **Frontend**: Next.js (App Router), React, Vanilla CSS (Glassmorphism theme), Lucide Icons.
- **Backend API**: FastAPI (Python 3.10+).
- **Custom Model**: Vision Transformer built from scratch with PyTorch.
- **Pre-trained Models**: Hugging Face Transformers — `BLIP` (captioning + VQA) and `Florence-2` (open-vocabulary detection + detailed captioning).
- **Text-to-Speech**: `edge-tts` (Microsoft Edge Neural TTS).
- **Translation**: `googletrans` (free Google Translate wrapper).

---

## 📂 Project Directory Structure

```text
Aabha/
├── vision_transformer_scratch/     # Custom ViT (built from scratch)
│   ├── patch_embedding.py          # Image → patch embeddings
│   ├── positional_embedding.py     # Learnable positional encoding
│   ├── cls_token.py                # [CLS] classification token
│   ├── attention.py                # Multi-Head Self-Attention
│   ├── mlp.py                      # Feed-Forward Network
│   ├── encoder_block.py            # Transformer Encoder Block
│   ├── vit.py                      # Full ViT model assembly
│   ├── train.py                    # CIFAR-10 training pipeline
│   └── checkpoints/                # Saved model weights
│       └── vit_cifar10_best.pth
├── backend/
│   ├── vision_model.py             # Unified model loader (BLIP + custom ViT)
│   ├── capable_model.py            # Florence-2 loader (open-vocab detection + caption)
│   ├── main.py                     # FastAPI server & endpoints
│   ├── requirements.txt            # Python dependencies
│   ├── translate.py                # Google Translate API abstraction
│   └── tts.py                      # Text-To-Speech wrapper
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── layout.js           # Next.js app wrapper & fonts
│   │       ├── page.js             # React UI (Upload + Live Camera + Comparison)
│   │       └── globals.css         # Accessibility theme styles
│   ├── package.json
│   └── next.config.mjs
├── .github/workflows/ci.yml        # CI: backend import + frontend lint/build
├── Dockerfile
└── README.md
```

---

## 🚀 Setup & Installation (Local)

### 1. Clone the repository
```bash
git clone <repository-url>
cd Aabha
```

### 2. Train the Custom ViT (first time only)
```bash
cd vision_transformer_scratch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python train.py --epochs 30
```
This downloads CIFAR-10 (~170MB) and trains the ViT-Tiny model. Takes ~1-2 hours on CPU. The best checkpoint is saved to `checkpoints/vit_cifar10_best.pth`.

### 3. Run the Backend API
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python main.py
```
The backend API runs on **`http://localhost:8000`**.

### 4. Run the Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```
The frontend runs on **`http://localhost:3000`**.

---

## 🧪 Testing ViT Components

Each ViT component can be tested independently:
```bash
cd vision_transformer_scratch
python patch_embedding.py
python positional_embedding.py
python cls_token.py
python attention.py
python mlp.py
python encoder_block.py
python vit.py
```

---

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/models` | Report which models (ours + capable) are loaded |
| `POST` | `/api/caption` | Upload image → `{ours, capable}` comparison + per-panel audio |
| `POST` | `/api/detect` | Camera frame (base64) → `{ours, capable}` comparison + audio |
| `POST` | `/api/chat` | Image + question → VQA answer + audio |

### Example comparison response (`POST /api/caption`)

```json
{
  "ours": {
    "caption_en": "a cat sitting on a sofa",
    "description": "This might be a cat. a cat sitting on a sofa.",
    "classification": { "class_name": "cat", "confidence": 0.82, "available": true },
    "model_choice": "combined",
    "text_for_tts": "a cat sitting on a sofa",
    "translated": null,
    "lang": "en",
    "audio_base64": "<base64 mp3>",
    "audio_format": "mp3"
  },
  "capable": {
    "available": true,
    "caption": "A gray cat resting on a cream sofa in a bright living room.",
    "detections": [{ "label": "cat", "score": null, "box": [214, 96, 621, 470] }],
    "error": null,
    "text_for_tts": "A gray cat resting on a cream sofa in a bright living room.",
    "translated": null,
    "lang": "en",
    "audio_base64": "<base64 mp3>",
    "audio_format": "mp3"
  }
}
```

---

## ⚠️ Known Limitations

- **CIFAR-10 Classes Only**: The custom ViT ("ours") recognizes only 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). This is the point of the comparison — the capable Florence-2 model detects open-vocabulary objects.
- **CPU Inference**: Florence-2 runs without a KV-cache for compatibility (`use_cache=False`), so it is slower than a CUDA deployment. Expect a few seconds per image on CPU.
- **CPU Training**: Training is designed for CPU. GPU will be faster but is not required.
- **Text-Heavy Images**: BLIP does not perform OCR on documents.
- **Hallucinations**: Like all vision-language models, BLIP/Florence-2 may occasionally misidentify objects in complex scenes.
