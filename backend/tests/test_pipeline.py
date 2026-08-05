"""
test_pipeline.py — Unit tests for the Aabha backend (hybrid comparison).

Run with: python -m pytest tests/ -v
"""

import io
import asyncio
import pytest
from PIL import Image

from vision_model import (
    load_models,
    generate_caption,
    classify_image,
    generate_answer,
    analyze_comparison,
)
from tts import caption_to_audio
from translate import translate_caption


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Load the captioning models once for all tests."""
    load_models()


@pytest.fixture
def sample_image():
    """Create a simple test image (solid blue, 100x100)."""
    return Image.new("RGB", (100, 100), color=(70, 130, 180))


@pytest.fixture
def sample_rgba_image():
    """Create an RGBA test image to verify mode conversion."""
    return Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))


# ---------------------------------------------------------------------------
# Caption / Classification / VQA
# ---------------------------------------------------------------------------
class TestVisionModel:
    def test_generates_nonempty_caption(self, sample_image):
        caption = generate_caption(sample_image)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_handles_rgba_image(self, sample_rgba_image):
        caption = generate_caption(sample_rgba_image)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_classify_returns_shape(self, sample_image):
        result = classify_image(sample_image)
        assert isinstance(result, dict)
        assert "class_name" in result
        assert "confidence" in result

    def test_answer_question(self, sample_image):
        answer = generate_answer(sample_image, "What color is this?")
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    def test_empty_question_raises(self, sample_image):
        with pytest.raises(ValueError):
            generate_answer(sample_image, "   ")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
class TestTTS:
    @pytest.mark.anyio
    async def test_returns_mp3_bytes(self):
        audio = await caption_to_audio("A dog running in a park.", lang="en")
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    @pytest.mark.anyio
    async def test_hindi_audio(self):
        audio = await caption_to_audio("एक कुत्ता पार्क में दौड़ रहा है।", lang="hi")
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    @pytest.mark.anyio
    async def test_empty_text_raises_error(self):
        with pytest.raises(ValueError, match="empty"):
            await caption_to_audio("")


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
class TestTranslation:
    def test_translates_to_hindi(self):
        result = translate_caption("A cat sitting on a mat.", target_lang="hi")
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "A cat sitting on a mat."

    def test_translates_to_marathi(self):
        result = translate_caption("A cat sitting on a mat.", target_lang="mr")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_text_raises_error(self):
        with pytest.raises(ValueError, match="empty"):
            translate_caption("")


# ---------------------------------------------------------------------------
# Hybrid Comparison pipeline
# ---------------------------------------------------------------------------
class TestComparison:
    @pytest.mark.anyio
    async def test_analyze_comparison_runs_ours(self, sample_image):
        """The comparison pipeline must always produce the OURS side."""
        comparison = await analyze_comparison(sample_image, model_choice="combined")
        assert "ours" in comparison
        assert "capable" in comparison
        assert comparison["ours"]["description"]
        assert comparison["capable"]["available"] in (True, False)