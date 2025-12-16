# ad_inference.py
"""
Audience + ad-style inference, based on Alejandro's work.

- Detect main face with YOLO-face.
- Estimate age and gender with CNNs.
- Classify clothing style with Xception.
- Map to age_segment, gender_segment, product_segment (in EN).
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import time

import numpy as np
import cv2
from PIL import Image

# ------------------------------------------------------------------
# 1. Model paths (ADJUST TO YOUR PC IF NEEDED)
# ------------------------------------------------------------------

BASE_MODELS_DIR = Path(r"E:\uni\Proyecto_jf\modelos_publicidad")

AGE_MODEL_PATH = BASE_MODELS_DIR / "age_cnn.h5"
GENDER_MODEL_PATH = BASE_MODELS_DIR / "gender_cnn.h5"
STYLE_MODEL_PATH = BASE_MODELS_DIR / "xception_7styles_256_256_3_linear_20_classbalanced.h5"
FACE_YOLO_PATH = BASE_MODELS_DIR / "yolov12n-face.pt"


# ------------------------------------------------------------------
# 2. Dataclasses for results
# ------------------------------------------------------------------

@dataclass
class AudiencePrediction:
    age: float          # estimated age (years)
    gender: str         # 'male' or 'female'
    style: str | None   # clothing style label (e.g. 'formal', 'sport', etc.)


@dataclass
class AdSegments:
    age_segment: str        # e.g. "Adults"
    gender_segment: str     # e.g. "Women"
    product_segment: str    # e.g. "casual clothing"


# ------------------------------------------------------------------
# 3. Lazy-loaded models (do NOT load on import)
# ------------------------------------------------------------------

print("[ADS] ad_inference module imported (models not loaded yet).")

_face_model = None
_age_model = None
_gender_model = None
_style_model = None
_xception_preprocess = None  # will hold keras.applications.xception.preprocess_input


def ensure_models_loaded():
    """
    Lazily load YOLO-face + Keras models the first time we need them.
    This avoids heavy work at module import time.
    """
    global _face_model, _age_model, _gender_model, _style_model, _xception_preprocess
    if _face_model is not None:
        # Already loaded
        return

    print("[ADS] Loading models for ad inference (YOLO + Keras)...")
    t0 = time.time()

    # Heavy imports kept inside this function
    from ultralytics import YOLO
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

    _xception_preprocess = xception_preprocess

    # Load models from disk
    _face_model = YOLO(str(FACE_YOLO_PATH))
    _age_model = load_model(str(AGE_MODEL_PATH))
    _gender_model = load_model(str(GENDER_MODEL_PATH))
    _style_model = load_model(str(STYLE_MODEL_PATH))

    dt = time.time() - t0
    print(f"[ADS] Models loaded successfully in {dt:.1f} s.")


# ------------------------------------------------------------------
# 4. Preprocessing utilities
# ------------------------------------------------------------------

def _extract_main_face(image_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect main face with YOLO-face and return a BGR crop.
    If no face is found, return None.
    """
    results = _face_model(image_bgr)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    # largest bounding box = main face
    max_area = 0.0
    best_box = None
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = (int(x1), int(y1), int(x2), int(y2))

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    face = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if face.size == 0:
        return None

    return face


def _predict_age_gender(face_bgr: np.ndarray) -> tuple[float, str]:
    """
    Apply age_cnn and gender_cnn to a cropped face.
    Return (approx_age, 'male'/'female').
    """
    # to RGB and resize to 200x200 as in training
    im = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    im = im.resize((200, 200), Image.Resampling.LANCZOS)

    ar = np.asarray(im).astype("float32") / 255.0
    ar = ar.reshape(-1, 200, 200, 3)

    age_pred = _age_model.predict(ar, verbose=0)
    gender_raw = _gender_model.predict(ar, verbose=0)

    age_value = float(age_pred.ravel()[0])

    gender_bin = int(np.round(gender_raw.ravel()[0]))
    gender = "male" if gender_bin == 0 else "female"

    return age_value, gender


def _predict_style(full_image_bgr: np.ndarray) -> str | None:
    """
    Classify clothing style with the Xception model.
    We use the full image (not just the face).
    """
    assert _xception_preprocess is not None

    img_rgb = cv2.cvtColor(full_image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (256, 256))
    img = img.astype("float32")
    x = _xception_preprocess(img)
    x = np.expand_dims(x, axis=0)

    preds = _style_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))

    # IMPORTANT: this list must match the training order
    style_labels = [
        "Creative", "Dramatic", "Elegant", "Magnetic", "Natural", "Romantic", "Traditional"
    ]

    if idx < 0 or idx >= len(style_labels):
        return None

    return style_labels[idx]


# ------------------------------------------------------------------
# 5. Mapping into ad segments (EN)
# ------------------------------------------------------------------

def _map_age_to_segment(age: float) -> str:
    """
    Map raw age (years) to an age_segment in EN, consistent with CLIP and ads table.
    """
    if age < 12:
        return "Young children"
    elif age < 18:
        return "Teenagers"
    elif age < 60:
        return "Adults"
    else:
        return "Older adults"


def _map_gender_to_segment(gender: str) -> str:
    """
    Map 'male' / 'female' to gender_segment in EN.
    """
    if gender == "male":
        return "Men"
    elif gender == "female":
        return "Women"
    else:
        return "Neutral"


def _map_style_to_product_segment(style: str | None) -> str:
    """
    Map clothing style to a product_segment from our EN catalog.
    Very simplified, demo-only.
    """
    if style is None:
        return "casual clothing"

    s = style.lower()
    if "sport" in s:
        return "sportswear"
    if "formal" in s or "elegant" in s:
        return "personal care"
    if "street" in s:
        return "casual clothing"
    if "uniform" in s:
        return "furniture and decoration"
    if "home" in s:
        return "home cleaning"

    return "casual clothing"


# ------------------------------------------------------------------
# 6. Public API
# ------------------------------------------------------------------

def analyze_image_for_ads(image_path: str | Path) -> tuple[AudiencePrediction, AdSegments]:
    """
    Given an image path, return:
      - AudiencePrediction (age, gender, style)
      - AdSegments with age_segment, gender_segment, product_segment (EN)
    """
    # ✅ Only here we load heavy models once
    ensure_models_loaded()

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # 1) detect main face
    face = _extract_main_face(img_bgr)
    if face is None:
        # fallback: use full image if no face found
        face = img_bgr.copy()

    # 2) age & gender from face
    age_value, gender = _predict_age_gender(face)

    # 3) style from full image
    style_label = _predict_style(img_bgr)

    audience = AudiencePrediction(age=age_value, gender=gender, style=style_label)

    # 4) convert to ad segments (EN)
    age_segment = _map_age_to_segment(age_value)
    gender_segment = _map_gender_to_segment(gender)
    product_segment = _map_style_to_product_segment(style_label)

    segments = AdSegments(
        age_segment=age_segment,
        gender_segment=gender_segment,
        product_segment=product_segment,
    )

    return audience, segments


# Small manual test
if __name__ == "__main__":
    test_image = r"E:\uni\Proyecto_jf\camaras\img_conteo_prueba_1.jpg"  # ADJUST IF NEEDED
    t0 = time.time()
    audience, segs = analyze_image_for_ads(test_image)
    print("=== Audience ===")
    print(audience)
    print("=== Segments ===")
    print(segs)
    print(f"Total time: {time.time() - t0:.1f} s")
