import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from pathlib import Path
from typing import Dict

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import time


# Local model path downloaded using huggingface-cli
MODEL_DIR = Path(r"E:\uni\Proyecto_jf\modelos_publicidad\clip-vit-base-patch32")

_clip_model = None
_clip_processor = None


def ensure_clip_loaded():
    """
    Loads the CLIP model from local disk if it hasn't been loaded yet.
    Called from apps.py and from _predict_category for safety.
    """
    global _clip_model, _clip_processor
    if _clip_model is not None and _clip_processor is not None:
        return

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"❌ Local model folder not found: {MODEL_DIR}")

    print(f"[CLIP] Starting LOCAL load from {MODEL_DIR}...", flush=True)
    start = time.perf_counter()

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # ⚠️ VERY IMPORTANT: load exclusively from local disk
    _clip_model = CLIPModel.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    mid = time.perf_counter()
    print(f"[CLIP] Model loaded in {mid - start:.1f}s", flush=True)

    _clip_processor = CLIPProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    end = time.perf_counter()
    print(f"[CLIP] Processor loaded in {end - mid:.1f}s (total {end - start:.1f}s)", flush=True)


# ============================
# Categories (PROMPTS) in Spanish
# ============================
AGE_CATEGORIES = [
    "Orientado a todo el público",
    "Orientado a niños pequeños",
    "Orientado a adolescentes",
    "Orientado a adultos",
    "Orientado a personas mayores",
    "Orientado a público familiar",
]

GENDER_CATEGORIES = [
    "Anuncio para hombres",
    "Anuncio para mujeres",
    "Anuncio para ambos géneros",
    "Anuncio neutro",
]

PRODUCT_CATEGORIES = [
    "comida rápida",
    "alimentos saludables",
    "bebidas azucaradas",
    "refrescos",
    "teléfonos móviles",
    "ordenadores y tablets",
    "accesorios electrónicos",
    "videojuegos",
    "ropa casual",
    "ropa deportiva",
    "calzado",
    "colonias y fragancias",
    "maquillaje",
    "producto para la salud dental",
    "cuidado personal",
    "juguetes infantiles",
    "muebles y decoración",
    "limpieza del hogar",
    "libros y revistas",
    "viajes y turismo",
    "coches eléctricos",
    "transporte público",
    "coches nuevos",
]


# ============================
# Maps ES → EN for saving to Supabase
# ============================

AGE_MAP_ES_EN: Dict[str, str] = {
    "Orientado a todo el público": "All audiences",
    "Orientado a niños pequeños": "Young children",
    "Orientado a adolescentes": "Teenagers",
    "Orientado a adultos": "Adults",
    "Orientado a personas mayores": "Older adults",
    "Orientado a público familiar": "Family audience",
}

GENDER_MAP_ES_EN: Dict[str, str] = {
    "Anuncio para hombres": "Men",
    "Anuncio para mujeres": "Women",
    "Anuncio para ambos géneros": "All genders",
    "Anuncio neutro": "Neutral",
}

PRODUCT_MAP_ES_EN: Dict[str, str] = {
    "comida rápida": "fast food",
    "alimentos saludables": "healthy food",
    "bebidas azucaradas": "sugary drinks",
    "refrescos": "soft drinks",
    "teléfonos móviles": "mobile phones",
    "ordenadores y tablets": "computers and tablets",
    "accesorios electrónicos": "electronic accessories",
    "videojuegos": "video games",
    "ropa casual": "casual clothing",
    "ropa deportiva": "sportswear",
    "calzado": "footwear",
    "colonias y fragancias": "colognes and fragrances",
    "maquillaje": "makeup",
    "producto para la salud dental": "dental care product",
    "cuidado personal": "personal care",
    "juguetes infantiles": "children's toys",
    "muebles y decoración": "furniture and decoration",
    "limpieza del hogar": "home cleaning",
    "libros y revistas": "books and magazines",
    "viajes y turismo": "travel and tourism",
    "coches eléctricos": "electric cars",
    "transporte público": "public transport",
    "coches nuevos": "new cars",
}


def _predict_category(image: Image.Image, categories: list[str]) -> str:
    # safety reload in case of failure
    ensure_clip_loaded()
    # The categories list is used as text prompts for CLIP
    inputs = _clip_processor(text=categories, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = _clip_model(**inputs)
        # Calculate similarity scores (logits) and convert to probabilities
        probs = outputs.logits_per_image.softmax(dim=1)

    idx = int(torch.argmax(probs, dim=1))
    return categories[idx]


def classify_ad_image(image_path: Path) -> Dict[str, str]:
    """
    Returns a dictionary with age_segment, gender_segment, product_segment
    for the given image.
    """
    ensure_clip_loaded()
    img = Image.open(image_path).convert("RGB")

    # Predictions in Spanish (using Spanish prompts)
    age_es = _predict_category(img, AGE_CATEGORIES)
    gender_es = _predict_category(img, GENDER_CATEGORIES)
    product_es = _predict_category(img, PRODUCT_CATEGORIES)

    # Translation to English for saving to Supabase
    age_en = AGE_MAP_ES_EN.get(age_es, age_es)
    gender_en = GENDER_MAP_ES_EN.get(gender_es, gender_es)
    product_en = PRODUCT_MAP_ES_EN.get(product_es, product_es)

    return {
        "age_segment": age_en,
        "gender_segment": gender_en,
        "product_segment": product_en,
    }