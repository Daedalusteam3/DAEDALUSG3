# daedalus/apps.py
import os
from django.apps import AppConfig



class DaedalusConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "daedalus"

    def ready(self):
        print("[DEBUG] DaedalusConfig.ready() llamado, inicializando CLIP...", flush=True)
        from .clip_ad_classifier import ensure_clip_loaded
        ensure_clip_loaded()
        print("[DEBUG] CLIP inicializado, continuando arranque de Django.", flush=True)
