from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "LabMate"
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    max_upload_bytes: int = int(os.getenv("LABMATE_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
    preview_rows: int = int(os.getenv("LABMATE_PREVIEW_ROWS", "8"))
    top_categories: int = int(os.getenv("LABMATE_TOP_CATEGORIES", "8"))


settings = Settings()
