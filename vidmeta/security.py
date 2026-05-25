from __future__ import annotations

import base64
import hashlib
import os
import secrets
from pathlib import Path


_ENCRYPTED_PREFIX = "enc:"


def _key_file_path() -> Path:
    from vidmeta.settings import data_dir
    return data_dir() / ".secret_key"


def _derive_key() -> bytes:
    key_str = os.environ.get("VIDMETA_SECRET_KEY", "")
    if not key_str:
        key_file = _key_file_path()
        if key_file.exists():
            key_str = key_file.read_text().strip()
        else:
            key_str = secrets.token_urlsafe(32)
            key_file.write_text(key_str)
            os.chmod(key_file, 0o600)
    return base64.urlsafe_b64encode(hashlib.sha256(key_str.encode()).digest())


def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(_derive_key())


def encrypt_secret(value: str) -> str:
    if not value:
        return value
    if value.startswith(_ENCRYPTED_PREFIX):
        return value
    try:
        token = _fernet().encrypt(value.encode()).decode()
        return f"{_ENCRYPTED_PREFIX}{token}"
    except Exception:
        return value


def decrypt_secret(value: str) -> str:
    if not value or not value.startswith(_ENCRYPTED_PREFIX):
        return value
    try:
        return _fernet().decrypt(value.removeprefix(_ENCRYPTED_PREFIX).encode()).decode()
    except Exception:
        return value
