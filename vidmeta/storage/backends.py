from __future__ import annotations

import shutil
from pathlib import Path

from vidmeta.settings import StorageSettings, processing_dir, upload_dir


def upload_path(upload_id: str, filename: str) -> Path:
    safe_name = Path(filename).name or "upload.bin"
    return upload_dir() / f"{upload_id}_{safe_name}"


def processing_path(upload_id: str, filename: str) -> Path:
    safe_name = Path(filename).name or "upload.bin"
    return processing_dir() / f"{upload_id}_{safe_name}"


def maybe_archive_upload(local_path: Path, upload_id: str, filename: str, settings: StorageSettings) -> str:
    if settings.backend == "local_disk":
        return str(local_path)
    if settings.backend == "s3_compatible":
        return _upload_to_s3(local_path, upload_id, filename, settings)
    raise ValueError(f"Unsupported storage backend: {settings.backend}")


def materialize_for_processing(stored_path: str, upload_id: str, filename: str, settings: StorageSettings) -> str:
    if settings.backend == "local_disk":
        return stored_path
    if settings.backend == "s3_compatible":
        target = processing_path(upload_id, filename)
        _download_from_s3(stored_path, target, settings)
        return str(target)
    raise ValueError(f"Unsupported storage backend: {settings.backend}")


def import_local_file_if_needed(source: str, job_id: str, settings: StorageSettings) -> str:
    if not settings.import_local_files:
        return source
    source_path = Path(source)
    target = upload_dir() / f"{job_id}_{source_path.name}"
    shutil.copyfile(source_path, target)
    return str(target)


def _upload_to_s3(local_path: Path, upload_id: str, filename: str, settings: StorageSettings) -> str:
    import boto3

    key = f"uploads/{upload_id}/{Path(filename).name}"
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        region_name=settings.s3_region or None,
        aws_access_key_id=settings.s3_access_key_id or None,
        aws_secret_access_key=settings.s3_secret_access_key or None,
    )
    client.upload_file(str(local_path), settings.s3_bucket, key)
    return f"s3://{settings.s3_bucket}/{key}"


def _download_from_s3(uri: str, target: Path, settings: StorageSettings) -> None:
    import boto3

    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket_key = uri.removeprefix("s3://")
    bucket, key = bucket_key.split("/", 1)
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        region_name=settings.s3_region or None,
        aws_access_key_id=settings.s3_access_key_id or None,
        aws_secret_access_key=settings.s3_secret_access_key or None,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))
