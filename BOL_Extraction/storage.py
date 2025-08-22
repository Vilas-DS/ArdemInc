# storage.py

from __future__ import annotations

import json
import mimetypes
import os
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# --- Configuration (local cache root) ---
RESULTS_DIRECTORY = os.path.join("static", "results")
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)


class StorageBackend:
    """Interface for saving artifacts. Implementations: LocalDiskStorage, S3Storage."""
    def ensure_dir(self, result_id: str) -> str: ...
    def write_text(self, result_id: str, rel_path: str, text: str) -> str: ...
    def write_json(self, result_id: str, rel_path: str, data: Any) -> str: ...
    def write_excel(self, result_id: str, rel_path: str, df: pd.DataFrame) -> str: ...
    def file_path(self, result_id: str, rel_path: str) -> str: ...
    def exists(self, result_id: str, rel_path: str) -> bool: ...


class LocalDiskStorage(StorageBackend):
    """
    Local disk implementation. Used directly or as the cache layer by S3Storage.
    """
    base_dir = RESULTS_DIRECTORY  # static/results

    def __init__(self, logger=None):
        self.logger = logger

    def ensure_dir(self, result_id: str) -> str:
        d = os.path.join(self.base_dir, result_id)
        os.makedirs(d, exist_ok=True)
        return d

    def write_text(self, result_id: str, rel_path: str, text: str) -> str:
        full = self.file_path(result_id, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        return full

    def write_json(self, result_id: str, rel_path: str, data: Any) -> str:
        full = self.file_path(result_id, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return full

    def write_excel(self, result_id: str, rel_path: str, df: pd.DataFrame) -> str:
        full = self.file_path(result_id, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        # Requires openpyxl for .xlsx (ensure it's in requirements)
        df.to_excel(full, index=False)
        return full

    def file_path(self, result_id: str, rel_path: str) -> str:
        return os.path.join(self.base_dir, result_id, rel_path.replace("\\", "/"))

    def exists(self, result_id: str, rel_path: str) -> bool:
        return os.path.exists(self.file_path(result_id, rel_path))


class S3Storage(StorageBackend):
    """
    Hybrid storage:
      - ALWAYS writes to local disk (so FastAPI StaticFiles can serve artifacts).
      - Also mirrors files to S3 under s3://{bucket}/{prefix}/{result_id}/{rel_path}.
    """
    def __init__(
        self,
        logger,
        bucket: str | None = None,
        prefix: str | None = None,
        region: str | None = None,
        keep_local_cache: bool = True,
        local_delegate: LocalDiskStorage | None = None,
    ):
        self.logger = logger
        self.bucket = (bucket or os.getenv("S3_BUCKET") or "").strip()
        if not self.bucket:
            raise RuntimeError("S3 bucket not configured. Set S3_BUCKET env var.")
        self.prefix = (prefix or os.getenv("S3_PREFIX") or "").strip().strip("/")
        self.region = (region or os.getenv("AWS_REGION") or "us-east-1").strip()
        self.keep_local_cache = keep_local_cache
        self.s3 = boto3.client("s3", region_name=self.region)

        # Delegate all local work to LocalDiskStorage (required by the current app)
        self.local = local_delegate or LocalDiskStorage(logger=self.logger)

    # --------- helpers ---------
    def _key(self, result_id: str, rel_path: str) -> str:
        rel = rel_path.replace("\\", "/").lstrip("/")
        base = f"{result_id}/{rel}"
        return f"{self.prefix}/{base}".strip("/") if self.prefix else base

    def _guess_content_type(self, path: str) -> str:
        ctype, _ = mimetypes.guess_type(path)
        return ctype or "application/octet-stream"

    def _upload(self, local_path: str, key: str) -> None:
        extra = {"ContentType": self._guess_content_type(local_path)}
        self.s3.upload_file(local_path, self.bucket, key, ExtraArgs=extra)

    # --------- interface impl ---------
    def ensure_dir(self, result_id: str) -> str:
        # Ensure local directory exists; extractor will write images there.
        return self.local.ensure_dir(result_id)

    def write_text(self, result_id: str, rel_path: str, text: str) -> str:
        local_path = self.local.write_text(result_id, rel_path, text)
        try:
            self._upload(local_path, self._key(result_id, rel_path))
        except Exception as e:
            if self.logger:
                self.logger.error(f"S3 upload failed for {rel_path}: {e}", exc_info=True)
        return local_path

    def write_json(self, result_id: str, rel_path: str, data: Any) -> str:
        local_path = self.local.write_json(result_id, rel_path, data)
        try:
            self._upload(local_path, self._key(result_id, rel_path))
        except Exception as e:
            if self.logger:
                self.logger.error(f"S3 upload failed for {rel_path}: {e}", exc_info=True)
        return local_path

    def write_excel(self, result_id: str, rel_path: str, df: pd.DataFrame) -> str:
        local_path = self.local.write_excel(result_id, rel_path, df)
        try:
            self._upload(local_path, self._key(result_id, rel_path))
        except Exception as e:
            if self.logger:
                self.logger.error(f"S3 upload failed for {rel_path}: {e}", exc_info=True)
        return local_path

    def file_path(self, result_id: str, rel_path: str) -> str:
        """
        Return a LOCAL path (so existing FileResponse works).
        If missing locally but present in S3, call materialize_to_local() first.
        """
        return self.local.file_path(result_id, rel_path)

    def exists(self, result_id: str, rel_path: str) -> bool:
        # Check local first
        if self.local.exists(result_id, rel_path):
            return True
        # Then S3
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(result_id, rel_path))
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            if self.logger:
                self.logger.warning(f"S3 head_object failed for {rel_path}: {e}")
            return False

    # --------- extras ---------
    def materialize_to_local(self, result_id: str, rel_path: str) -> str:
        """
        Ensure a file is present locally. If not, download it from S3 into the
        expected local cache location and return that local path.
        """
        local_path = self.local.file_path(result_id, rel_path)
        if os.path.exists(local_path):
            return local_path

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        key = self._key(result_id, rel_path)
        try:
            self.s3.download_file(self.bucket, key, local_path)
        except ClientError as e:
            raise FileNotFoundError(f"Object not found or download failed: s3://{self.bucket}/{key}") from e
        return local_path

    def sync_dir(self, result_id: str) -> None:
        """
        Upload every file currently under the local results folder for this result_id.
        Useful to push page images saved by the extractor, plus any later artifacts.
        """
        root = self.local.ensure_dir(result_id)
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                local_path = os.path.join(dirpath, name)
                rel_path = os.path.relpath(local_path, root).replace("\\", "/")
                key = self._key(result_id, rel_path)
                try:
                    self._upload(local_path, key)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"S3 sync failed for {rel_path}: {e}", exc_info=True)

    def presigned_url(self, result_id: str, rel_path: str, expires_seconds: int = 3600) -> str:
        """Generate a presigned GET URL for direct S3 download."""
        key = self._key(result_id, rel_path)
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )
