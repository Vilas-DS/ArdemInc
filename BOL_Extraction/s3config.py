# s3config.py

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import asyncio
import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ----------------- S3 Ingest Configuration -----------------
S3_BUCKET = os.environ.get("BOL_S3_BUCKET", "")            # REQUIRED
S3_PREFIX = os.environ.get("BOL_S3_PREFIX", "")            # optional prefix (folder)
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_LOOKBACK_HOURS = int(os.environ.get("BOL_S3_LOOKBACK_HOURS", "24"))  # recent window

# Only consider objects with this suffix (case-insensitive)
S3_SUFFIX = ".pdf"


def _s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
    )


def s3_list_recent_pdfs(bucket: str, prefix: str, lookback_hours: int) -> List[Dict[str, Any]]:
    """
    List objects newer than lookback_hours, matching suffix .pdf, not folders.
    Returns objects as dicts: {"Key": str, "LastModified": datetime, "Size": int}
    """
    if not bucket:
        return []

    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    # Compare in UTC to avoid local-time skew
    since_utc = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    objects: List[Dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
        for item in page.get("Contents", []):
            key = item["Key"]
            size = int(item.get("Size", 0))
            if size <= 0:
                continue
            if not key.lower().endswith(S3_SUFFIX):
                continue

            lm: Optional[datetime] = item.get("LastModified")
            # AWS gives tz-aware dt (usually UTC). Normalize to UTC for comparison.
            if lm is None:
                continue
            if lm.tzinfo is None:
                lm_utc = lm.replace(tzinfo=timezone.utc)
            else:
                lm_utc = lm.astimezone(timezone.utc)

            if lm_utc >= since_utc:
                # Store a naive UTC datetime for simpler downstream JSON encoding
                objects.append({"Key": key, "LastModified": lm_utc.replace(tzinfo=None), "Size": size})

    # newest first
    objects.sort(key=lambda o: o["LastModified"], reverse=True)
    return objects


def s3_download_to_temp(bucket: str, key: str, dest_path: str) -> None:
    """
    Download the S3 object to a local path (creates parent dirs as needed).
    """
    if not bucket:
        raise ValueError("Bucket is required")
    client = _s3_client()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    client.download_file(bucket, key, dest_path)


# -----------------------------------------------------------
# Simple broadcaster for S3 ingest SSE
class S3IngestBus:
    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self.last_run: Optional[str] = None

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def publish(self, payload: Dict[str, Any]) -> None:
        # fan-out without await on put (so one slow client won't block others)
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop if a client queue is full
                pass
