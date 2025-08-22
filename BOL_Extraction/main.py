# main.py

# =======================================================
# == PRE-LOADING BLOCK TO REDUCE WINDOWS DLL GLITCHES ==
# =======================================================
print("Pre-loading core libraries to prevent DLL conflicts...")
for _mod in ("numpy", "cv2", "torch", "paddleocr"):
    try:
        __import__(_mod)
        print(f" - {_mod} OK")
    except Exception as e:
        print(f" ! {_mod} preload warning: {e}")
print("Preload phase complete.")
# =======================================================

import os
import re
import logging
import json
import uuid
import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from botocore.exceptions import ClientError

# Project imports (assumed present in your repo)
from extractor import PDFExtractor
from text_extractor import TextExtractor, JsonExtractor
from table_extractor import TableExtractor
from storage import StorageBackend, LocalDiskStorage, S3Storage
from s3config import (
    S3_BUCKET, S3_PREFIX, S3_LOOKBACK_HOURS,
    S3IngestBus, s3_list_recent_pdfs, s3_download_to_temp, _s3_client
)

# SQL helpers
from dbconfig import _db_conn, ensure_tables, db_is_processed, db_mark_processed, db_upsert_result

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIRECTORY = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIRECTORY = STATIC_DIR / "results"
HISTORY_FILE = BASE_DIR / "history.json"
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Optional: Public base URL for launcher pages; fallback to runtime origin in HTML
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("bol_app")

# --- Global In-Memory Storage (per process) ---
extractor_instance: Optional[PDFExtractor] = None
results_storage: Dict[str, Dict[str, Any]] = {}

# S3 ingest messaging
s3_bus = S3IngestBus()
s3_ingest_task: Optional[asyncio.Task] = None
s3_task_lock = asyncio.Lock()

# Storage backend (local by default; swap to S3Storage if desired)
storage: StorageBackend = LocalDiskStorage(logger=logger)
# storage = S3Storage(...)

# History fallback lock
history_lock = asyncio.Lock()

# ------------ Utilities ------------
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

def _sanitize(name: str) -> str:
    return SAFE_NAME.sub("_", name.replace(" ", "_"))[:200]

def load_history_json() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

def save_history_json(data: List[Dict[str, Any]]) -> None:
    HISTORY_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")

async def append_history(entry: dict):
    async with history_lock:
        data = load_history_json()
        data.insert(0, entry)
        save_history_json(data)




def _render_results_launcher_html(result_id: str) -> str:
    """
    Generates a tiny HTML that redirects to /results/{id}.
    Prefers APP_BASE_URL if provided; falls back to window.location.origin at runtime.
    """
    # Safe-embed the configured base (can be empty)
    base = json.dumps(APP_BASE_URL)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Open Result {result_id}</title>
</head>
<body>
<script>
  (function(){{
    var configured = {base} || "";
    var originBase = (typeof location !== 'undefined' && location.origin) ? location.origin : "";
    var base = configured || originBase || "";
    var target = (base ? base : "") + "/results/{result_id}";
    try {{ location.replace(target); }} catch(e) {{}}
    document.write('Opening results… If you are not redirected, <a href="' + target + '">click here</a>.');
  }})();
</script>
<noscript>Open <a href="/results/{result_id}">result</a>.</noscript>
</body>
</html>"""

# ----------- SQL-backed history (or JSON fallback) -----------
USE_SQLSERVER = False

def _db_init() -> bool:
    """
    Try to connect and ensure the ExtractionHistory table exists.
    If connection fails or env is missing, returns False (JSON fallback).
    """
    try:
        cn = _db_conn()
    except Exception as e:
        logger.warning(f"SQL Server not configured or unreachable: {e}")
        return False

    ddl = r"""
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[ExtractionHistory]') AND type = N'U')
BEGIN
    CREATE TABLE [dbo].[ExtractionHistory](
        [result_id]  NVARCHAR(64) NOT NULL PRIMARY KEY,
        [filename]   NVARCHAR(512) NOT NULL,
        [uploaded_at] DATETIME2 NOT NULL,
        [pages] INT NULL,
        [bills] INT NULL
    );
    CREATE INDEX IX_ExtractionHistory_uploaded_at ON [dbo].[ExtractionHistory]([uploaded_at] DESC);
    CREATE INDEX IX_ExtractionHistory_filename ON [dbo].[ExtractionHistory]([filename]);
END
"""
    cur = cn.cursor()
    try:
        cur.execute(ddl)
    finally:
        try:
            cur.close()
        finally:
            cn.close()
    return True

def db_insert_history(result_id: str, filename: str, uploaded_at_iso: str,
                      pages: Optional[int], bills: Optional[int]) -> None:
    if not USE_SQLSERVER:
        hist = load_history_json()
        hist.insert(0, {
            "id": result_id,
            "filename": filename,
            "timestamp": uploaded_at_iso,
            "pages": pages,
            "bills": bills
        })
        save_history_json(hist)
        return

    cn = _db_conn()
    cur = cn.cursor()
    try:
        uploaded_dt = datetime.fromisoformat(uploaded_at_iso)
        cur.execute(
            """
IF EXISTS (SELECT 1 FROM [dbo].[ExtractionHistory] WHERE result_id=?)
    UPDATE [dbo].[ExtractionHistory]
       SET filename=?, uploaded_at=?, pages=?, bills=?
     WHERE result_id=?;
ELSE
    INSERT INTO [dbo].[ExtractionHistory](result_id, filename, uploaded_at, pages, bills)
    VALUES(?, ?, ?, ?, ?);
""",
            result_id, filename, uploaded_dt, pages, bills, result_id,
            result_id, filename, uploaded_dt, pages, bills
        )
    finally:
        try:
            cur.close()
        finally:
            cn.close()

def db_query_history(limit: int, offset: int,
                     q: Optional[str], start_date: Optional[str], end_date: Optional[str]) -> List[Dict[str, Any]]:
    if not USE_SQLSERVER:
        data = load_history_json()
        if q:
            ql = q.lower()
            data = [r for r in data if ql in (r.get("filename") or "").lower()]
        def _dt(s):
            try: return datetime.fromisoformat(s)
            except: return None
        if start_date:
            try:
                sd = datetime.fromisoformat(start_date)
                data = [r for r in data if r.get("timestamp") and _dt(r["timestamp"]) and _dt(r["timestamp"]) >= sd]
            except: pass
        if end_date:
            try:
                ed = datetime.fromisoformat(end_date)
                data = [r for r in data if r.get("timestamp") and _dt(r["timestamp"]) and _dt(r["timestamp"]) <= ed]
            except: pass
        data.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
        return data[offset: offset + limit]

    where_clauses, params = [], []
    if q:
        where_clauses.append("filename LIKE ?")
        params.append(f"%{q}%")
    if start_date:
        where_clauses.append("uploaded_at >= ?")
        try: params.append(datetime.fromisoformat(start_date))
        except: params.append(datetime.fromisoformat(start_date + "T00:00:00"))
    if end_date:
        where_clauses.append("uploaded_at <= ?")
        try: params.append(datetime.fromisoformat(end_date))
        except: params.append(datetime.fromisoformat(end_date + "T23:59:59"))

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = f"""
SELECT result_id, filename, uploaded_at, pages, bills
FROM [dbo].[ExtractionHistory]
{where_sql}
ORDER BY uploaded_at DESC
OFFSET ? ROWS FETCH NEXT ? ROWS ONLY;
"""
    params.extend([offset, limit])

    cn = _db_conn()
    cur = cn.cursor()
    rows: List[Dict[str, Any]] = []
    try:
        cur.execute(sql, params)
        for result_id, filename, uploaded_at, pages, bills in cur.fetchall():
            rows.append({
                "id": result_id,
                "filename": filename,
                "timestamp": uploaded_at.isoformat() if isinstance(uploaded_at, datetime) else str(uploaded_at),
                "pages": int(pages) if pages is not None else None,
                "bills": int(bills) if bills is not None else None
            })
    finally:
        try:
            cur.close()
        finally:
            cn.close()
    return rows

def db_reset_stale_jobs():
    # No-op: results table doesn't have a 'status' column.
    return

# ---------- Cleanup old artifacts (disabled by default) ----------
def cleanup_old_results():
    logger.info("Cleanup task disabled (retaining all results).")

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor_instance, USE_SQLSERVER

    logger.info("Application startup: Initializing extractor components...")
    try:
        text_extractor = TextExtractor(logger)
        table_extractor = TableExtractor(logger)
        json_extractor = JsonExtractor(logger)
        extractor_instance = PDFExtractor(logger, text_extractor, table_extractor, json_extractor)
        logger.info("PDFExtractor and its components are ready.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize PDFExtractor: {e}", exc_info=True)
        extractor_instance = None

    USE_SQLSERVER = _db_init()
    if USE_SQLSERVER:
        logger.info("SQL Server history store enabled.")
        try:
            ensure_tables()
            logger.info("SQL Server S3 bookkeeping tables ensured.")
        except Exception as e:
            logger.error(f"Failed to ensure S3 tables: {e}", exc_info=True)
    else:
        logger.info("Using JSON file history store (SQL Server disabled or unreachable).")

    db_reset_stale_jobs()

    yield
    logger.info("Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# CORS (loose for dev; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- System Status Endpoint ---
@app.get("/api/status")
async def get_system_status():
    # Check DB status
    db_status = "error"
    try:
        with _db_conn() as cn:
            cn.cursor().execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        logger.warning(f"DB connection check failed: {e}")
        db_status = "error"

    # Check S3 status
    s3_status = "not_configured"
    if S3_BUCKET:
        try:
            s3 = _s3_client()
            s3.head_bucket(Bucket=S3_BUCKET)
            s3_status = "connected"
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            logger.warning(f"S3 connection check failed with code: {error_code}")
            if error_code == '404':
                s3_status = "bucket_not_found"
            elif error_code == '403':
                s3_status = "access_denied"
            else:
                s3_status = "error"
        except Exception as e:
            logger.warning(f"S3 connection check failed: {e}")
            s3_status = "error"

    return JSONResponse({
        "db_status": db_status,
        "s3_status": s3_status
    })

# --- Dashboard Endpoints ---
def _parse_date_param(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        # Expect YYYY-MM-DD
        return datetime.strptime(val, "%Y-%m-%d")
    except Exception:
        return None

def _get_window(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    now = datetime.utcnow()
    s = _parse_date_param(start)
    e = _parse_date_param(end)
    if not s and not e:
        e = datetime(now.year, now.month, now.day) + timedelta(days=1)
        s = e - timedelta(days=7)
    elif s and not e:
        e = datetime(now.year, now.month, now.day) + timedelta(days=1)
    elif not s and e:
        s = e - timedelta(days=7)
    
    if e:
        e = datetime(e.year, e.month, e.day) + timedelta(days=1)
    if s:
        s = datetime(s.year, s.month, s.day)
    return s, e

@app.get("/api/dashboard/summary")
def dashboard_summary(start: str | None = Query(None), end: str | None = Query(None)):
    start_dt, end_dt = _get_window(start, end)
    
    try:
        with _db_conn() as cn:
            cur = cn.cursor()
            sql = """
            SELECT
              COUNT(*) AS received,
              SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) AS processed,
              SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
              SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected
            FROM results
            WHERE created_at >= ? AND created_at < ?
            """
            cur.execute(sql, start_dt, end_dt)
            row = cur.fetchone()
            if row:
                return {
                    "received": row.received or 0,
                    "processed": row.processed or 0,
                    "delivered": row.delivered or 0,
                    "rejected": row.rejected or 0,
                }
    except Exception as e:
        logger.error(f"Dashboard summary error: {e}")
        # Fallback if the query fails (e.g., 'status' column doesn't exist)
        with _db_conn() as cn:
            cur = cn.cursor()
            sql = "SELECT COUNT(*) FROM results WHERE created_at >= ? AND created_at < ?"
            cur.execute(sql, start_dt, end_dt)
            count = cur.fetchone()[0]
            return {"received": count, "processed": count, "delivered": 0, "rejected": 0}

    return {"received": 0, "processed": 0, "delivered": 0, "rejected": 0}

@app.get("/api/dashboard/trends")
def dashboard_trends(start: str | None = Query(None), end: str | None = Query(None)):
    start_dt, end_dt = _get_window(start, end)
    
    day_count = (end_dt - start_dt).days
    labels = [(start_dt + timedelta(days=i)).strftime("%a") for i in range(day_count)]
    
    received_data = [0] * len(labels)
    processed_data = [0] * len(labels)

    try:
        with _db_conn() as cn:
            cur = cn.cursor()
            # Query for received files
            sql_received = "SELECT CAST(created_at AS DATE) as date, COUNT(*) as count FROM results WHERE created_at >= ? AND created_at < ? GROUP BY CAST(created_at AS DATE) ORDER BY date"
            cur.execute(sql_received, start_dt, end_dt)
            for row in cur.fetchall():
                day_index = (row.date - start_dt.date()).days
                if 0 <= day_index < len(labels):
                    received_data[day_index] = row.count
            
            # Query for processed files
            sql_processed = "SELECT CAST(created_at AS DATE) as date, COUNT(*) as count FROM results WHERE status = 'processed' AND created_at >= ? AND created_at < ? GROUP BY CAST(created_at AS DATE) ORDER BY date"
            cur.execute(sql_processed, start_dt, end_dt)
            for row in cur.fetchall():
                day_index = (row.date - start_dt.date()).days
                if 0 <= day_index < len(labels):
                    processed_data[day_index] = row.count

    except Exception as e:
        logger.error(f"Dashboard trends error: {e}")


    return {
        "labels": labels,
        "received": received_data,
        "processed": processed_data,
    }

# --- Pages ---
@app.get("/")
async def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/results/{result_id}")
async def get_result_page(result_id: str):
    # Always serve the shared results template.
    return FileResponse(str(STATIC_DIR / "results.html"))

@app.get("/history")
async def history_page():
    return FileResponse(str(STATIC_DIR / "history.html"))

# --- API: Results fetch ---
@app.get("/api/results/{result_id}")
async def get_result_data(result_id: str):
    # 1) try in-memory
    result = results_storage.get(result_id)
    if result:
        return JSONResponse(content=result)

    # 2) fall back to persisted JSON
    json_path = storage.file_path(result_id, "result.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)

    raise HTTPException(status_code=404, detail="Result not found.")

# --- API: History with filters (SQL or JSON) ---
@app.get("/api/history")
async def get_history(
    limit: int = Query(5, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    rows = db_query_history(limit=limit, offset=offset, q=q, start_date=start_date, end_date=end_date)
    return JSONResponse(content=rows)

# =========================
# Persisting artifacts (JSON/Excel/manifest/launcher)
# =========================
def persist_full_result(filename: str, result_id: str, result_obj: Dict[str, Any]) -> None:
    storage.ensure_dir(result_id)

    # 1) Save structured JSON
    storage.write_json(result_id, "result.json", result_obj)

    # 2) Generate and persist Excels
    base = _sanitize(Path(filename).stem)
    manifest_excels: List[str] = []
    data = result_obj.get("data", {})
    for bill_name, content in data.items():
        safe_bill = _sanitize(bill_name)
        tables = content.get("data_tables", []) or []
        for idx, table in enumerate(tables, start=1):
            try:
                df = pd.DataFrame.from_records(table)
            except Exception:
                df = pd.DataFrame(table)
            excel_name = f"{base}_{safe_bill}_table_{idx}.xlsx"
            storage.write_excel(result_id, excel_name, df)
            manifest_excels.append(excel_name)

    # 3) Save a tiny launcher that redirects to the original viewer route
    launcher_html = _render_results_launcher_html(result_id)
    storage.write_text(result_id, "open.html", launcher_html)

    # 4) Save a manifest (optional but handy)
    manifest_images: List[str] = []
    for bill in data.values():
        for img_url in bill.get("page_images", []) or []:
            name = os.path.basename(img_url)  # /static/results/{id}/page_*.png -> page_*.png
            manifest_images.append(name)

    storage.write_json(result_id, "manifest.json", {
        "filename": filename,
        "result_id": result_id,
        "saved_at": datetime.now().isoformat(),
        "json": "result.json",
        "open": "open.html",
        "excels": manifest_excels,
        "images": manifest_images
    })

    # 5) Mirror to S3 if using S3Storage
    if isinstance(storage, S3Storage):
        storage.sync_dir(result_id)

# =========================
# S3 Ingest (optional)
# =========================
@app.post("/api/ingest/s3/scan")
async def api_s3_scan():
    """Trigger an S3 scan+ingest if not already running."""
    global s3_ingest_task
    if not S3_BUCKET:
        raise HTTPException(status_code=400, detail="S3 bucket not configured (S3_BUCKET).")

    async with s3_task_lock:
        if s3_ingest_task is None or s3_ingest_task.done():
            s3_ingest_task = asyncio.create_task(run_s3_ingest())
            return JSONResponse({"started": True})
        else:
            return JSONResponse({"started": False, "message": "Ingest already running."}, status_code=202)

@app.get("/api/ingest/s3/status")
async def api_s3_status():
    """Return last run time; useful for UI status line."""
    return JSONResponse({"last_run": s3_bus.last_run})

@app.get("/api/ingest/s3/stream")
async def api_s3_stream():
    """
    SSE stream of ingest events. Events:
      {status: 'processing', filename, currentFile, totalFiles}
      {status: 'error', filename, message}
      {status: 'processed', filename, result_id, currentFile, totalFiles}
      {status: 'complete', result_ids: [...]}
    """
    client_q = await s3_bus.subscribe()

    async def event_gen():
        try:
            while True:
                item = await client_q.get()
                yield f"data: {json.dumps(item)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await s3_bus.unsubscribe(client_q)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

async def run_s3_ingest():
    if not extractor_instance:
        await s3_bus.publish({"status": "error", "message": "Extractor service is not available."})
        return

    bucket = S3_BUCKET
    prefix = S3_PREFIX
    try:
        objs = await asyncio.to_thread(s3_list_recent_pdfs, bucket, prefix, S3_LOOKBACK_HOURS)
    except Exception as e:
        logger.error(f"S3 list failed: {e}", exc_info=True)
        await s3_bus.publish({"status": "error", "message": f"S3 list failed: {e}"})
        return

    work = [o for o in objs if not db_is_processed(bucket, o["Key"])]
    total = len(work)
    if total == 0:
        await s3_bus.publish({"status": "complete", "result_ids": []})
        s3_bus.last_run = datetime.now().isoformat()
        return

    all_result_ids: List[str] = []

    for idx, obj in enumerate(work, start=1):
        key = obj["Key"]
        filename = os.path.basename(key)
        await s3_bus.publish({
            "status": "processing",
            "filename": filename,
            "currentFile": idx,
            "totalFiles": total
        })

        result_id = str(uuid.uuid4())
        tmp_pdf = str(UPLOAD_DIRECTORY / f"{result_id}_{_sanitize(filename)}")
        out_dir = str(RESULTS_DIRECTORY / result_id)
        os.makedirs(out_dir, exist_ok=True)

        try:
            # download & process off the event loop
            await asyncio.to_thread(s3_download_to_temp, bucket, key, tmp_pdf)
            bills_data = await asyncio.to_thread(
                extractor_instance.extract_json_data_tables, tmp_pdf, result_id, out_dir
            )

            # in-memory cache for /api/results/{id}
            results_storage[result_id] = {"filename": filename, "data": bills_data}

            all_result_ids.append(result_id)

            # **FIX**: Record in history DB for "View All" page
            uploaded_at = datetime.now().isoformat()
            meta = bills_data.get("meta", {})
            pages = meta.get("pages")
            bills = meta.get("bills")
            db_insert_history(
                result_id=result_id,
                filename=filename,
                uploaded_at_iso=uploaded_at,
                pages=pages,
                bills=bills
            )

            # SQL bookkeeping
            try:
                db_upsert_result(result_id=result_id, filename=filename, s3_key=key)
                db_mark_processed(bucket, key, result_id)
            except Exception as db_e:
                logger.warning(f"DB upsert/mark failed for {key}: {db_e}")

            # notify UI
            await s3_bus.publish({
                "status": "processed",
                "filename": filename,
                "result_id": result_id,
                "currentFile": idx,
                "totalFiles": total
            })

        except Exception as e:
            logger.error(f"S3 ingest failed for {key}: {e}", exc_info=True)
            await s3_bus.publish({"status": "error", "filename": filename, "message": str(e)})

        finally:
            try:
                if os.path.exists(tmp_pdf):
                    os.remove(tmp_pdf)
            except Exception:
                pass

    await s3_bus.publish({"status": "complete", "result_ids": all_result_ids})
    s3_bus.last_run = datetime.now().isoformat()

# =========================
# Upload & Extract (SSE)
# =========================
from uuid import uuid4

@app.post("/api/extract/")
async def extract_data_from_pdf(files: List[UploadFile] = File(...)):
    if not extractor_instance:
        raise HTTPException(status_code=503, detail="Extractor service is not available.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    buffered: List[Dict[str, Any]] = []
    for idx, f in enumerate(files, start=1):
        filename = f.filename or f"upload_{idx}.pdf"
        if not filename.lower().endswith(".pdf"):
            try:
                while await f.read(1024 * 1024): pass
            finally:
                await f.close()
            buffered.append({"filename": filename, "temp_path": None, "error": "Not a PDF"})
            continue

        sanitized = _sanitize(filename)
        temp_path = UPLOAD_DIRECTORY / f"in_{uuid4()}_{sanitized}"
        try:
            with temp_path.open("wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk: break
                    out.write(chunk)
        finally:
            await f.close()

        buffered.append({"filename": filename, "temp_path": str(temp_path), "error": None})

    total_files = len(buffered)

    async def event_stream():
        for file_index, item in enumerate(buffered, start=1):
            filename = item["filename"]
            temp_pdf_path = item["temp_path"]

            if item["error"]:
                yield f"data: {json.dumps({'event': 'error', 'filename': filename, 'fileIndex': file_index, 'totalFiles': total_files, 'message': item['error']})}\n\n"
                continue

            yield f"data: {json.dumps({'event': 'start_file', 'filename': filename, 'fileIndex': file_index, 'totalFiles': total_files})}\n\n"
            await asyncio.sleep(0.01)

            result_id = str(uuid.uuid4())
            result_dir = storage.ensure_dir(result_id)
            queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def progress_cb(payload: Dict[str, Any]):
                payload.update({"filename": filename, "fileIndex": file_index, "totalFiles": total_files})
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, payload)
                except RuntimeError: pass

            try:
                logger.info(f"Processing file: {filename}")
                task = asyncio.create_task(asyncio.to_thread(
                    extractor_instance.extract_json_data_tables,
                    temp_pdf_path, result_id, result_dir, progress_cb
                ))

                while not task.done():
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.25)
                        yield f"data: {json.dumps(item)}\n\n"
                    except asyncio.TimeoutError:
                        pass
                
                result_obj = await task
                results_storage[result_id] = {"filename": filename, **result_obj}
                persist_full_result(filename, result_id, results_storage[result_id])
                
                uploaded_at = datetime.now().isoformat()
                meta = result_obj.get("meta", {})
                pages = meta.get("pages")
                bills = meta.get("bills")
                
                db_insert_history(result_id=result_id, filename=filename, uploaded_at_iso=uploaded_at, pages=pages, bills=bills)
                
                # **FIX**: Log manual uploads to results table for dashboard
                db_upsert_result(result_id=result_id, filename=filename, s3_key=None)

                yield f"data: {json.dumps({'event': 'complete_file', 'result_id': result_id, 'filename': filename, 'uploadedAt': uploaded_at, 'pages': pages, 'bills': bills, 'fileIndex': file_index, 'totalFiles': total_files})}\n\n"

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}", exc_info=True)
                yield f"data: {json.dumps({'event': 'error', 'filename': filename, 'fileIndex': file_index, 'totalFiles': total_files, 'message': str(e)})}\n\n"
            finally:
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        yield f"data: {json.dumps({'event': 'complete_all'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# =========================
# Downloads – always from persisted files
# =========================
@app.get("/api/download/json/{result_id}")
async def download_json_data(result_id: str):
    json_path = storage.file_path(result_id, "result.json")
    if not os.path.exists(json_path):
        in_mem = results_storage.get(result_id)
        if not in_mem:
            raise HTTPException(status_code=404, detail="Result not found.")
        storage.write_json(result_id, "result.json", in_mem)
    return FileResponse(json_path, filename=f"{result_id}_extracted_data.json", media_type="application/json")

@app.get("/api/download/excel/{result_id}/{bill_name}/{table_index}")
async def download_excel_table(result_id: str, bill_name: str, table_index: int):
    bill_safe = _sanitize(bill_name)
    base_name = result_id
    manifest_path = storage.file_path(result_id, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            base_name = _sanitize(Path(m.get("filename", result_id)).stem)
        except Exception:
            pass

    excel_name = f"{base_name}_{bill_safe}_table_{table_index + 1}.xlsx"
    excel_path = storage.file_path(result_id, excel_name)

    if not os.path.exists(excel_path):
        json_path = storage.file_path(result_id, "result.json")
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail="Result data not found.")
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data_root = payload.get("data") or {}
        key = bill_name if bill_name in data_root else next((k for k in data_root if k.lower() == bill_name.lower()), None)
        if not key:
            raise HTTPException(status_code=404, detail=f"Bill not found: {bill_name}")
        try:
            table = data_root[key]["data_tables"][table_index]
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Table not found: {e}")
        try:
            df = pd.DataFrame.from_records(table)
        except Exception:
            df = pd.DataFrame(table)
        storage.write_excel(result_id, excel_name, df)

    return FileResponse(
        path=excel_path,
        filename=excel_name,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
