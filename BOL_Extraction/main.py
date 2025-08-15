# =======================================================
# == PRE-LOADING BLOCK TO FIX WINDOWS DLL ISSUES ==
# On Windows, the import order of complex libraries like
# PyTorch and OpenCV can matter. Importing them upfront in
# a specific order helps resolve underlying DLL conflicts.
# =======================================================
print("Pre-loading core libraries to prevent DLL conflicts...")
try:
    import numpy
    import torch
    import cv2
    import paddleocr
    print("Core libraries pre-loaded successfully.")
except ImportError as e:
    print(f"Error pre-loading libraries: {e}")
# =======================================================

import os
import shutil
import logging
import json
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Import the refactored extractor components
from extractor import PDFExtractor
from text_extractor import TextExtractor, JsonExtractor
from table_extractor import TableExtractor

# --- Configuration ---
UPLOAD_DIRECTORY = "uploads"
RESULTS_DIRECTORY = os.path.join("static", "results")
HISTORY_FILE = "history.json"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

# --- Logging (configured by log_config.yaml) ---
logger = logging.getLogger(__name__)

# --- Global In-Memory Storage ---
extractor_instance = None
results_storage = {} 

# --- Helper Functions ---
def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    with open(HISTORY_FILE, 'r') as f:
        try: return json.load(f)
        except json.JSONDecodeError: return []

def save_history(data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def cleanup_old_results():
    """Removes result artifacts older than 24 hours."""
    logger.info("Running cleanup task for old results...")
    for result_id in os.listdir(RESULTS_DIRECTORY):
        result_path = os.path.join(RESULTS_DIRECTORY, result_id)
        try:
            mod_time = os.path.getmtime(result_path)
            if datetime.now() - datetime.fromtimestamp(mod_time) > timedelta(hours=24):
                shutil.rmtree(result_path)
                logger.info(f"Removed old result directory: {result_path}")
        except Exception as e:
            logger.error(f"Error during cleanup of {result_path}: {e}")

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor_instance
    logger.info("Application startup: Initializing extractor components...")
    try:
        text_extractor = TextExtractor(logger)
        table_extractor = TableExtractor(logger)
        json_extractor = JsonExtractor(logger)
        extractor_instance = PDFExtractor(logger, text_extractor, table_extractor, json_extractor)
        logger.info("PDFExtractor and its components are ready.")
        cleanup_old_results()
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize PDFExtractor: {e}", exc_info=True)
        extractor_instance = None
    
    yield
    
    logger.info("Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/results/{result_id}")
async def get_result_page(result_id: str):
    return FileResponse('static/results.html')

@app.get("/api/results/{result_id}")
async def get_result_data(result_id: str):
    result = results_storage.get(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found.")
    return JSONResponse(content=result)

@app.post("/api/extract/")
async def extract_data_from_pdf(files: list[UploadFile] = File(...)):
    if not extractor_instance:
        raise HTTPException(status_code=503, detail="Extractor service is not available.")

    file_buffers = [(file.filename, await file.read()) for file in files]

    async def event_stream():
        total_files = len(file_buffers)
        all_result_ids = []
        history = load_history()

        for i, (filename, content) in enumerate(file_buffers):
            progress = i + 1
            if not filename.lower().endswith('.pdf'):
                logger.warning(f"Skipping non-PDF file: {filename}")
                continue
            
            yield f"data: {json.dumps({'status': 'processing', 'filename': filename, 'currentFile': progress, 'totalFiles': total_files})}\n\n"
            await asyncio.sleep(0.1)

            result_id = str(uuid.uuid4())
            output_dir = os.path.join(RESULTS_DIRECTORY, result_id)
            os.makedirs(output_dir, exist_ok=True)
            temp_pdf_path = os.path.join(UPLOAD_DIRECTORY, f"{result_id}_{filename}")

            try:
                with open(temp_pdf_path, "wb") as f:
                    f.write(content)

                logger.info(f"Processing file: {filename}")
                bills_data = extractor_instance.extract_json_data_tables(temp_pdf_path, result_id, output_dir)
                
                results_storage[result_id] = {"filename": filename, "data": bills_data}
                all_result_ids.append(result_id)

                history.insert(0, {"id": result_id, "filename": filename, "timestamp": datetime.now().isoformat()})

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}", exc_info=True)
                yield f"data: {json.dumps({'status': 'error', 'filename': filename, 'message': str(e)})}\n\n"
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        save_history(history)
        yield f"data: {json.dumps({'status': 'complete', 'result_ids': all_result_ids})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/history")
async def get_history():
    return JSONResponse(content=load_history())

@app.get("/api/download/json/{result_id}")
async def download_json_data(result_id: str):
    result = results_storage.get(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found.")
    
    combined_json = {
        bill_name: content["json_data"]
        for bill_name, content in result.get("data", {}).items()
    }
    json_content = json.dumps(combined_json, indent=4)
    filename = f"{os.path.splitext(result['filename'])[0]}_extracted_data.json"
    
    return Response(
        content=json_content,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""}
    )

@app.get("/api/download/excel/{result_id}/{bill_name}/{table_index}")
async def download_excel_table(result_id: str, bill_name: str, table_index: int):
    result = results_storage.get(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found.")

    try:
        table_data = result["data"][bill_name]["data_tables"][table_index]
        df = pd.DataFrame(table_data)
        
        temp_excel_path = os.path.join(UPLOAD_DIRECTORY, f"{uuid.uuid4()}.xlsx")
        df.to_excel(temp_excel_path, index=False)
        
        base_filename = os.path.splitext(result['filename'])[0]
        download_filename = f"{base_filename}_{bill_name}_table_{table_index + 1}.xlsx"

        return FileResponse(
            path=temp_excel_path,
            filename=download_filename,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            background=lambda: os.remove(temp_excel_path)
        )
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=404, detail=f"Table not found: {e}")
    except Exception as e:
        logger.error(f"Excel generation failed for result {result_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate Excel file.")