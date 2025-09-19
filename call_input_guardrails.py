import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from datetime import datetime
from pathlib import Path
import uvicorn

# Application created
from app.input_guardrails_main import process_query_batch

# FastAPI app
app = FastAPI()

# Mount templates folder
templates = Jinja2Templates(directory="templates")

# Log file path
LOG_FILE = Path("query_logs.jsonl")
LOG_FILE.touch(exist_ok=True)  # create file if not exists

# Pydantic model for input
class QueryBatch(BaseModel):
    queries: List[str]

# Serve index.html in the templates folder
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Batch query processing - main function
@app.post("/process-queries")
async def process_queries(batch: QueryBatch):
    start_time = time.time()
    try:
        results = process_query_batch(batch.queries)
        elapsed_time = round((time.time() - start_time) * 1000, 2)  # ms

        response = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(batch.queries),
            "results": results,
            "processing_summary": summarize_status(results),
            "time_taken_ms": elapsed_time
        }

        # Log query + response
        log_entry = {
            "timestamp": response["timestamp"],
            "queries": batch.queries,
            "response": response,
            "time_taken_ms": elapsed_time
        }
        write_log(log_entry)

        return JSONResponse(content=response)

    except Exception as e:
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "queries": batch.queries
        }
        write_log(error_response)
        return JSONResponse(content={"detail": str(e)}, status_code=500)


# Utility to summarize status counts
def summarize_status(results):
    summary = {}
    for res in results:
        status = res.get("status", "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary


# Utility to write logs
def write_log(entry: dict):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, indent=2))
        f.write("\n\n")  # two-line gap


# Start the server when script is run directly
if __name__ == "__main__":
    print("Starting Input Guardrails API...")
    print("Visit http://localhost:8000 to use the web interface")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
