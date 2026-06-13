from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from tasks.celery_app import ingest_file_task, ingest_database_task

router = APIRouter(prefix="/ingest", tags=["ingest"])


class TaskResponse(BaseModel):
    task_id: str
    status: str
    filename: str | None = None


class DatabaseIngestRequest(BaseModel):
    connection_string: str
    table_names: list[str] | None = None
    collection: str = "default"


@router.post("/upload", response_model=TaskResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = "default",
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    task = ingest_file_task.delay(
        filename=file.filename,
        content_hex=content.hex(),
        collection=collection,
    )
    return TaskResponse(task_id=task.id, status="queued", filename=file.filename)


@router.post("/database", response_model=TaskResponse)
async def ingest_database(req: DatabaseIngestRequest):
    task = ingest_database_task.delay(
        connection_string=req.connection_string,
        table_names=req.table_names,
        collection=req.collection,
    )
    return TaskResponse(task_id=task.id, status="queued")


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    from tasks.celery_app import celery_app
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status":  result.status,
        "result":  result.result if result.ready() else None,
    }
