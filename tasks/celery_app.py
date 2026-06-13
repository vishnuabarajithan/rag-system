from celery import Celery
from config.settings import settings

celery_app = Celery(
    "rag_ingestion",
    broker=settings.redis_url + "/0",
    backend=settings.redis_url + "/1",
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="tasks.ingest_file")
def ingest_file_task(
    self,
    filename: str,
    content_hex: str,
    collection: str = "default",
):
    from ingestion.pipeline import IngestionPipeline
    try:
        pipeline = IngestionPipeline()
        content = bytes.fromhex(content_hex)
        count = pipeline.ingest_file(filename, content, collection)
        return {"status": "ok", "chunks_upserted": count}
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="tasks.ingest_database")
def ingest_database_task(
    self,
    connection_string: str,
    table_names: list[str] | None,
    collection: str = "default",
):
    from ingestion.pipeline import IngestionPipeline
    try:
        pipeline = IngestionPipeline()
        count = pipeline.ingest_database(connection_string, table_names, collection)
        return {"status": "ok", "chunks_upserted": count}
    except Exception as exc:
        raise self.retry(exc=exc)
