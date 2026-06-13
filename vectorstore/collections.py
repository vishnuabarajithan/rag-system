from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
)
from config.settings import settings


def ensure_collection(client: QdrantClient, collection: str = settings.qdrant_collection) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )

    for field, schema in [
        ("filename",     PayloadSchemaType.KEYWORD),
        ("file_type",    PayloadSchemaType.KEYWORD),
        ("content_type", PayloadSchemaType.KEYWORD),
        ("page_num",     PayloadSchemaType.INTEGER),
        ("created_at",   PayloadSchemaType.DATETIME),
    ]:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )
