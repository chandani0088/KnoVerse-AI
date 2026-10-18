from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI

from app.api.v1.documents import router as documents_router
from app.api.v1.search import router as search_router
from app.api.v1.ask import router as ask_router


app = FastAPI(
    title="KnoVerse AI",
    description="AI-powered Document Intelligence Platform",
    version="0.1.0",
)


app.include_router(
    documents_router,
    prefix="/api/v1"
)

app.include_router(
    search_router,
    prefix="/api/v1"
)
app.include_router(ask_router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "KnoVerse AI API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Use the OpenAPI version supported by Swagger UI file inputs.
    schema["openapi"] = "3.0.3"

    components = schema.get("components", {}).get("schemas", {})

    for component in components.values():
        properties = component.get("properties", {})

        for prop in properties.values():
            # Fix a single file field.
            if (
                prop.get("type") == "string"
                and prop.get("contentMediaType") == "application/octet-stream"
            ):
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"

            # Fix a list of files.
            items = prop.get("items", {})
            if (
                prop.get("type") == "array"
                and items.get("contentMediaType") == "application/octet-stream"
            ):
                items.pop("contentMediaType", None)
                items["type"] = "string"
                items["format"] = "binary"

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
    