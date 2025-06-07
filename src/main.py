from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import health, upload, query
from src.config import KEYFRAMES_DIR, CROPPED_KEYFRAMES_DIR, MASKED_KEYFRAMES_DIR
import shutil


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Clean temporary keyframe directories
    directories_to_clean = [KEYFRAMES_DIR, CROPPED_KEYFRAMES_DIR, MASKED_KEYFRAMES_DIR]

    for directory in directories_to_clean:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(exist_ok=True)

    yield


app = FastAPI(
    title="Tagging Engine API",
    description="API for the tagging engine application",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(query.router)


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Tagging Engine API is running"}
