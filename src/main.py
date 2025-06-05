from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import health, upload

app = FastAPI(
    title="Tagging Engine API",
    description="API for the tagging engine application",
    version="0.1.0",
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


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Tagging Engine API is running"}
