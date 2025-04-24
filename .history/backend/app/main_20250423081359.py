from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import uvicorn
from uuid import uuid4
import time

from backend.app.core.config import get_settings
from backend.app.core.logger import setup_logger
from backend.app.api.endpoints import router as api_router

# Initialize settings
settings = get_settings()

# Setup logging
logger = setup_logger()

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Bayesian Optimization Engine API",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
    }

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    error_id = str(uuid4())
    logger.error(f"Unhandled exception occurred: {exc}", exc_info=True, extra={"error_id": error_id})
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": str(exc) if settings.APP_ENV != "production" else "An unexpected error occurred",
        },
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
    ) 
