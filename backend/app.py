from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router

app = FastAPI(
    title="Groundwater Analysis API",
    description="Backend service for downloading Sentinel TIFFs and predicting groundwater status",
    version="1.0.0"
)

# Allow frontend from localhost:3000 to access the backend
origins = [
    "http://localhost:3000",  # your frontend dev server
    # Add more domains here when needed
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # or use ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the prediction route
app.include_router(predict_router)
