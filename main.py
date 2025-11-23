from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import api
import uvicorn

app = FastAPI(
    title="LiDAR Semantic Segmentation API",
    description="DGCNN для сегментации облаков точек",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(api.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "LiDAR Segmentation API запущен! Перейди на /docs"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)