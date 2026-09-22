from fastapi import FastAPI

app = FastAPI(
    title="말씨 지도 (Malssi Map) API",
    description="음성 및 텍스트를 기반으로 지역 방언을 분석하는 API",
    version="0.1.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Malssi Map API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
