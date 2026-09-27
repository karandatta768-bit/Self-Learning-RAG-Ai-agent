from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Self-Learning AI Agent", version="0.1.0")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
