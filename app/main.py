from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.vertex_client import init_vertex, generate_image

app = FastAPI(title="Vertex Imagen Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    init_vertex()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/generate")
async def gen(body: dict):
    prompt = (body or {}).get("prompt", "").strip()
    size = (body or {}).get("size", "1024x1024")
    if not prompt:
        raise HTTPException(400, "prompt required")
    try:
        img = generate_image(prompt, size)
        return Response(content=img, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
