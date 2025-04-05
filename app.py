from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import pillow_heif
from PIL import Image

# Registra il supporto per i file HEIC in Pillow
pillow_heif.register_heif_opener()

app = FastAPI()


@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".heic"):
        raise HTTPException(
            status_code=400, detail="Il file deve essere un'immagine HEIC."
        )
    try:
        contents = await file.read()
        # Apri l'immagine HEIC
        image = Image.open(io.BytesIO(contents))
        # Converti in RGB per salvare in JPEG
        image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG")
        output.seek(0)
        return StreamingResponse(output, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore durante la conversione: {e}"
        )


# create / endpoint
@app.get("/")
async def read_root():
    return {"welcome": "HEIC to JPEG converter"}


# implement 404 for all other routes
@app.get("/{path:path}")
async def read_path(path: str):
    raise HTTPException(status_code=404, detail="Route not found for path: " + path)
