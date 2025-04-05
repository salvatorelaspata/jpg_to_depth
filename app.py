import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import src.depth_map as depth_map
# model
import torch
import torchvision.transforms as transforms

app = FastAPI()

# Caricamento del modello all'avvio
# Prepara il modello MiDaS una sola volta
#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Prepara il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Trasformazioni per preparare l'immagine
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Apri l'immagine HEIC
        image = Image.open(io.BytesIO(contents))
        # Converti in RGB per salvare in JPEG
        image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG")
        output.seek(0)
        # Salva l'immagine JPEG in memoria
        folder_original_jpg = "original_jpg"
        folder_depth_map = "depth_map"
        # Crea le cartelle se non esistono
        if not os.path.exists(folder_original_jpg):
            os.makedirs(folder_original_jpg)
        if not os.path.exists(folder_depth_map):
            os.makedirs(folder_depth_map)
        # Salva l'immagine JPEG nella cartella
        image.save(f"{folder_original_jpg}/{file.filename}.jpg", format="JPEG")
        # Salva l'immagine JPEG in memori
        depth_map.process_images(folder_original_jpg, folder_depth_map, midas, transform, device)
        output_converted = io.BytesIO()
        # Apri l'immagine di profondità
        depth_image = Image.open(f"{folder_depth_map}/{file.filename}.jpg")
        # Converti in JPEG
        depth_image = depth_image.convert("RGB")
        depth_image.save(output_converted, format="JPEG")
        output_converted.seek(0)
        # Restituisci l'immagine JPEG come risposta
        return StreamingResponse(
            output_converted, media_type="image/jpeg", headers={"Content-Disposition": "inline; filename=depth_map.jpg"}
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=f"File non trovato: {e}"
        )
    except OSError as e:
        raise HTTPException(
            status_code=400, detail=f"Errore durante la conversione: {e}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Errore durante la conversione: {e}"
        )
    except TypeError as e:
        raise HTTPException(
            status_code=400, detail=f"Errore durante la conversione: {e}"
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=403, detail=f"Errore durante la conversione: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore durante la conversione: {e}"
        )

# create / endpoint
@app.get("/")
async def read_root():
    return {"welcome": "JPEG to DEPTH conversion API"}


# implement 404 for all other routes
@app.get("/{path:path}")
async def read_path(path: str):
    raise HTTPException(status_code=404, detail="Route not found for path: " + path)
