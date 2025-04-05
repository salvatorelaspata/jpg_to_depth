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
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
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
        # Open the uploaded image
        input_image = Image.open(io.BytesIO(contents))
        # Convert to RGB
        input_image = input_image.convert("RGB")
        
        # Process the image directly in memory
        depth_image = depth_map.process_single_image(input_image, midas, transform, device)
        
        # Return the depth image
        output = io.BytesIO()
        depth_image.save(output, format="JPEG")
        output.seek(0)
        
        return StreamingResponse(
            output, 
            media_type="image/jpeg", 
            headers={"Content-Disposition": "inline; filename=depth_map.jpg"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {str(e)}"
        )

# create / endpoint
@app.get("/")
async def read_root():
    return {"welcome": "JPEG to DEPTH conversion API"}


# implement 404 for all other routes
@app.get("/{path:path}")
async def read_path(path: str):
    raise HTTPException(status_code=404, detail="Route not found for path: " + path)
