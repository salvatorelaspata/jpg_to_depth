import torch
import cv2
import numpy as np
from PIL import Image
import os
import gc  # Garbage collector
import sys

def process_images(input_dir, output_dir_depth, midas, transform, device):
    """
    Processa le immagini rimuovendo lo sfondo e generando mappe di profondità.

    Args:
        input_dir (str): Percorso della directory con le immagini di input
        output_dir_no_bg (str): Percorso della directory per le immagini senza sfondo
        output_dir_depth (str): Percorso della directory per le mappe di profondità
    """

    # verifica se la directory di output esiste
    if not os.path.exists(output_dir_depth):
        raise FileNotFoundError(f"Directory {output_dir_depth} non trovata")

    image_files = [
        f
        for f in os.listdir(input_dir)  # <-- input_dir è original_jpg
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    i = 1
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        try:
            # Carica l'immagine
            input_image = Image.open(input_path)
            # Genera mappa di profondità
            depth_image = create_depth_map(input_path, midas, transform, device)

            # Prepara i percorsi di output
            output_path_depth = os.path.join(output_dir_depth, image_file)

            # Salva le immagini
            depth_image.save(output_path_depth)

            print(f"Depth image {i}/{len(image_files)}: {image_file}")
            i += 1
            # Libera memoria
            del input_image, depth_image
            gc.collect()

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Pulizia finale
    del midas
    torch.cuda.empty_cache()
    gc.collect()

def create_depth_map(image_path, midas, transform, device):
    """
    Genera una mappa di profondità utilizzando un modello MiDaS pre-caricato.

    Args:
        image_path (str): Percorso dell'immagine di input
        midas (torch.nn.Module): Modello MiDaS pre-caricato
        transform (torchvision.transforms): Trasformazioni pre-definite
        device (torch.device): Dispositivo di calcolo

    Returns:
        PIL.Image: Mappa di profondità normalizzata
    """
    # Carica l'immagine
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Converti e prepara l'input
    input_batch = transform(Image.fromarray((img * 255).astype(np.uint8))).unsqueeze(0)

    # Genera profondità
    with torch.no_grad():
        prediction = midas(input_batch.to(device))
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Normalizza
    depth_numpy = depth.cpu().numpy()
    depth_normalized = cv2.normalize(
        depth_numpy, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Libera memoria
    del img, input_batch, prediction, depth, depth_numpy
    torch.cuda.empty_cache()
    gc.collect()

    return Image.fromarray(depth_normalized)


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_images(input_folder, output_folder)
