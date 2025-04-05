FROM python:3.9-slim

# Imposta la directory di lavoro
WORKDIR /app

# Installa le dipendenze di sistema necessarie per pillow-heif
RUN apt-get update && apt-get install -y libheif1 libheif-dev && rm -rf /var/lib/apt/lists/*

# Copia il file delle dipendenze e installa i pacchetti Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY app.py .

# Espone la porta su cui gira il servizio
EXPOSE 8001

# Avvia l'applicazione con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
