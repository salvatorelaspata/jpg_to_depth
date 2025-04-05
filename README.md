# HEIC to JPG Converter

This project provides a simple service to convert HEIC images to JPG format using FastAPI.

## Project Structure

- `bruno_test/bruno.json`: Configuration file for the project.
- `bruno_test/convert.bru`: Script to test the conversion endpoint.
- `compose.yml`: Docker Compose file to set up the service.
- `Dockerfile`: Dockerfile to build the service image.
- `requirements.txt`: Python dependencies for the project.
- `app.py`: FastAPI application to handle the image conversion.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Installation

1. Clone the repository:
  ```sh
  git clone <repository_url>
  cd heic_to_jpg
  ```

2. Build and start the service using Docker Compose:
  ```sh
  docker-compose up --build
  ```

### Usage

1. Send a POST request to `http://localhost:8001/convert` with a HEIC file in the `file` field of a multipart form.

### Example

You can use the provided `bruno_test/convert.bru` script to test the conversion endpoint.

### Dependencies

- FastAPI
- Uvicorn
- Pillow
- Pillow-HEIF
- Python-Multipart

### License

This project is licensed under the MIT License.