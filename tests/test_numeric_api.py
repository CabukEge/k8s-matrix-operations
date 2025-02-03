# tests/test_api.py
from fastapi.testclient import TestClient
from main import app
import numpy as np
from PIL import Image
import io

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Numeric API"}

def test_add_numbers():
    response = client.post("/add", json={"x": 2, "y": 3})
    assert response.status_code == 200
    assert response.json() == {"result": 5}

def test_subtract_numbers():
    response = client.post("/subtract", json={"x": 5, "y": 3})
    assert response.status_code == 200
    assert response.json() == {"result": 2}

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_process_numeric_image():
    # Create a test image with known numeric content
    img = Image.new('L', (28, 28), color=0)  # Create a black image
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    response = client.post("/process", files=files)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_image_format():
    # Test with invalid file format
    invalid_data = b"not an image"
    files = {'file': ('test.txt', invalid_data, 'text/plain')}
    response = client.post("/process", files=files)
    assert response.status_code == 400

def test_image_size():
    # Create an image with invalid size
    img = Image.new('L', (100, 100), color=0)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    response = client.post("/process", files=files)
    # Should either return 400 or 200 depending on your implementation
    assert response.status_code in [200, 400]

def test_batch_process():
    # Create multiple test images
    img = Image.new('L', (28, 28), color=0)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = [('files', ('test1.png', img_byte_arr, 'image/png')),
             ('files', ('test2.png', img_byte_arr, 'image/png'))]
    
    response = client.post("/batch-process", files=files)
    assert response.status_code == 200
    assert isinstance(response.json(), list)