# tests/test_numeric_api.py
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app
import numpy as np
from PIL import Image
import io
import base64

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Numeric Operations API. Visit /docs for API documentation."}

def test_lu_decomposition():
    matrix = [[4, 3], [6, 3]]
    response = client.post("/api/lu", json={"matrix": matrix})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "L" in result
    assert "U" in result
    assert "P" in result

def test_solve_linear_system():
    matrix = [[4, 3], [6, 3]]
    vector = [10, 12]
    response = client.post("/api/solve", json={"matrix": matrix, "vector": vector})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "solution" in result

def test_least_squares():
    matrix = [[1, 1], [1, 2], [1, 3]]
    vector = [2, 4, 6]
    response = client.post("/api/least_squares", json={"matrix": matrix, "vector": vector})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "least_squares_solution" in result

def test_qr_algorithm():
    matrix = [[4, 3], [6, 3]]
    response = client.post("/api/qr", json={"matrix": matrix})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "eigenvalues" in result

def test_power_iteration():
    matrix = [[4, 3], [6, 3]]
    vector = [1, 1]
    response = client.post("/api/power_iteration", json={"matrix": matrix, "vector": vector})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "power_iteration_eigenvalues" in result

def test_rayleigh_iteration():
    matrix = [[4, 3], [6, 3]]
    vector = [1, 1]
    response = client.post("/api/rayleigh_iteration", json={"matrix": matrix, "vector": vector})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "rayleigh_iteration_eigenvalues" in result

def test_svd_decomposition():
    matrix = [[4, 3], [6, 3]]
    response = client.post("/api/svd", json={"matrix": matrix})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "U" in result
    assert "Sigma" in result
    assert "V" in result

def test_image_compression():
    # Create a test image
    img = Image.new('L', (100, 100), color=128)  # grayscale image
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    response = client.post("/api/image/compress?component_count=2", files=files)
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 200
    result = response.json()
    assert "compression_ratio" in result
    assert "decompressed_image" in result
    
    # Test if the decompressed image is a valid base64 string
    try:
        img_data = base64.b64decode(result["decompressed_image"])
        img = Image.open(io.BytesIO(img_data))
        assert img.mode == "L"  # Should be grayscale
    except Exception as e:
        assert False, f"Invalid image data: {str(e)}"

def test_invalid_matrix():
    # Test with non-square matrix for eigenvalue computation
    matrix = [[1, 2, 3], [4, 5, 6]]
    response = client.post("/api/qr", json={"matrix": matrix})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 400

def test_invalid_dimensions():
    # Test with incompatible dimensions in linear system
    matrix = [[1, 2], [3, 4]]
    vector = [1, 2, 3]  # Wrong dimension
    response = client.post("/api/solve", json={"matrix": matrix, "vector": vector})
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 400

def test_invalid_image_format():
    # Test with invalid file format
    invalid_data = b"not an image"
    files = {'file': ('test.txt', invalid_data, 'text/plain')}
    response = client.post("/api/image/compress?component_count=2", files=files)
    print("Response JSON:", response.json())  # Debug output
    assert response.status_code == 400
    
def test_matrix_operation_accuracy():
    # Test if LU decomposition actually produces L*U = P*A
    matrix = [[4, 3], [6, 3]]
    response = client.post("/api/lu", json={"matrix": matrix})
    assert response.status_code == 200
    result = response.json()
    
    # Convert results back to numpy arrays for matrix multiplication
    L = np.array(result["L"])
    U = np.array(result["U"])
    P = np.array(result["P"])
    A = np.array(matrix)
    
    # Check if L*U = P*A
    LU = L @ U
    PA = P @ A
    
    # Check if the difference is smaller than a tolerance
    assert np.allclose(LU, PA, rtol=1e-10)

'''
def test_matrix_size_limit():
    # Create a large matrix that should exceed processing limits
    large_size = 1000
    large_matrix = [[1.0] * large_size for _ in range(large_size)]
    
    response = client.post("/api/svd", json={"matrix": large_matrix})
    print("Response JSON:", response.json())  # Debug output
    
    # Should return 400 Bad Request due to size limitations
    assert response.status_code == 400
    
    # Optionally check for specific error message
    result = response.json()
    assert "detail" in result  # API should return error details
'''
