from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import uvicorn
import base64
from io import BytesIO
from PIL import Image

# Import numerical modules
from numeric07_lu import LUP, SolveLinearSystemLUP, LeastSquares
from numeric08_qr import QRAlgorithm
from numeric09_eigenvalue import PowerIteration, RayleighQuotientIteration
from numeric04_imagecompression import Compress, Decompress
from app.numeric04_linearsolver import ComputeSVD, PseudoInverse, LinearSolve
from app.numeric05_leastsquares import QR, BackSubstitution, LeastSquares as LS_LeastSquares
from numeric06_svd import ComputeSVD as ComputeSVD6

app = FastAPI(
    title="Numeric Operations API",
    description="A web service exposing endpoints for matrix operations and image compression.",
    version="1.0.0"
)

# ---------- Request Models ----------

class LURequest(BaseModel):
    matrix: list  # A two-dimensional list

class SolveRequest(BaseModel):
    matrix: list  # Two-dimensional list
    vector: list  # One-dimensional list

class LeastSquaresRequest(BaseModel):
    matrix: list
    vector: list

class EigenRequest(BaseModel):
    matrix: list

class IterationRequest(BaseModel):
    matrix: list
    vector: list

class SVDRequest(BaseModel):
    matrix: list

# ---------- Endpoints ----------

@app.post("/api/lu", summary="LU Decomposition")
def lu_decomposition(req: LURequest):
    try:
        A = np.array(req.matrix, dtype=float)
        L, U, P = LUP(A)
        return {"L": L.tolist(), "U": U.tolist(), "P": P.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/solve", summary="Solve Linear System via LU")
def solve_linear_system(req: SolveRequest):
    try:
        A = np.array(req.matrix, dtype=float)
        b = np.array(req.vector, dtype=float).reshape(-1, 1)
        x = SolveLinearSystemLUP(A, b)
        return {"solution": x.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/least_squares", summary="Least Squares via LU")
def least_squares(req: LeastSquaresRequest):
    try:
        A = np.array(req.matrix, dtype=float)
        b = np.array(req.vector, dtype=float)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        x = LeastSquares(A, b)
        return {"least_squares_solution": x.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/qr", summary="Eigenvalues via QR Algorithm")
def qr_algorithm(req: EigenRequest):
    try:
        A = np.array(req.matrix, dtype=float)
        evals = QRAlgorithm(A)
        return {"eigenvalues": evals.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/power_iteration", summary="Power Iteration")
def power_iteration(req: IterationRequest):
    try:
        A = np.array(req.matrix, dtype=float)
        v = np.array(req.vector, dtype=float).reshape(-1, 1)
        lambda_list = PowerIteration(A, v)
        return {"power_iteration_eigenvalues": lambda_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/rayleigh_iteration", summary="Rayleigh Quotient Iteration")
def rayleigh_iteration(req: IterationRequest):
    try:
        A = np.array(req.matrix, dtype=float)
        v = np.array(req.vector, dtype=float).reshape(-1, 1)
        lambda_list = RayleighQuotientIteration(A, v)
        return {"rayleigh_iteration_eigenvalues": lambda_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/svd", summary="SVD Decomposition")
def svd_decomposition(req: SVDRequest):
    try:
        A = np.array(req.matrix, dtype=float)
        U, Sigma, V = ComputeSVD6(A)
        return {"U": U.tolist(), "Sigma": Sigma.tolist(), "V": V.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/image/compress", summary="Image Compression using SVD")
async def image_compress(component_count: int, file: UploadFile = File(...)):
    """
    Accepts an image file (PNG, JPEG, etc.) and an integer parameter ‘component_count’.
    Returns the compression ratio and the decompressed image (as a Base64‐encoded PNG).
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("L")  # grayscale
        img_array = np.array(image, dtype=float)
        U, singular_values, Vt, compression_ratio = Compress(img_array, component_count)
        decompressed = Decompress(U, singular_values, Vt)
        # Clip and convert to uint8 for image display
        decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)
        decompressed_image = Image.fromarray(decompressed)
        buffer = BytesIO()
        decompressed_image.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return {"compression_ratio": compression_ratio, "decompressed_image": img_base64}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", summary="Welcome Endpoint")
def root():
    return {"message": "Welcome to the Numeric Operations API. Visit /docs for API documentation."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
