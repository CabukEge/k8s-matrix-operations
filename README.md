# Numeric Operations API

This project provides a FastAPI‑based web service for my weekly tasks for numerics (therefore the variables are also in German)! that exposes endpoints for various numerical operations (LU, QR, SVD, etc.) and image compression via SVD. It demonstrates containerization with Docker, deployment on Kubernetes, and a CI/CD pipeline using GitHub Actions.

## Features

- **Matrix Operations:**  
  - LU Decomposition and solving linear systems  
  - Least squares solutions  
  - QR algorithm for eigenvalue computation  
  - Power iteration and Rayleigh quotient iteration  
  - SVD decomposition

- **Image Compression:**  
  - Compress and decompress grayscale images using SVD

- **DevOps Demo:**  
  - Docker multi‑stage builds  
  - Kubernetes manifests (Deployment, Service, Ingress, HPA, health probes)  
  - GitHub Actions for automated testing, security scanning, and deployment

## Project Structure


```
.
app/
    main.py
    numeric07_lu.py
    numeric08_qr.py
    numeric09_eigenvalue.py
    numeric04_imagecompression.py
    numeric04_linearsolver.py
    numeric05_leastsquares.py
    numeric06_svd.py
```

```
k8s/
    deployment.yaml
    service.yaml
    ingress.yaml
    hpa.yaml
```

```
.github/
    workflows/
        ci-cd.yaml
```

```
requirements.txt
Dockerfile
README.md
```
