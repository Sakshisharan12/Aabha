# ==========================================
# Stage 1: Build the Next.js Frontend
# ==========================================
FROM node:18-alpine AS builder

WORKDIR /app/frontend

# Install dependencies first (for caching)
COPY frontend/package.json ./
RUN npm install

# Copy source code and build static export
COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Serve Backend & Static Frontend
# ==========================================
FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU build first
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy the built Next.js frontend static output
COPY --from=builder /app/frontend/out ./frontend/out

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3) or exit(1)"

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
