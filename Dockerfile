FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from copying code --
# Docker caches layers, so if only your code changes (not requirements.txt),
# this layer is reused instead of reinstalling everything on every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

# Both FastAPI (8000) and Streamlit (8501) run from this same image,
# just with different startup commands, set in docker-compose.yml
EXPOSE 8000 8501