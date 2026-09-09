# Minimal JARVIS container (zero dependencies beyond the Python stdlib).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Copy the app in (no build step / no requirements).
COPY main.py .
COPY jarvis/ ./jarvis/
COPY public/ ./public/

EXPOSE 8000

# Healthcheck placeholder (the server has no health route yet).
HEALTHCHECK NONE

CMD ["python", "main.py"]
