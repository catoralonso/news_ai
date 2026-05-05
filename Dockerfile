FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/user/.local/bin:/root/.local/bin:$PATH"
WORKDIR /app
COPY --from=builder /root/.local /home/user/.local
RUN useradd -m -u 1000 user && \
    chown -R user:user /app /home/user/.local && \
    mkdir -p /app/data/articles /app/data/social_media_output /app/data/clickstream && \
    chown -R user:user /app/data && \
    mkdir -p /home/user/.cache && \
    chown -R user:user /home/user
COPY --chown=user:user . .         
USER user
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

