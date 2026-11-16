FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-core \
    fonts-lohit-deva \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as UID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

WORKDIR $HOME/app

COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . $HOME/app
RUN mkdir -p $HOME/app/data/jobs $HOME/app/uploads $HOME/app/work $HOME/app/outputs && \
    chown -R user:user $HOME/app

USER user

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
