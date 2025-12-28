FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify Rust installation
RUN rustc --version && cargo --version

# Set working directory
WORKDIR /app

# Copy source code
# NOTE: For HF Spaces, core/ and optimizer/ must be synced from main project before deploy
COPY . .

# Install maturin and build Rust extension
RUN pip install --no-cache-dir "maturin>=1.0,<2.0"
RUN cd rust && maturin build --release --out /app/wheels && ls -la /app/wheels/

# Install Rust wheel and verify
RUN pip install --no-cache-dir /app/wheels/sparse_core-*.whl
RUN python -c "import sparse_core; print('Rust module loaded:', sparse_core)"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Set environment variables
ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Expose Gradio default port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
