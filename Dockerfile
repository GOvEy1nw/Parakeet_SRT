# 1. Start from a lean PyTorch runtime image with CUDA support.
# This is much smaller than the full NeMo Framework image.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 2. Set default environment variables for the service.
# These can be overridden at runtime using the `docker run -e` flag.
ENV MODEL_NAME="parakeet-tdt-0.6b-v2.nemo"
ENV TIMESTAMP_LEVEL="segment"
# The models directory, host, and port are fixed for simplicity.
# Mount your host models directory to /app/models using `docker run -v`.
# The server will listen on 0.0.0.0:5000.

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the requirements file first to leverage Docker's build cache.
# This step only re-runs if the requirements.txt file changes.
COPY requirements.txt .

# 5. Install the Python packages.
# The `nemo_toolkit[asr]` package will pull in all necessary dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code into the container
# This includes main.py and client.py. The models will be mounted as a volume.
COPY . .

# 7. Expose the port that the Flask server will run on.
EXPOSE 5000

# 8. Define the command to run when the container starts.
# This starts our Flask server.
CMD ["python", "main.py"]