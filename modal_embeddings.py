import socket
import subprocess
from pathlib import Path

import modal

GPU_CONFIG = modal.gpu.L40S()
MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32
DOCKER_IMAGE = (
    # "ghcr.io/huggingface/text-embeddings-inference:hopper-1.6"  # Hopper 90 for H100s (marked experimental)
    "ghcr.io/huggingface/text-embeddings-inference:89-1.6"  # Lovelace 89 for L40S
    # "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4
)

DATA_PATH = Path("/data/dataset.jsonl")

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
]


def spawn_server() -> subprocess.Popen:
    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


def download_model():
    # Wait for server to start. This downloads the model weights when not present.
    spawn_server().terminate()


app = modal.App("self-expansion-embeddings")

tei_image = (
    modal.Image.from_registry(
        DOCKER_IMAGE,
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, gpu=GPU_CONFIG)
    .pip_install("httpx")
)

with tei_image.imports():
    from httpx import AsyncClient


@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    # Use up to 20 GPU containers at once.
    concurrency_limit=20,
    # Allow each container to process up to 10 batches at once.
    allow_concurrent_inputs=10,
)
class TextEmbeddingsInference:
    @modal.enter()
    def setup_server(self):
        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000")

    @modal.exit()
    def teardown_server(self):
        self.process.terminate()

    @modal.method()
    async def embed(self, inputs):
        resp = await self.client.post("/embed", json={"inputs": inputs})
        resp.raise_for_status()
        outputs = resp.json()

        return outputs


image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "pandas", "db-dtypes", "tqdm"
)


@app.function(
    image=image,
)
def embed(data):
    model = TextEmbeddingsInference()

    return model.embed.remote(data)[0]


@app.local_entrypoint()
def main(text: str = "hello"):
    embedding = embed.remote([text])
    print(text, embedding[:10], "...")
