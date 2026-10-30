# Local model evaluation

The local runtime is optional. The evaluation package and its existing
zero-dependency core continue to work without Ollama or model weights.

## Model storage

Use the established FlossWare model directory as the Ollama model store:

~~~text
~/.FlossWare/models/
~~~

~~~bash
mkdir -p "$HOME/.FlossWare/models"
OLLAMA_MODELS="$HOME/.FlossWare/models" ollama serve
~~~

For a systemd-managed Ollama service, configure the service environment rather
than starting a second server:

~~~bash
sudo systemctl edit ollama
~~~

Add an absolute path to the existing user's model directory:

~~~ini
[Service]
Environment="OLLAMA_MODELS=/absolute/path/to/home/.FlossWare/models"
~~~

A system service may run as a dedicated account. Check it with
`systemctl show ollama -p User` and grant that account access to the model
directory as appropriate. Do not use `%h` blindly: in a system service it
can refer to the service account's home, not the interactive user's home.

Then reload and restart the service:

~~~bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
~~~

Verify the effective service configuration and model location before pulling
large models. Do not move or delete an existing Ollama model store until its
contents have been migrated and verified.

## Install and fetch candidate models

Install Ollama using its official installation instructions, then fetch one
small model at a time. Initial candidates for a laptop with 64 GB RAM and
integrated Intel Arc graphics:

~~~bash
ollama pull qwen3.5:2b
ollama pull gemma4:e2b
~~~

Optional larger comparisons:

~~~bash
ollama pull qwen3.5:4b
ollama pull gemma4:e4b
~~~

Model tags and availability can change. Confirm the tags in the Ollama library
before pulling. Keep only one evaluation model loaded at a time initially.

## Use the backend

~~~python
import asyncio

from evaluation_ai import ChatMessage
from evaluation_ai.backends.ollama import OllamaBackend


async def main():
    backend = OllamaBackend(
        keep_alive="5m",
        max_concurrency=1,
        timeout=300,
    )
    response = await backend.chat(
        [ChatMessage(role="user", content="Return a JSON label for this example.")],
        model="qwen3.5:2b",
        temperature=0.0,
        num_ctx=4096,
    )
    print(response.content)
    print(response.usage)


asyncio.run(main())
~~~

The backend uses Ollama's local /api/chat endpoint, disables streaming,
runs blocking HTTP I/O off the event loop, and limits concurrent inference
requests to one by default. keep_alive="0" asks Ollama to unload the model
after the request; use a short duration such as "5m" for a batch.

## Laptop resource policy

Start conservatively:

- One model loaded at a time.
- One inference request at a time.
- 4K context initially; increase only when transcript evidence requires it.
- Run historical labeling in batches and allow the model to unload between runs.
- Benchmark CPU and Intel GPU paths rather than assuming GPU offload is active.
- Do not run local inference continuously in the background on a development
  laptop.

The backend accepts num_ctx, num_predict, top_k, top_p, min_p, repeat_penalty,
seed, and stop as Ollama generation options. The runtime owns model loading and
device placement; evaluation-ai does not download model weights or configure
GPU drivers.
