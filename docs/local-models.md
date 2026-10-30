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

## Manual model inventory and version snapshots

The helper script uses only Python's standard library. Run it from the
repository root; no package installation is required:

~~~bash
python3 scripts/model-catalog.py list
python3 scripts/model-catalog.py inspect qwen3.5:2b
python3 scripts/model-catalog.py snapshot qwen3.5:2b
python3 scripts/model-catalog.py catalog
~~~

- `list` shows every model currently installed in the Ollama instance,
  including its exact digest and runtime-reported details.
- `inspect` asks Ollama for detailed model metadata, including family,
  parameter size, quantization, and model information when exposed by that
  model/runtime combination.
- `snapshot` records the model identity and Ollama runtime version in
  `~/.FlossWare/models/catalog/model-catalog.json`.
- `catalog` prints the saved snapshots.

The catalog records the human-friendly model reference (including its tag),
Ollama's immutable content digest, model metadata, capture time, and Ollama
runtime version. Tags are mutable names; the digest is the artifact identity.
A later pull that changes the digest is a distinct model artifact and should
be evaluated as such. The script preserves historical snapshots and does not
download, remove, or modify model artifacts.

For another Ollama endpoint or model root, use `--url` and
`--models-dir`:

~~~bash
python3 scripts/model-catalog.py --url http://127.0.0.1:11434 \\
  --models-dir "$HOME/.FlossWare/models" snapshot qwen3.5:2b
~~~

Catalog metadata lives under the same model root, but outside Ollama's
`manifests/` and `blobs/` directories. Back up the catalog along with any
evaluation records that refer to these model digests.


## Compare models on Thompson outcome judgments

The repository includes a small synthetic regression suite covering clear
success, material factual correction, preference correction followed by
successful completion, insufficient feedback, explicit task failure, and a
reasonable clarification followed by completion.

Run several already-installed models against exactly the same cases:

~~~bash
python3 scripts/compare-models.py \\
  --models qwen3.5:2b,gemma4:e2b \\
  --cases tests/fixtures/thompson-outcome-cases.jsonl
~~~

The runner processes models and cases sequentially. It does not parallelize
inference, download models, call cloud APIs, or write to Thompson. Defaults
are temperature 0, seed 42, one 4K context, and one request at a time. It records
wall-clock latency separately from Ollama's inference duration, prompt/output
token counts when Ollama provides them, raw response,
parsed judgment, model digest, Ollama version, and generation settings.

Detailed JSONL results are written by default beneath
`~/.FlossWare/models/runs/`. The output contains the input's expected labels
for local scoring, but those labels are deliberately excluded from the prompt
sent to the model. Treat result files as sensitive if the cases contain real
transcripts.

The summary reports JSON parse rate and accuracy against the synthetic
expected outcomes. Accuracy is calculated over parseable judgments; inspect
the parse rate too, since a model that emits invalid JSON should not get a
free pass. This small fixture is a smoke/regression suite, not a statistically
meaningful benchmark or proof that a judge is reliable.

### Use your own sanitized Thompson samples

Create a JSONL file with one JSON object per line. Each case needs a unique
`id` and a `transcript` array. Each message has a `role` and string
`content`. Optionally include `expected_outcome` with `success`,
`failure`, or `unknown` for a labeled test set. Do not include expected
labels unless you have independently established them; they are used only
for local comparison and never sent to the judge.

Example:

~~~json
{"id":"turn-001","transcript":[{"role":"user","content":"Summarize the issue."},{"role":"assistant","content":"Here is the summary."}],"expected_outcome":"unknown"}
~~~

Then run:

~~~bash
python3 scripts/compare-models.py \\
  --models qwen3.5:2b,gemma4:e2b,qwen3.5:4b \\
  --cases "$HOME/tmp/thompson-samples.jsonl" \\
  --num-ctx 4096 \\
  --num-predict 512 \\
  --keep-alive 2m
~~~

You can change the generation settings with `--temperature`, `--seed`,
`--num-ctx`, `--num-predict`, `--keep-alive`, and `--timeout`.
Use exact installed Ollama tags in `--models`. Results preserve the resolved
digest so comparisons remain attributable to a specific artifact.

The runner uses the evidence-based rubric in
[`prompts/thompson-outcome-judge.md`](../prompts/thompson-outcome-judge.md).
It judges the completed interaction, distinguishes presentation corrections
from material errors, and returns `unknown` when the transcript is
insufficient. It does not write labels back to source transcripts or update
Thompson statistics.
