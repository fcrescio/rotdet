# Evaluate MCP Server (rotdet-mcp)

RotDet ships an MCP (Model Context Protocol) server that exposes the evaluation
pipeline as a tool. This lets agentic clients call the evaluator programmatically
over stdio and receive JSON metrics.

The server is implemented in `scripts/rotdet_mcp_server.py` and provides a single
tool, `evaluate_rotdet`, which wraps `run_evaluation` from `evaluate_rotdet.py`.【F:scripts/rotdet_mcp_server.py†L1-L92】

---

## ✅ What you get

* **Tool name**: `evaluate_rotdet`
* **Transport**: stdio (MCP server launched as a local process)
* **Return value**: JSON string with evaluation results (accuracy + confusion matrix)

---

## 🧩 Prerequisites

1. Install project dependencies (see README for full install options).
2. Make sure the CLI entrypoint is available:

```bash
uv run rotdet-mcp
```

The `rotdet-mcp` command points at `rotdet_mcp_server:main` in `pyproject.toml`.【F:pyproject.toml†L27-L52】

---

## 🚀 Starting the server

Run the MCP server in a terminal:

```bash
uv run rotdet-mcp
```

This launches the MCP stdio server and waits for client connections. The server
name is `rotdet-eval`.【F:scripts/rotdet_mcp_server.py†L11-L35】

---

## 🔧 Tool interface

### Tool name

```
evaluate_rotdet
```

### Input schema

The server accepts a JSON object with the following fields (all optional):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `repo_id` | string | `fcrescio/rotdet` | HF repo ID or local checkpoint path |
| `filename` | string | `model.safetensors` | File inside repo (ignored for local path) |
| `dataset` | string | `nielsr/funsd` | Hugging Face dataset name |
| `input_size` | integer | `128` | Image size fed to model |
| `config` | string \| null | `null` | HF dataset config |
| `split` | string | `test` | Dataset split |
| `streaming` | boolean | `false` | Use streaming dataset mode |
| `max_samples` | integer | `0` | Limit samples (0 = no limit) |
| `pages_per_doc` | integer \| null | `null` | Multi-page docs per example |
| `batch_size` | integer | `64` | Evaluation batch size |
| `num_workers` | integer | `2` | DataLoader workers |
| `rotate_prob` | number | `0.5` | Rotation probability in dataset wrapper |
| `snapshot_dirs` | array<string> \| null | `null` | Snapshot directories |
| `fail_log` | string \| null | `null` | JSONL output for failures |
| `save_fail_images` | string \| null | `null` | Folder for failed images |
| `model` | string | `c4net` | Model architecture (from `models.available_model_names`) |
| `model_kwargs` | object \| null | `null` | Extra model kwargs |

These map 1:1 to `run_evaluation` arguments.【F:scripts/rotdet_mcp_server.py†L14-L66】

### Output

The tool returns a JSON string. Example payload:

```json
{"acc": 0.982, "confusion_matrix": [[491, 9], [8, 492]]}
```

---

## 📦 Example tool call

```json
{
  "tool": "evaluate_rotdet",
  "arguments": {
    "repo_id": "fcrescio/rotdet",
    "filename": "model.safetensors",
    "dataset": "nielsr/funsd",
    "split": "test",
    "max_samples": 2000
  }
}
```

---

## 🧭 Example client configurations

> **Note**: Most MCP-enabled clients use the same stdio configuration: a command
> plus arguments. Adjust the config location to your client.

### Mistral Vibe (example)

```toml
[[mcp_servers]]
name = "rotdet-eval"
transport = "stdio"
command = "uv"
args = ["run", "rotdet-mcp"]
```

### Claude Desktop (example)

```json
{
  "mcpServers": {
    "rotdet-eval": {
      "command": "uv",
      "args": ["run", "rotdet-mcp"]
    }
  }
}
```

### Continue / VS Code (example)

```json
{
  "mcpServers": [
    {
      "name": "rotdet-eval",
      "command": "uv",
      "args": ["run", "rotdet-mcp"]
    }
  ]
}
```

---

## ✅ Troubleshooting

* **`model_kwargs must be an object`**: Make sure you pass an object (not a list
  or string) for `model_kwargs`.【F:scripts/rotdet_mcp_server.py†L14-L19】
* **Hugging Face auth**: If you need private datasets/models, set
  `HF_TOKEN` in the environment for your MCP server process.
* **Dataset config issues**: Use the `config` field to explicitly select a
  dataset configuration.
