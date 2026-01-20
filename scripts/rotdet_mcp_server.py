import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from evaluate_rotdet import run_evaluation
from models import available_model_names

SERVER_NAME = "rotdet-eval"


def _build_args(arguments):
    model_kwargs = arguments.get("model_kwargs")
    if model_kwargs is not None and not isinstance(model_kwargs, dict):
        raise ValueError("model_kwargs must be an object")
    return {
        "repo_id": arguments.get("repo_id", "fcrescio/rotdet"),
        "filename": arguments.get("filename", "model.safetensors"),
        "dataset": arguments.get("dataset", "nielsr/funsd"),
        "input_size": arguments.get("input_size", 128),
        "config": arguments.get("config"),
        "split": arguments.get("split", "test"),
        "streaming": arguments.get("streaming", False),
        "max_samples": arguments.get("max_samples", 0),
        "pages_per_doc": arguments.get("pages_per_doc"),
        "batch_size": arguments.get("batch_size", 64),
        "num_workers": arguments.get("num_workers", 2),
        "rotate_prob": arguments.get("rotate_prob", 0.5),
        "snapshot_dirs": arguments.get("snapshot_dirs"),
        "fail_log": arguments.get("fail_log"),
        "save_fail_images": arguments.get("save_fail_images"),
        "model": arguments.get("model", "c4net"),
        "model_kwargs": model_kwargs,
    }


server = Server(SERVER_NAME)


@server.list_tools()
async def list_tools():
    model_choices = available_model_names()
    return [
        Tool(
            name="evaluate_rotdet",
            description="Evaluate rotdet checkpoints on a Hugging Face dataset and return metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string", "default": "fcrescio/rotdet"},
                    "filename": {"type": "string", "default": "model.safetensors"},
                    "dataset": {"type": "string", "default": "nielsr/funsd"},
                    "input_size": {"type": "integer", "default": 128},
                    "config": {"type": ["string", "null"]},
                    "split": {"type": "string", "default": "test"},
                    "streaming": {"type": "boolean", "default": False},
                    "max_samples": {"type": "integer", "default": 0},
                    "pages_per_doc": {"type": ["integer", "null"]},
                    "batch_size": {"type": "integer", "default": 64},
                    "num_workers": {"type": "integer", "default": 2},
                    "rotate_prob": {"type": "number", "default": 0.5},
                    "snapshot_dirs": {"type": ["array", "null"], "items": {"type": "string"}},
                    "fail_log": {"type": ["string", "null"]},
                    "save_fail_images": {"type": ["string", "null"]},
                    "model": {"type": "string", "enum": model_choices, "default": "c4net"},
                    "model_kwargs": {"type": ["object", "null"]},
                },
                "additionalProperties": False,
            },
        )
    ]


@server.call_tool()
async def call_tool(name, arguments):
    if name != "evaluate_rotdet":
        raise ValueError(f"Unknown tool: {name}")
    args = _build_args(arguments or {})
    result = await asyncio.to_thread(run_evaluation, **args)
    payload = json.dumps(result, ensure_ascii=False)
    return [TextContent(type="text", text=payload)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
