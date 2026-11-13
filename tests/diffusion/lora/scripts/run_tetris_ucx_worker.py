#!/usr/bin/env python
"""Launch the external TetrisLoRA UCX worker configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as source:
        config = json.load(source)
    required = (
        "role",
        "listen_host",
        "listen_port",
        "physical_gpu",
        "adapter_path",
        "ucx_bridge_path",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing worker configuration fields: {missing}")
    if config["role"] != "lora":
        raise SystemExit("this launcher only starts the external LoRA worker")
    os.environ["TETRIS_LORA_TRANSPORT"] = "ucx"
    os.environ["TETRIS_LORA_LORA_GPU"] = str(config["physical_gpu"])
    os.environ["TETRIS_LORA_ADAPTER_PATH"] = str(config["adapter_path"])
    os.environ["TETRIS_LORA_UCX_BRIDGE"] = str(config["ucx_bridge_path"])
    os.environ["UCX_INSTALL_PATH"] = str(
        config.get("ucx_install_path", "/opt/hpcx/ucx")
    )
    raise SystemExit(
        "UCX external worker control-plane launcher is configuration-only until "
        "the peer handshake adapter is selected; refusing a partial data-plane run"
    )


if __name__ == "__main__":
    main()
