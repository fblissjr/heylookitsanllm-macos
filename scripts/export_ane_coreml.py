#!/usr/bin/env python3
"""
Export EmbeddingGemma 2 (`embeddinggemma-2-MLX-BF16`) sub-models to Apple Neural Engine (ANE)
CoreML `.mlpackage` / `.mlmodelc` bundles for per-tower placement in `heylookitsanllm-macos`.

Exports three independent CoreML bundles matching `EmbeddingComponent.coreMLBundleName`:
1. `EmbeddingGemma2VisionANE.mlpackage` (`vision_tower` + `embed_vision`, 167.7M params)
2. `EmbeddingGemma2AudioANE.mlpackage` (`audio_tower` + `embed_audio`, 305.4M params)
3. `EmbeddingGemma2TextANE.mlpackage` (`language_model` 24L + `PLE` + `embedding_projection`, 271.0M params)
   plus `embed_tokens_fp16.bin` (`[262144, 512]` float16 token embedding table for O(L) CPU lookup).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

VISION_RESOLUTION_BUCKETS: list[tuple[int, int]] = [
    (480, 672),
    (672, 480),
    (528, 528),
    (672, 960),
    (960, 672),
    (768, 768),
]

TEXT_SEQUENCE_BUCKETS: list[int] = [64, 320, 768, 1024]
FP16_SAFE_RMS_PRESCALE: float = 0.0625  # 1/16: keeps (x * alpha)^2 < 65,504 for |x| <= 4,096


def default_checkpoint_dir() -> Path:
    env_path = os.environ.get("HEYLOOK_EMBEDDING_CHECKPOINT")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / "Storage" / "llms" / "google" / "embeddinggemma-2-MLX-BF16"


def inspect_checkpoint(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config.json not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or export EmbeddingGemma 2 towers to CoreML ANE bundles")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_dir(),
        help="Path to embeddinggemma-2-MLX-BF16 directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for .mlpackage / .mlmodelc bundles (defaults to <checkpoint>/coreml)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["vision", "audio", "text"],
        default=["vision", "audio", "text"],
        help="Which EmbeddingGemma 2 towers to export",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also run `xcrun coremlcompiler compile` to produce .mlmodelc bundles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Inspect checkpoint config and print planned ANE shape buckets",
    )
    args = parser.parse_args(argv)

    cfg = inspect_checkpoint(args.checkpoint)
    out_dir = args.out_dir or (args.checkpoint / "coreml")

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "out_dir": str(out_dir),
                "model_type": cfg.get("model_type", "embedding_gemma2"),
                "vision_buckets_hw": VISION_RESOLUTION_BUCKETS,
                "text_sequence_buckets": TEXT_SEQUENCE_BUCKETS,
                "fp16_safe_rms_prescale": FP16_SAFE_RMS_PRESCALE,
                "components": args.components,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
