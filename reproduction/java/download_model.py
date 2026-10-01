#!/usr/bin/env python3
from pathlib import Path

from huggingface_hub import snapshot_download

import json

from reproduce_java import MODEL_ID, MODEL_REVISION, sha256


destination = Path(__file__).resolve().parent / "models" / "codet5-base"
destination.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=MODEL_ID,
    revision=MODEL_REVISION,
    local_dir=destination,
)
(destination.parent / "manifest.json").write_text(
    json.dumps(
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_file": "codet5-base/pytorch_model.bin",
            "weight_sha256": sha256(destination / "pytorch_model.bin"),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(destination)
