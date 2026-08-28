"""Paths, credentials and endpoint settings shared by the auto-annotation pipeline.

Everything that names a location or a secret lives here so no other module has to guess.
Secrets are read from a plain ``KEY="value"`` env file at call time and never written to
disk, logged, or copied into the manifest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# ── Locations ─────────────────────────────────────────────────────────────────
# All generated data lands under one root so a run can be inspected or deleted wholesale.
DATA_ROOT = Path(os.environ.get("OOPSIE_ANNOT_ROOT", "/data/carlq/oopsie_annotation"))

INDEX_CSV = DATA_ROOT / "episodes.csv"      # release index, copied from the hub
MANIFEST = DATA_ROOT / "manifest.json"      # the sampled episodes for this run
RAW_DIR = DATA_ROOT / "raw"                 # .h5 + .mp4, mirroring release_path
FRAME_DIR = DATA_ROOT / "frames"            # jpeg frames handed to the model
CLIP_DIR = DATA_ROOT / "clips"              # normalised mp4 clips handed to the model
ANNOT_DIR = DATA_ROOT / "annotations"       # one json sidecar per episode
ANNOTATED_DIR = DATA_ROOT / "annotated"     # copies of the .h5 carrying model annotations
LOG_DIR = DATA_ROOT / "logs"                # raw request/response transcripts

# ── Source dataset ────────────────────────────────────────────────────────────
RELEASE_REPO = "OopsieData-Submissions/oopsiedata-v0.1"
RELEASE_REVISION = "main"

# ── Model endpoint ────────────────────────────────────────────────────────────
# The gateway is OpenAI-compatible and currently serves exactly one model. It is a
# reasoning model: it spends completion budget on `reasoning_content` before emitting
# `content`, so token budgets here are deliberately generous.
MODEL = "Qwen/Qwen3.8-27B"

# Stamped as the annotator subgroup name. Distinct from any human annotator in these files.
ANNOTATOR_NAME = "auto_cpiq_qwen3p8"
DEFAULT_BASE_URL = "https://api.costplusiq.com/v1"

SECRETS_ENV = Path(os.environ.get("OOPSIE_SECRETS", "/scratch/cluster/carlq/research/secrets.env"))


def load_secrets(path: Path = SECRETS_ENV) -> Dict[str, str]:
    """Parse a ``KEY="value"`` env file into a dict.

    Tolerates ``export`` prefixes and unquoted values, both of which have appeared in this
    file. Missing file is an error the caller should see immediately rather than a silent
    empty dict that fails later as a confusing 401.
    """
    if not path.exists():
        raise FileNotFoundError(f"secrets file not found: {path}")
    out: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def hf_token() -> str:
    """Read-only use only. The token in this file carries write scope on the org."""
    token = load_secrets().get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN missing from the secrets file")
    return token


def api_key() -> str:
    key = load_secrets().get("COSTPLUSIQ_API_KEY", "")
    if not key:
        raise RuntimeError("COSTPLUSIQ_API_KEY missing from the secrets file")
    return key


def base_url() -> str:
    return load_secrets().get("base_url", DEFAULT_BASE_URL).rstrip("/")
