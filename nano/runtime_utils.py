"""Model discovery and text segmentation; no GPU imports or downloads."""
import os
import unicodedata
from pathlib import Path


def valid_model(path):
    path = Path(path)
    return (path / "config.json").is_file() and (path / "audiovae.pth").is_file() and any(
        p.is_file() and p.stat().st_size > 0 for p in path.glob("*.safetensors")
    )


def resolve_model(env=None):
    env = os.environ if env is None else env
    explicit = env.get("VOXCPM_MODEL_DIR")
    if explicit:
        path = Path(explicit)
        if not valid_model(path):
            raise RuntimeError("VOXCPM_MODEL_DIR is missing config, VAE or safetensors weights")
        return path, "explicit_local"
    hub = Path(env.get("VOXCPM_CACHED_HUB", "/runpod-volume/huggingface-cache/hub"))
    repo = hub / "models--openbmb--VoxCPM2"
    revision = env.get("VOXCPM_MODEL_REVISION", "main")
    ref = repo / "refs" / revision
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
    candidate = repo / "snapshots" / revision
    if valid_model(candidate):
        return candidate, "runpod_cached_model"
    # A pinned revision must never silently select different weights.
    if env.get("VOXCPM_MODEL_REVISION") not in (None, "", "main"):
        raise RuntimeError("Pinned VoxCPM2 revision not found in RunPod cached model")
    snapshots = sorted((repo / "snapshots").glob("*"))
    valid = [p for p in snapshots if valid_model(p)]
    if len(valid) == 1:
        return valid[0], "runpod_cached_model"
    if len(valid) > 1:
        raise RuntimeError("Multiple model snapshots: set VOXCPM_MODEL_REVISION explicitly")
    raise RuntimeError("Cached VoxCPM2 not found. Set RunPod Model to openbmb/VoxCPM2; no runtime download is attempted")


def split_text(text, target=250, hard_limit=350):
    """Prefer punctuation/phrase boundaries; retain every source character."""
    target = max(50, int(target))
    hard_limit = max(target, int(hard_limit))
    parts = []
    while len(text) > hard_limit:
        before = text[:hard_limit]
        sentence = [i + 1 for i, ch in enumerate(before) if ch in ".!?。！？\n" and i >= target // 2]
        spaces = [i + 1 for i, ch in enumerate(before) if ch.isspace() and i >= target // 2]
        options = sentence or spaces
        cut = min(options, key=lambda n: abs(n - target)) if options else hard_limit
        # Do not start the next segment with a Thai vowel/tone combining mark.
        while cut < len(text) and unicodedata.category(text[cut]).startswith("M"):
            cut += 1
        parts.append(text[:cut])
        text = text[cut:]
    if text:
        parts.append(text)
    return parts
