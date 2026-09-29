import base64, glob, io, math, os, shutil, subprocess, tempfile, time
from pathlib import Path

import requests
import runpod
import soundfile as sf
import torch
from voxcpm import VoxCPM

DEVICE = os.getenv("VOXCPM_DEVICE", "cuda")
VOL = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
CLEAN = Path(os.getenv("VOXCPM_VOLUME_MODEL_DIR", str(VOL / "voxcpm2-local")))
HFVOL = Path(os.getenv("VOXCPM_VOLUME_HF_CACHE", str(VOL / "huggingface-cache")))
RUNTIME = Path(os.getenv("VOXCPM_RUNTIME_DIR", "/tmp/voxcpm2-runtime"))
OPT = os.getenv("VOXCPM_OPTIMIZE", "false").lower() in ("1", "true", "yes", "on")
MAX_CHARS = int(os.getenv("VOXCPM_MAX_TEXT_CHARS", "2000"))


def clamp(value, low, high, default):
    try:
        x = float(value)
        if not math.isfinite(x):
            return default
        return max(low, min(high, x))
    except (TypeError, ValueError):
        return default


def find_weight(name):
    direct = CLEAN / name
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    pats = [
        str(HFVOL / "models--openbmb--VoxCPM2" / "snapshots" / "*" / name),
        str(HFVOL / "**" / name),
    ]
    for pat in pats:
        for m in glob.glob(pat, recursive=True):
            p = Path(m)
            if p.is_file() and p.stat().st_size > 0:
                return p
    return None


def prepare_model():
    if not CLEAN.is_dir():
        return "openbmb/VoxCPM2"
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for n in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_voxcpm2.py",
    ]:
        s = CLEAN / n
        if not s.is_file() or s.stat().st_size == 0:
            raise RuntimeError(f"Missing model file: {s}")
        shutil.copy2(s, RUNTIME / n)
    for n in ["model.safetensors", "audiovae.pth"]:
        s = find_weight(n)
        if s is None:
            raise RuntimeError(f"Missing weight: {n}. Checked {CLEAN} and {HFVOL}.")
        d = RUNTIME / n
        if d.exists() or d.is_symlink():
            d.unlink()
        os.symlink(s, d)
        print(f"[startup] {n} -> {s}")
    return str(RUNTIME)


def load_model():
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    src = prepare_model()
    t = time.time()
    model = VoxCPM.from_pretrained(
        src,
        cache_dir="/tmp/hf-cache",
        load_denoiser=False,
        optimize=OPT,
        device=DEVICE,
    )
    elapsed = time.time() - t
    print(f"[startup] VoxCPM2 ready in {elapsed:.2f}s optimize={OPT}")
    return model, elapsed


MODEL, LOAD_S = load_model()
SR = MODEL.tts_model.sample_rate


def ref_file(inp, tmp):
    b64 = inp.get("reference_audio_b64")
    url = inp.get("reference_audio_url")
    if b64 and url:
        raise ValueError("Use only one reference audio source")
    if not b64 and not url:
        return None
    p = Path(tmp) / "reference.wav"
    if b64:
        if b64.strip().lower().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        p.write_bytes(base64.b64decode(b64, validate=True))
    else:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
    return str(p)


def wav_bytes(wav):
    b = io.BytesIO()
    sf.write(b, wav, SR, format="WAV", subtype="PCM_16")
    return b.getvalue()


def normalized_modifier(inp):
    raw = inp.get("modifier") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "speed": clamp(raw.get("speed", 1.0), 0.75, 1.25, 1.0),
        "pitch_semitones": clamp(raw.get("pitch_semitones", 0.0), -4.0, 4.0, 0.0),
        "volume": clamp(raw.get("volume", 1.0), 0.5, 1.5, 1.0),
    }


def post_process_wav(data, tmp, modifier):
    speed = modifier["speed"]
    pitch = modifier["pitch_semitones"]
    volume = modifier["volume"]
    if abs(speed - 1.0) < 1e-6 and abs(pitch) < 1e-6 and abs(volume - 1.0) < 1e-6:
        return data, False

    src = Path(tmp) / "generated_raw.wav"
    dst = Path(tmp) / "generated_modified.wav"
    src.write_bytes(data)

    filters = []
    if abs(pitch) >= 1e-6:
        factor = 2.0 ** (pitch / 12.0)
        # asetrate changes pitch + tempo; atempo compensates duration while
        # also applying the requested speed in one stable pass.
        tempo = speed / factor
        filters.extend([
            f"asetrate={SR}*{factor:.10f}",
            f"aresample={SR}",
            f"atempo={tempo:.10f}",
        ])
    elif abs(speed - 1.0) >= 1e-6:
        filters.append(f"atempo={speed:.10f}")

    if abs(volume - 1.0) >= 1e-6:
        filters.append(f"volume={volume:.6f}")

    # Keep headroom when users boost volume or pitch processing creates peaks.
    filters.append("alimiter=limit=0.98")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", ",".join(filters),
        "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 44:
        err = proc.stderr.decode("utf-8", "replace")[-1200:]
        raise RuntimeError(f"Voice modifier failed: {err or 'ffmpeg output missing'}")
    return dst.read_bytes(), True


def handler(job):
    inp = job.get("input") or {}
    action = str(inp.get("action", "generate")).lower()

    if action == "health":
        return {
            "ok": True,
            "engine": "voxcpm2",
            "version": "0.1.4",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "sample_rate": SR,
            "optimize": OPT,
            "model_load_seconds": round(LOAD_S, 3),
            "voice_modifier": True,
        }

    if action != "generate":
        raise ValueError("action must be generate or health")

    text = str(inp.get("text", "")).strip()
    if not text:
        raise ValueError("input.text is required")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text exceeds {MAX_CHARS} chars")

    reference_text = str(inp.get("reference_text", "")).strip()
    if len(reference_text) > MAX_CHARS:
        raise ValueError(f"reference_text exceeds {MAX_CHARS} chars")

    cfg = clamp(inp.get("cfg_value", 2.0), 1.0, 3.0, 2.0)
    steps = int(clamp(inp.get("inference_timesteps", 10), 4, 30, 10))
    seed = inp.get("seed")
    if seed is not None:
        seed = int(seed)
    modifier = normalized_modifier(inp)

    t = time.time()
    with tempfile.TemporaryDirectory(prefix="voxcpm2-") as tmp:
        ref = ref_file(inp, tmp)
        if reference_text and not ref:
            raise ValueError("reference_text requires reference audio")

        kw = {"text": text, "cfg_value": cfg, "inference_timesteps": steps}
        if seed is not None:
            kw["seed"] = seed
        if ref:
            kw["reference_wav_path"] = ref
            if reference_text:
                kw["prompt_wav_path"] = ref
                kw["prompt_text"] = reference_text

        wav = MODEL.generate(**kw)
        generation_elapsed = time.time() - t
        raw_duration = len(wav) / SR
        data = wav_bytes(wav)

        post_t = time.time()
        data, modified = post_process_wav(data, tmp, modifier)
        post_elapsed = time.time() - post_t

        # Re-read final WAV duration after speed modification.
        try:
            final_info = sf.info(io.BytesIO(data))
            final_duration = float(final_info.frames) / float(final_info.samplerate)
        except Exception:
            final_duration = raw_duration / modifier["speed"]

        if ref and reference_text:
            mode = "ultimate_clone"
        elif ref:
            mode = "voice_clone"
        else:
            mode = "tts"

        out = {
            "ok": True,
            "engine": "voxcpm2",
            "version": "0.1.4",
            "mode": mode,
            "sample_rate": SR,
            "duration_seconds": round(final_duration, 3),
            "raw_duration_seconds": round(raw_duration, 3),
            "generation_seconds": round(generation_elapsed, 3),
            "postprocess_seconds": round(post_elapsed, 3),
            "rtf": round(generation_elapsed / raw_duration, 3) if raw_duration else None,
            "cfg_value": cfg,
            "inference_timesteps": steps,
            "modifier": modifier,
            "modified": modified,
        }

        up = inp.get("output_upload_url")
        if up:
            r = requests.put(up, data=data, headers={"Content-Type": "audio/wav"}, timeout=60)
            r.raise_for_status()
            out["output"] = "uploaded"
        else:
            out["output"] = "base64"
            out["audio_b64"] = base64.b64encode(data).decode("ascii")
        return out


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
