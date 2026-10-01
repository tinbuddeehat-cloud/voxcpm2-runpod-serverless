"""Compatibility adapter: existing RunPod input -> Nano-vLLM -> WAV/R2."""
import base64
import hashlib
import io
import math
import os
import subprocess
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

from nano.runtime_utils import resolve_model, split_text

VERSION = "0.2.0-nano-experimental"
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
MAX_INLINE_WAV_BYTES = 14_000_000  # Base64 + metadata stays below /runsync's 20 MB cap.


def clamp(value, lo, hi, default):
    try:
        value = float(value)
        return max(lo, min(hi, value)) if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def https_url(value):
    url = str(value or "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Audio URLs must use HTTPS")
    return url


def reference_bytes(inp, session):
    encoded, url = inp.get("reference_audio_b64"), inp.get("reference_audio_url")
    if encoded and url:
        raise ValueError("Use only one reference audio source")
    if encoded:
        value = str(encoded)
        if value.lower().startswith("data:"):
            value = value.split(",", 1)[1]
        if len(value) > (MAX_REFERENCE_BYTES * 4 // 3 + 8):
            raise ValueError("Reference audio is too large")
        data = base64.b64decode(value, validate=True)
    elif url:
        data = bytearray()
        try:
            with session.get(https_url(url), stream=True, timeout=(5, 30), allow_redirects=False) as response:
                if response.status_code != 200:
                    raise RuntimeError("Reference download failed")
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > MAX_REFERENCE_BYTES:
                        raise ValueError("Reference audio is too large")
        except requests.RequestException:
            raise RuntimeError("Reference download failed") from None
        data = bytes(data)
    else:
        return None
    if not data or len(data) > MAX_REFERENCE_BYTES:
        raise ValueError("Invalid reference audio size")
    return data


def prepare_reference(data):
    """Validate, downmix, and normalize once; preserve the original voice."""
    try:
        info = sf.info(io.BytesIO(data))
        duration = info.frames / info.samplerate
        if not 1 <= duration <= 60 or info.channels > 8:
            raise ValueError("Reference must be 1-60 seconds with at most 8 channels")
        wav, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        raise ValueError("Reference audio cannot be decoded; use WAV, MP3 or FLAC") from None
    mono = wav.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError("Invalid samples in reference audio")
    out = io.BytesIO()
    sf.write(out, mono, rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def modify_wav(data, sample_rate, modifier):
    modifier = modifier if isinstance(modifier, dict) else {}
    speed = clamp(modifier.get("speed", 1), .75, 1.25, 1)
    pitch = clamp(modifier.get("pitch_semitones", 0), -4, 4, 0)
    volume = clamp(modifier.get("volume", 1), .5, 1.5, 1)
    values = {"speed": speed, "pitch_semitones": pitch, "volume": volume}
    if (speed, pitch, volume) == (1, 0, 1):
        return data, False, values
    filters = []
    if pitch:
        factor = 2 ** (pitch / 12)
        filters = [f"asetrate={sample_rate}*{factor:.10f}", f"aresample={sample_rate}", f"atempo={speed / factor:.10f}"]
    elif speed != 1:
        filters.append(f"atempo={speed:.10f}")
    if volume != 1:
        filters.append(f"volume={volume:.6f}")
    filters.append("alimiter=limit=0.98")
    with tempfile.TemporaryDirectory(prefix="voxcpm-nano-") as tmp:
        source, dest = Path(tmp) / "raw.wav", Path(tmp) / "final.wav"
        source.write_bytes(data)
        result = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                                 "-af", ",".join(filters), "-ar", str(sample_rate), "-ac", "1", str(dest)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if result.returncode or not dest.is_file():
            raise RuntimeError("Voice modifier failed")
        return dest.read_bytes(), True, values


class NanoWorker:
    def __init__(self, server=None, settings=None):
        self.settings = os.environ.copy() if settings is None else settings
        self.steps = int(self.settings.get("VOXCPM_INFERENCE_TIMESTEPS", "10"))
        if not 4 <= self.steps <= 30:
            raise ValueError("VOXCPM_INFERENCE_TIMESTEPS must be 4-30")
        self.request_index = 0
        self.reference_cache = OrderedDict()
        self.cached_bytes = 0
        self.cache_limit = int(self.settings.get("VOXCPM_REFERENCE_CACHE_MB", "16")) * 1024 * 1024
        self.max_chars = int(self.settings.get("VOXCPM_MAX_TEXT_CHARS", "2000"))
        self.session = requests.Session()
        started = time.perf_counter()
        if server is None:
            import torch
            from nanovllm_voxcpm import VoxCPM
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required")
            model, self.model_source = resolve_model(self.settings)
            self.model_revision = model.name
            # One GPU, one job at a time. Capture only batch-size 1 graphs.
            server = VoxCPM.from_pretrained(
                model=str(model), devices=[0], inference_timesteps=self.steps,
                max_num_batched_tokens=4096, max_model_len=4096, max_num_seqs=1,
                gpu_memory_utilization=clamp(self.settings.get("NANO_GPU_MEMORY_UTILIZATION", ".65"), .3, .9, .65),
                enforce_eager=self.settings.get("NANO_ENFORCE_EAGER", "false").lower() == "true",
            )
            self.gpu = torch.cuda.get_device_name(0)
        else:
            self.model_source, self.model_revision, self.gpu = "test_double", "test", "mock"
        self.server = server
        info = self.server.get_model_info()
        self.sample_rate = int(info["output_sample_rate"])
        self.load_seconds = time.perf_counter() - started
        self.worker_id = self.settings.get("RUNPOD_POD_ID") or self.settings.get("HOSTNAME", "unknown")
        print(f"[startup] nano ready: source={self.model_source} load_seconds={self.load_seconds:.3f}", flush=True)

    def close(self):
        self.session.close()
        self.server.stop()

    def encode_reference(self, raw):
        key = hashlib.sha256(raw).digest()
        if key in self.reference_cache:
            self.reference_cache.move_to_end(key)
            return self.reference_cache[key], True
        latents = self.server.encode_latents(prepare_reference(raw), "wav")
        # Cache CPU bytes, not a growing set of GPU tensors or signed URLs.
        if len(latents) <= self.cache_limit:
            while self.reference_cache and self.cached_bytes + len(latents) > self.cache_limit:
                _, evicted = self.reference_cache.popitem(last=False)
                self.cached_bytes -= len(evicted)
            self.reference_cache[key] = latents
            self.cached_bytes += len(latents)
        return latents, False

    def run(self, inp):
        if not isinstance(inp, dict):
            raise ValueError("input must be an object")
        started = time.perf_counter()
        action = str(inp.get("action", "generate")).lower()
        common = {"ok": True, "engine": "voxcpm2", "backend": "nano-vllm", "version": VERSION,
                  "worker_id": self.worker_id, "sample_rate": self.sample_rate,
                  "model_source": self.model_source, "model_revision": self.model_revision,
                  "model_load_seconds": round(self.load_seconds, 3), "gpu": self.gpu,
                  "inference_timesteps": self.steps, "voice_modifier": True}
        if action == "health":
            return {**common, "generate_requests": self.request_index, "extra_warmup": False}
        if action != "generate":
            raise ValueError("action must be generate or health")
        text = str(inp.get("text", "")).strip()
        if not text or len(text) > self.max_chars:
            raise ValueError(f"text must contain 1-{self.max_chars} characters")
        transcript = str(inp.get("reference_text", "")).strip()
        if len(transcript) > self.max_chars:
            raise ValueError("reference_text exceeds character limit")
        requested_steps = int(clamp(inp.get("inference_timesteps", self.steps), 4, 30, self.steps))
        if requested_steps != self.steps:
            raise ValueError("inference_timesteps must match VOXCPM_INFERENCE_TIMESTEPS; worker is not reloaded per request")
        upload = https_url(inp["output_upload_url"]) if inp.get("output_upload_url") else None
        download_start = time.perf_counter()
        raw = reference_bytes(inp, self.session)
        download_s = time.perf_counter() - download_start
        if transcript and raw is None:
            raise ValueError("reference_text requires reference audio")
        encode_start = time.perf_counter()
        latents, cache_hit = self.encode_reference(raw) if raw is not None else (None, False)
        encode_s = time.perf_counter() - encode_start
        target = int(self.settings.get("VOXCPM_SEGMENT_CHARS", "250"))
        segments = split_text(text, target, max(target, 350)) if target else [text]
        cfg = clamp(inp.get("cfg_value", 2), 1, 3, 2)
        seed = int(inp["seed"]) if inp.get("seed") is not None else None
        wavs, segment_seconds = [], []
        generation_start = time.perf_counter()
        for index, segment in enumerate(segments):
            seg_start = time.perf_counter()
            kw = {"target_text": segment.strip(), "cfg_value": cfg, "temperature": 1.0,
                  "max_generate_length": 2000}
            if latents is not None:
                kw["ref_audio_latents"] = latents
                if transcript:
                    kw.update(prompt_latents=latents, prompt_text=transcript)
            if seed is not None:
                kw["seed"] = seed + index
            chunks = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in self.server.generate(**kw)]
            if not chunks:
                raise RuntimeError("Engine returned empty audio")
            wav = np.concatenate(chunks)
            if not wav.size or not np.isfinite(wav).all():
                raise RuntimeError("Engine returned invalid audio")
            # Bound runaway output; verify segment endings in GPU validation.
            if len(wav) / self.sample_rate > 180:
                raise RuntimeError("Segment exceeded duration limit; use shorter segments")
            wavs.append(wav)
            segment_seconds.append(round(time.perf_counter() - seg_start, 3))
        generation_s = time.perf_counter() - generation_start
        combined = np.concatenate(wavs)
        raw_duration = len(combined) / self.sample_rate
        encode_audio_start = time.perf_counter()
        output = io.BytesIO()
        sf.write(output, combined, self.sample_rate, format="WAV", subtype="PCM_16")
        data, modified, modifier = modify_wav(output.getvalue(), self.sample_rate, inp.get("modifier"))
        post_s = time.perf_counter() - encode_audio_start
        duration = sf.info(io.BytesIO(data)).duration
        upload_start = time.perf_counter()
        if upload:
            try:
                response = self.session.put(upload, data=data, headers={"Content-Type": "audio/wav"},
                                            timeout=(5, 30), allow_redirects=False)
                if response.status_code not in (200, 201, 204):
                    raise RuntimeError("Output upload failed")
            except requests.RequestException:
                raise RuntimeError("Output upload failed") from None
            result_audio = {"output": "uploaded"}
        else:
            if len(data) > MAX_INLINE_WAV_BYTES:
                raise RuntimeError("WAV is too large for inline output; supply output_upload_url")
            result_audio = {"output": "base64", "audio_b64": base64.b64encode(data).decode("ascii")}
        upload_s = time.perf_counter() - upload_start
        self.request_index += 1
        return {**common, **result_audio, "mode": "ultimate_clone" if latents is not None and transcript else "voice_clone" if latents is not None else "tts",
                "duration_seconds": round(duration, 3), "raw_duration_seconds": round(raw_duration, 3),
                "generation_seconds": round(generation_s, 3), "rtf": round(generation_s / raw_duration, 3),
                "postprocess_seconds": round(post_s, 3), "reference_download_seconds": round(download_s, 3),
                "reference_encode_seconds": round(encode_s, 3), "upload_seconds": round(upload_s, 3),
                "handler_seconds": round(time.perf_counter() - started, 3), "reference_cache_hit": cache_hit,
                "request_index": self.request_index, "segments": len(segments), "segment_seconds": segment_seconds,
                "characters": len(text), "cfg_value": cfg, "modifier": modifier, "modified": modified}
