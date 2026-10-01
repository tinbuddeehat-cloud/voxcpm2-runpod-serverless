import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from nano.runtime_utils import resolve_model, split_text
from nano.worker import NanoWorker, reference_bytes, https_url


class FakeServer:
    def __init__(self):
        self.encoded = 0
        self.calls = []

    def get_model_info(self):
        return {"output_sample_rate": 48000, "encoder_sample_rate": 24000, "patch_size": 2}

    def encode_latents(self, wav, wav_format):
        self.encoded += 1
        assert wav_format == "wav"
        return b"latents-test"

    def generate(self, **kw):
        self.calls.append(kw)
        yield np.zeros(24000, dtype=np.float32)
        yield np.zeros(24000, dtype=np.float32)

    def stop(self):
        pass


def reference():
    out = io.BytesIO()
    wave = np.sin(np.arange(48000) * .02).astype(np.float32) * .1
    sf.write(out, wave, 24000, format="WAV")
    return base64.b64encode(out.getvalue()).decode("ascii")


class ModelTests(unittest.TestCase):
    def model(self, path):
        path.mkdir(parents=True)
        for name in ["config.json", "audiovae.pth", "model.safetensors"]:
            (path / name).write_bytes(b"test")

    def test_runpod_hub_path_and_main_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "models--openbmb--VoxCPM2"
            model = repo / "snapshots" / "revision-a"
            self.model(model)
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("revision-a")
            found, source = resolve_model({"VOXCPM_CACHED_HUB": tmp})
            self.assertEqual(found, model)
            self.assertEqual(source, "runpod_cached_model")

    def test_missing_cache_fails_instead_of_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "no runtime download"):
                resolve_model({"VOXCPM_CACHED_HUB": tmp})

    def test_ambiguous_snapshot_and_pinned_revision_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models--openbmb--VoxCPM2" / "snapshots"
            self.model(root / "a")
            self.model(root / "b")
            with self.assertRaisesRegex(RuntimeError, "Multiple"):
                resolve_model({"VOXCPM_CACHED_HUB": tmp})
            with self.assertRaisesRegex(RuntimeError, "Pinned"):
                resolve_model({"VOXCPM_CACHED_HUB": tmp, "VOXCPM_MODEL_REVISION": "missing"})

    def test_explicit_model_must_be_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                resolve_model({"VOXCPM_MODEL_DIR": tmp})
            self.model(Path(tmp) / "model")
            path, source = resolve_model({"VOXCPM_MODEL_DIR": str(Path(tmp) / "model")})
            self.assertEqual(source, "explicit_local")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.worker = NanoWorker(self.server, {"VOXCPM_INFERENCE_TIMESTEPS": "10", "VOXCPM_SEGMENT_CHARS": "250"})

    def tearDown(self):
        self.worker.close()

    def test_ultimate_clone_contract_encode_once_reuse_for_segments(self):
        inp = {"text": "สวัสดีครับ วันนี้เราจะทดสอบเสียงภาษาไทย. " * 30,
               "reference_text": "สวัสดีครับ นี่เป็นบทพูดต้นฉบับ", "reference_audio_b64": reference(),
               "inference_timesteps": 10, "cfg_value": 2, "seed": 42, "dialect": "isan"}
        out = self.worker.run(inp)
        self.assertEqual(out["engine"], "voxcpm2")
        self.assertEqual(out["backend"], "nano-vllm")
        self.assertEqual(out["mode"], "ultimate_clone")
        self.assertGreater(out["segments"], 1)
        self.assertEqual(self.server.encoded, 1)
        for call in self.server.calls:
            self.assertEqual(call["prompt_latents"], call["ref_audio_latents"])
            self.assertEqual(call["prompt_text"], inp["reference_text"])
        wav = sf.info(io.BytesIO(base64.b64decode(out["audio_b64"])))
        self.assertEqual(wav.samplerate, 48000)
        self.assertEqual(wav.duration, out["segments"])
        again = self.worker.run(inp)
        self.assertTrue(again["reference_cache_hit"])
        self.assertEqual(self.server.encoded, 1)
        self.assertEqual(again["request_index"], 2)
        json.dumps(again)

    def test_reference_mode_without_transcript(self):
        self.worker.run({"text": "ทดสอบ", "reference_audio_b64": reference()})
        self.assertIn("ref_audio_latents", self.server.calls[0])
        self.assertNotIn("prompt_latents", self.server.calls[0])

    def test_health_does_not_generate_or_increment(self):
        out = self.worker.run({"action": "health"})
        self.assertFalse(out["extra_warmup"])
        self.assertEqual(out["generate_requests"], 0)
        self.assertEqual(self.server.calls, [])

    def test_invalid_steps_and_missing_reference_fail_before_generation(self):
        for inp in [{"text": "ทดสอบ", "inference_timesteps": 8},
                    {"text": "ทดสอบ", "reference_text": "บทพูด"},
                    {"text": "x" * 2001}]:
            with self.assertRaises(ValueError):
                self.worker.run(inp)
        self.assertEqual(self.server.calls, [])

    def test_reference_cache_disabled_is_bounded(self):
        self.worker.cache_limit = 0
        raw = base64.b64decode(reference())
        for _ in range(2):
            self.worker.encode_reference(raw)
        self.assertEqual(self.server.encoded, 2)
        self.assertEqual(self.worker.cached_bytes, 0)

    def test_reference_bytes_are_bounded_and_sources_exclusive(self):
        with self.assertRaises(ValueError):
            reference_bytes({"reference_audio_b64": "YQ==", "reference_audio_url": "https://example.com"}, Mock())
        with self.assertRaises(ValueError):
            reference_bytes({"reference_audio_b64": "!invalid"}, Mock())

    def test_upload_preserves_bridge_contract_and_does_not_return_audio(self):
        self.worker.session.put = Mock(return_value=Mock(status_code=200))
        out = self.worker.run({"text": "ทดสอบ", "output_upload_url": "https://example.com/api/runpod/output?sig=secret"})
        self.assertEqual(out["output"], "uploaded")
        self.assertNotIn("audio_b64", out)
        self.assertNotIn("secret", json.dumps(out))
        self.assertEqual(self.worker.session.put.call_args.kwargs["headers"]["Content-Type"], "audio/wav")

    def test_upload_failure_is_not_reported_as_success(self):
        self.worker.session.put = Mock(return_value=Mock(status_code=500))
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            self.worker.run({"text": "ทดสอบ", "output_upload_url": "https://example.com/upload"})
        self.assertEqual(self.worker.request_index, 0)

    def test_large_wav_requires_bridge_instead_of_oversized_json(self):
        with patch("nano.worker.MAX_INLINE_WAV_BYTES", 100):
            with self.assertRaisesRegex(RuntimeError, "output_upload_url"):
                self.worker.run({"text": "ทดสอบ"})
            self.worker.session.put = Mock(return_value=Mock(status_code=200))
            out = self.worker.run({"text": "ทดสอบ", "output_upload_url": "https://example.com/upload"})
            self.assertEqual(out["output"], "uploaded")

    def test_audio_urls_require_https(self):
        for url in ["file:///etc/passwd", "http://example.com", "https://user:pass@example.com"]:
            with self.assertRaises(ValueError):
                https_url(url)


class TextTests(unittest.TestCase):
    def test_thai_700_and_1000_characters_retained_exactly(self):
        source = "วันนี้เราทดสอบเสียงภาษาไทยให้ครบทุกประโยค และรักษาสำเนียงเดิมครับ. " * 30
        for n in [700, 1000]:
            text = source[:n]
            parts = split_text(text)
            self.assertEqual("".join(parts), text)
            self.assertTrue(all(len(p) <= 350 for p in parts))

    def test_no_split_before_thai_combining_mark(self):
        parts = split_text("ก้" * 600)
        self.assertTrue(all(p[0] != "้" for p in parts))
        self.assertEqual("".join(parts), "ก้" * 600)


if __name__ == "__main__":
    unittest.main()
