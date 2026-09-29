# VoxCPM2 RunPod Serverless v0.1.4 — Voice Modifier

เพิ่ม `input.modifier`:

```json
{
  "speed": 1.0,
  "pitch_semitones": 0,
  "volume": 1.0
}
```

ช่วงที่อนุญาต: speed 0.75–1.25, pitch -4..+4 semitones, volume 0.5–1.5.
FFmpeg post-process ทำหลัง VoxCPM2 generate และมี limiter ป้องกัน clipping.
