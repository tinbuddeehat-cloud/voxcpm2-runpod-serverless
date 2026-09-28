# GitHub Actions Build Guide — v0.1.1

เวอร์ชันนี้ออกแบบให้ **ไม่ต้อง Build Docker บน PC** ของคุณ

## จุดที่เปลี่ยนจาก v0.1.0

- ใช้ `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` ซึ่งเบากว่า RunPod PyTorch dev image เดิมมาก
- Build บน GitHub-hosted runner
- Push image เข้า GitHub Container Registry (GHCR) อัตโนมัติ
- ล้าง SDK ที่ไม่ใช้บน runner ก่อน Build เพื่อเพิ่มพื้นที่ว่าง
- PC ของคุณไม่ต้องดาวน์โหลด CUDA/PyTorch image อีก

## 1) สร้าง GitHub repository

แนะนำชื่อ:

`voxcpm2-runpod-serverless`

Public หรือ Private ก็ได้

## 2) อัปโหลดไฟล์ทั้งหมดของโฟลเดอร์นี้ขึ้น root ของ repository

ต้องเห็นอย่างน้อย:

```text
.github/workflows/build-ghcr.yml
Dockerfile
handler.py
requirements.txt
README_TH.md
test_health.json
test_input.json
```

## 3) Commit เข้า branch main

Workflow จะเริ่มเอง หรือไปที่:

`Actions → Build VoxCPM2 RunPod Image → Run workflow`

## 4) เมื่อ Build สำเร็จ

จะได้ image:

```text
ghcr.io/<github-user>/<repo>:0.1.1
ghcr.io/<github-user>/<repo>:latest
```

ชื่อ image จะถูกแปลงเป็น lowercase อัตโนมัติ

## 5) ตั้ง GHCR package เป็น Public สำหรับรอบทดสอบ

`GitHub Profile → Packages → เลือก package → Package settings → Change visibility → Public`

การเปิด Docker image เป็น public **ไม่ได้เปิดเผย VoxCPM2 weights** ที่อยู่ใน RunPod Global Volume

## 6) RunPod Serverless

สร้าง custom Serverless endpoint แล้วใช้ image:

```text
ghcr.io/<github-user>/<repo>:0.1.1
```

ตั้งค่ารอบแรก:

- Worker min: `0`
- Worker max: `1`
- GPU: 16–24 GB
- Attach Global Volume เดิมที่มี VoxCPM2
- `VOXCPM_OPTIMIZE=false`
- `VOXCPM_DENOISER=false`
- `VOXCPM_MAX_TEXT_CHARS=2000`

Serverless Global Volume จะถูกอ่านจาก `/runpod-volume` โดย handler

## 7) ทดสอบตามลำดับ

1. `test_health.json`
2. `test_input.json`
3. Voice Clone ผ่าน `reference_audio_url` หรือ `reference_audio_b64`

หลัง integration ผ่านค่อย benchmark `VOXCPM_OPTIMIZE=true`
