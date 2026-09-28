# VoxCPM2 Runpod Serverless v0.1.0

## เป้าหมาย
ย้ายจาก Pod ทดลองไปเป็น Queue Serverless worker โดยไม่เปิด GPU ค้าง

## ก่อนเริ่ม
กด Stop Pod ได้เลย โมเดลและไฟล์ที่อยู่ใน Global Volume จะยังอยู่
ใน Serverless Global Volume จะ mount ที่ `/runpod-volume` ไม่ใช่ `/workspace`.

## Build image
```powershell
docker build --platform linux/amd64 -t YOUR_USER/voxcpm2-runpod:0.1.0 .
docker login
docker push YOUR_USER/voxcpm2-runpod:0.1.0
```

## สร้าง Endpoint
Runpod > Serverless > New Endpoint > Import from Docker Registry
- Endpoint type: Queue
- Workers min: 0
- Workers max: 1
- Idle timeout: 300-600 sec
- Flash Boot: On
- GPU: เริ่มจาก 16GB/24GB
- Advanced > Global volume: เลือก volume เดิมที่มี VoxCPM2

## Environment
`VOXCPM_OPTIMIZE=false` ในรอบ integration แรก เพื่อลด cold-start compile
หลังระบบผ่านค่อย benchmark `true`

## Health
```json
{"input":{"action":"health"}}
```

## TTS
```json
{"input":{"action":"generate","text":"สวัสดีครับ นี่คือเสียงจาก VoxCPM2","cfg_value":2.0,"inference_timesteps":10}}
```

## Voice Clone
ส่งอย่างใดอย่างหนึ่ง:
- `reference_audio_url` = signed R2 GET URL
- `reference_audio_b64` = WAV base64

ถ้าไม่ส่ง `output_upload_url` worker จะตอบ WAV เป็น base64.
Production ควรส่ง signed R2 PUT URL ใน `output_upload_url` เพื่อให้ worker PUT ไฟล์เข้า R2 โดยตรง.

## Runpod API
Sync: `POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`
Async: `POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run`

AI Voice Studio production ควรใช้ async `/run` + เช็ก job status.
