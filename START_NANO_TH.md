# เริ่มทดสอบ VoxCPM2 + Nano-vLLM บน RunPod

รุ่นทดสอบ: `0.2.0-nano` — เตรียมให้รับ input/output เดิมของ AI Voice Studio v5.2.4

เป้าหมาย: ข้อความภาษาไทย 700 และ 1,000 ตัวอักษร สร้างจนได้ WAV ภายใน 60 วินาที โดยไม่เปิด GPU รอ 300 วินาที

Adapter ผ่าน unit tests 16 รายการด้วย engine จำลอง และทดสอบ FFmpeg modifier กับ WAV จริงแล้ว ยังไม่ได้ build container หรือทดสอบ GPU/RunPod จริง จึงยังยืนยันคุณภาพเสียงและเวลาไม่ถึง 60 วินาทีไม่ได้

ชุดนี้เพิ่มไฟล์ทดลองเท่านั้น ไม่ต้องรวม Video Generate หรือแก้ payment/credit

## 1. Build บน GitHub

สิทธิ์ GitHub connector ปัจจุบันอ่าน repository ได้ แต่สร้าง branch ไม่ได้ (HTTP 403: Resource not accessible by integration) จึงยังไม่มี branch, PR หรือ image รุ่นนี้บน GitHub

ทำขั้นตอนนี้ผ่านหน้าเว็บ GitHub:

1. แตก ZIP `VOXCPM2-NANO-v0.2.0-PATCH.zip`
2. เปิด https://github.com/tinbuddeehat-cloud/voxcpm2-runpod-serverless
3. กดตัวเลือก branch `main` แล้วสร้าง branch ชื่อ `codex/voxcpm2-nano` จาก main
4. อยู่บน branch ใหม่ → Add file → Upload files
5. อัปโหลด **ไฟล์และโฟลเดอร์ภายใน ZIP** ไปที่ root ของ repository รวม `.github`, `nano`, `scripts`, `tests`, `samples` อย่าสร้างโฟลเดอร์ครอบเพิ่ม และไม่ต้องลบไฟล์เดิม
6. Commit เข้า branch ทดสอบนี้

ตรวจว่า path เป็น `.github/workflows/build-nano.yml` และ `nano/handler.py` ไม่ใช่ `VOXCPM2-NANO-v0.2.0-PATCH/nano/handler.py`

เปิด Actions แล้วเลือก **Build Nano-vLLM experimental image**

Workflow ตรวจ adapter ก่อน build Dockerfile.nano และ push เฉพาะ tag:

```
ghcr.io/tinbuddeehat-cloud/voxcpm2-runpod-serverless:0.2.0-nano
```

ไฟล์ `.github/workflows/build-nano.yml` ทำงานอัตโนมัติเมื่อ push branch นี้ ถ้า workflow ยังไม่ขึ้น ให้ตรวจว่าไฟล์ถูก push ครบและ Actions เปิดอยู่

ใช้ tag `nano-<commit SHA>` หรือ image digest ที่ Actions แสดงเพื่อทดสอบซ้ำจาก image เดิม อย่าใช้ `latest`

ถ้าติดข้อจำกัด package visibility ของ GHCR ให้ตั้ง package เป็น Public หรือใส่ registry credentials ใน RunPod ตามการตั้งค่าเดิม

ไม่ต้อง build Docker, ติดตั้ง Nano-vLLM หรือดาวน์โหลดโมเดลลง Windows

## 2. สร้าง Endpoint ทดสอบ

RunPod → Serverless → New Endpoint → Docker image

| ค่า | ตั้งเป็น |
|---|---|
| Name | voxcpm2-nano-test |
| Image | tag 0.2.0-nano ด้านบน หลัง build สำเร็จ |
| GPU | RTX 4090 / 4090 PRO |
| GPUs per worker | 1 |
| Active workers | 0 |
| Max workers | 1 |
| Idle timeout | 5 วินาที |
| FlashBoot | เปิด |
| Model / Cached model | openbmb/VoxCPM2 |
| Execution timeout | 120 วินาทีสำหรับการทดลองแรก |

Worker ใช้ `/runpod-volume/huggingface-cache/hub/` ที่ RunPod เตรียมให้ เลือก Cached Model จริง ไม่ใช่เพียง mount Network Volume ที่ใช้ path ชื่อคล้ายกัน

Worker ไม่ดาวน์โหลดโมเดลเอง หาก cache ไม่ครบจะหยุดพร้อมแจ้งเหตุผล เราจึงไม่เผลอจ่ายค่า GPU รอดาวน์โหลดใหม่

Environment defaults อยู่ใน image แล้ว:

```
VOXCPM_INFERENCE_TIMESTEPS=10
VOXCPM_SEGMENT_CHARS=250
NANO_GPU_MEMORY_UTILIZATION=0.65
NANO_ENFORCE_EAGER=false
```

ไม่ต้องตั้ง VOXCPM_MODEL_DIR เว้นแต่ต้องการทดสอบ local model ที่เตรียมเอง ถ้าใช้หลาย model snapshots ให้กำหนด VOXCPM_MODEL_REVISION เป็น hash ให้แน่นอน

อย่าย้ายเว็บจริงมาที่ Endpoint นี้จนกว่าจะผ่านการฟังและทดสอบครบ

## 3. ทดสอบครั้งเดียวจาก Windows PowerShell

ใช้ไฟล์เสียงต้นฉบับเดียวกับที่โคลนแล้วเสียงเหมือนที่สุด เช่น `myvoice.wav` 10–30 วินาที และสร้าง `reference_text.txt` แบบ UTF-8 ให้เป็นคำพูดที่ตรงกับไฟล์เสียงทุกคำ

เปิด PowerShell ในโฟลเดอร์โปรเจกต์ แล้วรัน (แทน ENDPOINT_ID ด้วย ID ของ Endpoint ใหม่):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\benchmark-nano.ps1 -EndpointId "ENDPOINT_ID" -ReferenceAudio ".\myvoice.wav" -ReferenceTextFile ".\reference_text.txt"
```

สคริปต์ถาม RunPod API key แบบซ่อน ไม่ต้องใส่ key ในโค้ดหรือส่งมาทางแชต จะสร้างเพียง **1 งาน** โดยไม่ส่ง health หรือ warm-up ทิ้งก่อน

สคริปต์ใช้ `/runsync?wait=1000` แล้ว poll ผล เพื่อใช้เพดาน payload 20 MB สำหรับ WAV แทน 10 MB ของ `/run` พร้อมกำหนด execution timeout และ TTL ต่อคำขอ ไฟล์ WAV แบบ inline จำกัด 14 MB (ประมาณ 146 วินาทีที่ 48 kHz mono PCM16) ถ้าเสียงยาวกว่านี้ให้ทดสอบผ่าน `output_upload_url`/R2 ของเว็บจริง; ห้ามลดคุณภาพหรือย่อข้อความเพื่อรายงานว่าผ่าน 1,000 ตัวอักษร

ได้ WAV และ `benchmark-results/results.csv` เปิด WAV ฟังให้ครบ โดยเฉพาะรอยต่อประโยคและช่วงท้าย

ถ้ารอเกิน 120 วินาที สคริปต์จะพยายาม cancel งานและหยุด คำสั่ง cancel ไม่ยืนยันว่าหยุดได้ทันทีทุกกรณี ให้ตรวจสถานะงานใน RunPod ด้วย

## 4. อ่านผลก่อนทดลองต่อ

| ฟิลด์ | ความหมาย |
|---|---|
| end_to_end_seconds | ตั้งแต่ส่งงานจนรับผล รวมการรอและ poll; ใช้ตัดสินเกณฑ์ 60 วินาที |
| delay_seconds | RunPod delayTime รวมการรอ/เตรียม Worker |
| execution_seconds | RunPod executionTime |
| model_load_seconds | โหลดและเตรียม Nano engine ตอนเริ่ม process; อาจทับซ้อนกับ delayTime อย่าบวกซ้ำ |
| generation_seconds | สร้างทุก segment หลังเตรียมเสียงอ้างอิง |
| reference_encode_seconds | แปลงเสียงต้นฉบับเป็น latents |
| reference_cache_hit | true หมายถึงใช้ข้อมูลเสียงเดิมซ้ำภายใน Worker |
| worker_id / request_index | ช่วยแยกว่าใช้ process เดิมหรือเริ่มใหม่ |

งานแรกบน process ใหม่อาจยังมีเวลา CUDA warm-up/graph capture แม้ไม่มีการเจนทิ้ง 3 รอบ ถ้าส่วนเริ่มใหม่ยังเกินเป้า ให้ทดลอง Endpoint อีกตัวด้วย `NANO_ENFORCE_EAGER=true` แล้ววัด **เวลารวม** และต้นทุนเทียบกัน โหมดนี้อาจเริ่มเร็วขึ้นแต่สร้างเสียงช้าลง จึงต้องวัดทั้งสองส่วน

## 5. ทดสอบ 1,000 ตัวอักษรเมื่อ 700 ผ่าน

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\benchmark-nano.ps1 -EndpointId "ENDPOINT_ID" -ReferenceAudio ".\myvoice.wav" -ReferenceTextFile ".\reference_text.txt" -TextFile ".\samples\thai-1000.txt"
```

เก็บหลายผล ทั้ง worker ใหม่และ process เดิม ตรวจ **เวลารวม <60 วินาที**, ความเหมือนเสียง, คำอ่าน, ไม่มีข้อความหาย/ซ้ำ/ตัดท้าย และค่าใช้จ่ายจริงต่อครั้ง

ไฟล์ตัวอย่างมีจำนวน 700/1,000 ตัวอักษร ใช้สำหรับวัดเวลา ควรเพิ่มบทจริงของ Boss ก่อนตัดสินคุณภาพ

Idle 5 วินาทีและ Active 0 ยังมีค่าเริ่ม Worker ค่าเวลาประมวลผล และเวลารอปิดสั้น ๆ รวมถึงค่า storage ที่เกี่ยวข้อง ไม่ใช่ไม่มีค่าใช้จ่ายเลย และระบบที่ scale to zero ไม่สามารถรับประกันต่ำกว่า 60 วินาทีทุกครั้งเมื่อ GPU ไม่ว่าง

## 6. ต่อกับ AI Voice Studio หลังผ่าน

Input เดิมรองรับ `reference_audio_url`, `reference_text`, `output_upload_url`, `modifier`, `cfg_value`, `inference_timesteps=10` และคืน WAV เข้า R2 ผ่าน PUT URL เดิม ไม่มีการย้ายเสียงไป provider อื่น

เปลี่ยน `RUNPOD_ENDPOINT_ID` ใน Cloudflare เป็น ID ใหม่เมื่อผลทดสอบผ่านแล้ว และทดสอบสร้างเสียงผ่านเว็บจริงอีกครั้ง ชุดทดลองนี้ไม่ได้แก้ schema ของ Voice library หรือฐานข้อมูล

ค่า `dialect` รับได้เพื่อความเข้ากันได้ แต่เอนจินไม่ได้เพิ่มการควบคุมสำเนียงเอง สำเนียงขึ้นกับเสียงอ้างอิงและข้อความ

ถ้าผลไม่ผ่าน ให้คง Endpoint และ image 0.1.4 เดิมไว้เป็นทางกลับ เปลี่ยนเฉพาะ Endpoint ID กลับ ไม่มีการแก้ payment/credit หรือรวม Video Generate ในชุดนี้

## แหล่งอ้างอิงและ dependency

- RunPod Cached Models: https://docs.runpod.io/serverless/endpoints/model-caching
- RunPod request/result payload limits: https://docs.runpod.io/serverless/endpoints/operation-reference
- RunPod handler payload guidance: https://docs.runpod.io/serverless/workers/handler-functions
- Nano-vLLM API ที่ pin: https://github.com/a710128/nanovllm-voxcpm/tree/c702631ea33db63932f0c2288cc09a543e68a3ef
- FlashAttention wheel official release 2.8.3: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3

FlashAttention ติดตั้งด้วย prebuilt wheel ที่ตรงกับ PyTorch/CUDA/Python/ABI ตอน build บน GitHub ไม่ compile ตอนรับงาน GPU จริง
