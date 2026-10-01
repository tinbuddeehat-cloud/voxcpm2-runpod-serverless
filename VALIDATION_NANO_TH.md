# ผลตรวจชุดทดลอง Nano-vLLM

ตรวจวันที่ 1 ตุลาคม 2026

ฐาน worker: repository `tinbuddeehat-cloud/voxcpm2-runpod-serverless`, main commit `d4ca3b0d364c240902fc40c167705ea99fd2e256` (v0.1.4)

ชุด PATCH เพิ่มไฟล์ทดลองเท่านั้น ไม่แก้ handler.py, Dockerfile, workflow build-ghcr.yml หรือ tag latest ของ production

## ผ่านแล้ว

- Python compileall ของ nano, scripts และ tests
- Unit tests 16 รายการด้วย Nano engine จำลอง: cached model path, pinned revision, cache ขาดแล้วไม่ดาวน์โหลดใหม่, ขอบเขตเสียงอ้างอิง, clone conditioning, encode ครั้งเดียวต่อเสียง/ใช้ซ้ำทุก segment, LRU cache, WAV รวม, output PUT contract, upload failure, payload guard และข้อความไทย 700/1,000 ตัวอักษรไม่หายเมื่อแบ่ง
- FFmpeg จริง: WAV mono 48 kHz 1 วินาที ผ่าน speed=1.25, pitch=2 semitones, volume=1.2 ได้ WAV 48 kHz mono ความยาวประมาณ 0.803 วินาที
- Workflow YAML อ่านได้ และ PowerShell ผ่านการ parse ด้วย tree-sitter grammar; ยังไม่ใช่การรันด้วย PowerShell บน Windows
- ตรวจ source API ของ Nano-vLLM ที่ commit `c702631ea33db63932f0c2288cc09a543e68a3ef`; sync engine ทำงานใน thread เดียวแยกจาก asyncio ของ RunPod

## ยังไม่มีผลตรวจ

- Docker container build / import FlashAttention และ Nano ใน image จริง
- PowerShell syntax/runtime บน Windows; GitHub workflow มีขั้นตรวจ syntax ด้วย PowerShell ก่อน build
- VRAM, cold startup, การสร้างเสียงจริง, คุณภาพภาษาไทยและความเหมือนเสียง
- End-to-end latency บน RunPod และค่าใช้จ่ายจริงต่อคำขอ
- AI Voice Studio → RunPod → R2 ทั้งเส้นทางบน deployment จริง

เครื่องมือนี้ไม่มี Docker, PowerShell หรือ GPU และ GitHub connector ตอบ HTTP 403 เมื่อสร้าง branch จึงไม่มี remote commit, PR, Actions run หรือ image ที่เผยแพร่จากชุดนี้

## เกณฑ์ก่อนเปลี่ยน production endpoint

1. Workflow validate + build ผ่าน และ RunPod pull image ใหม่ได้
2. 700 และ 1,000 ตัวอักษรได้เสียงครบ ฟังรอยต่อและท้ายประโยค ไม่มีหาย/ซ้ำ/ตัดจบ
3. ทดสอบทั้งคำขอแรกหลัง worker หยุดและคำขอซ้ำ เก็บ end_to_end_seconds หลายผล พร้อม worker_id/request_index
4. เป้าหมาย <60 วินาทีต้องนับเวลารอ worker, model startup, reference encode, generate และรับไฟล์ครบ ไม่ใช้เฉพาะ generation_seconds
5. ตรวจการสร้างเสียงจากเว็บจริง การบันทึก R2 และการคิดเครดิตตามเดิม

ผล unit tests ไม่ใช่หลักฐานว่ารุ่นนี้พร้อม production หรือสร้างข้อความไทยได้ภายใน 60 วินาที
