import base64, glob, io, os, shutil, tempfile, time
from pathlib import Path
import requests, runpod, soundfile as sf, torch
from voxcpm import VoxCPM

DEVICE=os.getenv("VOXCPM_DEVICE","cuda")
VOL=Path(os.getenv("RUNPOD_VOLUME_PATH","/runpod-volume"))
CLEAN=Path(os.getenv("VOXCPM_VOLUME_MODEL_DIR",str(VOL/"voxcpm2-local")))
HFVOL=Path(os.getenv("VOXCPM_VOLUME_HF_CACHE",str(VOL/"huggingface-cache")))
RUNTIME=Path(os.getenv("VOXCPM_RUNTIME_DIR","/tmp/voxcpm2-runtime"))
OPT=os.getenv("VOXCPM_OPTIMIZE","false").lower() in ("1","true","yes","on")
MAX_CHARS=int(os.getenv("VOXCPM_MAX_TEXT_CHARS","2000"))

def find_weight(name):
    pats=[str(HFVOL/"models--openbmb--VoxCPM2"/"snapshots"/"*"/name),str(HFVOL/"**"/name)]
    for pat in pats:
        for m in glob.glob(pat,recursive=True):
            p=Path(m)
            if p.is_file() and p.stat().st_size>0:return p
    return None

def prepare_model():
    if not CLEAN.is_dir(): return "openbmb/VoxCPM2"
    RUNTIME.mkdir(parents=True,exist_ok=True)
    for n in ["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","tokenization_voxcpm2.py"]:
        s=CLEAN/n
        if not s.is_file() or s.stat().st_size==0: raise RuntimeError(f"Missing model file: {s}")
        shutil.copy2(s,RUNTIME/n)
    for n in ["model.safetensors","audiovae.pth"]:
        s=find_weight(n)
        if s is None: raise RuntimeError(f"Missing weight: {n}")
        d=RUNTIME/n
        if d.exists() or d.is_symlink(): d.unlink()
        os.symlink(s,d)
    return str(RUNTIME)

def load_model():
    if DEVICE.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA not available")
    src=prepare_model(); t=time.time()
    m=VoxCPM.from_pretrained(src,cache_dir="/tmp/hf-cache",load_denoiser=False,optimize=OPT,device=DEVICE)
    print(f"[startup] VoxCPM2 ready in {time.time()-t:.2f}s optimize={OPT}")
    return m,time.time()-t

MODEL,LOAD_S=load_model()
SR=MODEL.tts_model.sample_rate

def ref_file(inp,tmp):
    b64=inp.get("reference_audio_b64"); url=inp.get("reference_audio_url")
    if b64 and url: raise ValueError("Use only one reference audio source")
    if not b64 and not url:return None
    p=Path(tmp)/"reference.wav"
    if b64:
        if b64.strip().lower().startswith("data:"): b64=b64.split(",",1)[1]
        p.write_bytes(base64.b64decode(b64,validate=True))
    else:
        r=requests.get(url,timeout=60); r.raise_for_status(); p.write_bytes(r.content)
    return str(p)

def wav_bytes(w):
    b=io.BytesIO(); sf.write(b,w,SR,format="WAV",subtype="PCM_16"); return b.getvalue()

def handler(job):
    inp=job.get("input") or {}
    action=str(inp.get("action","generate")).lower()
    if action=="health":
        return {"ok":True,"engine":"voxcpm2","gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","sample_rate":SR,"optimize":OPT}
    if action!="generate": raise ValueError("action must be generate or health")
    text=str(inp.get("text","")).strip()
    if not text: raise ValueError("input.text is required")
    if len(text)>MAX_CHARS: raise ValueError(f"text exceeds {MAX_CHARS} chars")
    cfg=float(inp.get("cfg_value",2.0)); steps=int(inp.get("inference_timesteps",10))
    t=time.time()
    with tempfile.TemporaryDirectory(prefix="voxcpm2-") as tmp:
        ref=ref_file(inp,tmp)
        kw={"text":text,"cfg_value":cfg,"inference_timesteps":steps}
        if ref: kw["reference_wav_path"]=ref
        wav=MODEL.generate(**kw)
        elapsed=time.time()-t; duration=len(wav)/SR; data=wav_bytes(wav)
        out={"ok":True,"engine":"voxcpm2","mode":"voice_clone" if ref else "tts","sample_rate":SR,"duration_seconds":round(duration,3),"generation_seconds":round(elapsed,3),"rtf":round(elapsed/duration,3) if duration else None}
        up=inp.get("output_upload_url")
        if up:
            r=requests.put(up,data=data,headers={"Content-Type":"audio/wav"},timeout=60); r.raise_for_status(); out["output"]="uploaded"
        else:
            out["output"]="base64"; out["audio_b64"]=base64.b64encode(data).decode("ascii")
        return out

if __name__=="__main__":
    runpod.serverless.start({"handler":handler})
