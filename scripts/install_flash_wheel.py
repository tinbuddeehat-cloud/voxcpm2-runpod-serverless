"""Install an official prebuilt FlashAttention wheel; never compile on a GPU."""
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

version = "2.8.3"
if platform.machine() != "x86_64" or sys.platform != "linux":
    raise RuntimeError("This image requires Linux x86_64")
if not torch.__version__.startswith("2.8.") or not str(torch.version.cuda).startswith("12."):
    raise RuntimeError("FlashAttention wheel requires PyTorch 2.8 / CUDA 12")
cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
name = f"flash_attn-{version}+cu12torch2.8cxx11abi{abi}-{cp}-{cp}-linux_x86_64.whl"
url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/{name.replace('+', '%2B')}"
with tempfile.TemporaryDirectory() as tmp:
    wheel = Path(tmp) / name
    urllib.request.urlretrieve(url, wheel)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
