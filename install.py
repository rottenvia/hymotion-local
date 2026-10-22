"""Install hymotion-local: HY-Motion 1.0's code (a pinned commit), a Python env with CUDA torch, and the
weights (~22 GB: the motion model 4.2 GB, Qwen3-8B 16.4 GB, CLIP-L 1.7 GB). Standard library only, no
git needed. Safe to run again: whatever is already in place is kept.

  python install.py --accept-license [--home DIR] [--cuda cu126] [--system-torch] [--no-weights]

  --accept-license  you accept the Tencent HY-Motion 1.0 Community License (read it first:
                    https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/main/License.txt). It does not
                    apply in the EU, UK or South Korea; services over 1M monthly users need Tencent's OK.
  --home DIR        where HY-Motion, its env and weights go (default ./HY-Motion-1.0 here); an existing
                    checkout is reused as it is
  --cuda TAG        the PyTorch wheel index (cu126 default; cu128 for RTX 50xx)
  --system-torch    build the env on this Python's own packages (a CUDA torch already installed here)
  --no-weights      code and env only
"""
import argparse
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent
COMMIT = "4e426f5a1021cbcf7f375458c37b840ee7225229"
ZIP = f"https://github.com/Tencent-Hunyuan/HY-Motion-1.0/archive/{COMMIT}.zip"
MEDIA = f"https://media.githubusercontent.com/media/Tencent-Hunyuan/HY-Motion-1.0/{COMMIT}/"
LFS = [f"scripts/gradio/static/assets/dump_wooden/{n}" for n in
       ("faces.bin", "j_template.bin", "keypoints.bin", "kintree.bin", "skinIndice.bin", "skinWeights.bin",
        "uvs.bin", "v_template.bin")] + ["stats/Mean.npy", "stats/Std.npy"]
WEIGHTS = [("tencent/HY-Motion-1.0", ["HY-Motion-1.0/*"], "ckpts/tencent"),
           ("openai/clip-vit-large-patch14", ["*.json", "*.txt", "model.safetensors"], "ckpts/clip-vit-large-patch14"),
           ("Qwen/Qwen3-8B", ["*.json", "*.txt", "*.safetensors"], "ckpts/Qwen3-8B")]


def sh(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-license", action="store_true")
    ap.add_argument("--home")
    ap.add_argument("--cuda", default="cu126")
    ap.add_argument("--system-torch", action="store_true")
    ap.add_argument("--no-weights", action="store_true")
    a = ap.parse_args()
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is needed.")
    home = Path(a.home).resolve() if a.home else TOOL / "HY-Motion-1.0"

    if not (home / "hymotion").is_dir():  # 1. the code, at the commit this tool was written against
        if home.exists() and any(home.iterdir()):
            sys.exit(f"{home} exists and is not a HY-Motion checkout.")
        print(f">> HY-Motion code -> {home}", flush=True)
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ZIP).read())) as z:
            top = z.namelist()[0].split("/")[0]
            for m in z.infolist():
                rel = m.filename[len(top) + 1:]
                if rel and not m.is_dir():
                    (home / rel).parent.mkdir(parents=True, exist_ok=True)
                    (home / rel).write_bytes(z.read(m))
    for rel in LFS:  # 2. its git-lfs files (motion stats, body template): pointers in an archive
        p = home / rel
        if not p.exists() or p.read_bytes()[:40].startswith(b"version https://git-lfs"):
            print(f">> {rel}", flush=True)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(urllib.request.urlopen(MEDIA + rel).read())

    venv = home / ".venv"  # 3. the env
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        sh(sys.executable, "-m", "venv", *(["--system-site-packages"] if a.system_torch else []), venv)
    cuda = subprocess.run([str(py), "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                          capture_output=True).returncode == 0
    if not cuda:
        sh(py, "-m", "pip", "install", "torch", "--index-url", f"https://download.pytorch.org/whl/{a.cuda}")
    sh(py, "-m", "pip", "install", "-q", "-r", TOOL / "requirements.txt")

    if not a.no_weights:  # 4. the weights
        if not a.accept_license:
            sys.exit("The weights are under the Tencent HY-Motion 1.0 Community License:\n"
                     "https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/main/License.txt\n"
                     "It does not apply in the EU, UK or South Korea; services over 1M monthly users need\n"
                     "Tencent's OK. Read it, then run again with --accept-license to download them.")
        code = ("import sys\nfrom huggingface_hub import snapshot_download\n"
                f"for repo, pats, to in {WEIGHTS!r}:\n"
                "    print('>>', repo, flush=True)\n"
                f"    snapshot_download(repo, allow_patterns=pats, local_dir={str(home)!r} + '/' + to, max_workers=8)\n")
        sh(py, "-c", code)

    (TOOL / "config.json").write_text(json.dumps({"home": str(home)}, indent=1))
    print("$ motion.py doctor", flush=True)
    sys.exit(subprocess.run([str(py), str(TOOL / "motion.py"), "doctor"]).returncode)


if __name__ == "__main__":
    main()
