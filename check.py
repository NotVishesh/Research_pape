"""Environment check — run this first to verify everything is ready."""
import sys
import os

print("=== EMOD Environment Check ===\n")

print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}\n")

# PyTorch / CUDA
try:
    import torch
    print(f"torch        : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version  : {torch.version.cuda}")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM          : {props.total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: CUDA not available. Fix with:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
except ImportError:
    print("torch: NOT INSTALLED")

print()

# diffusers / transformers
for pkg in ["diffusers", "transformers", "accelerate", "safetensors", "bitsandbytes", "torchao"]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{pkg:<15}: {ver}")
    except ImportError:
        print(f"{pkg:<15}: NOT INSTALLED")

print()

# Model files
CHECKPOINT = "./models/sd15/v1-5-pruned-emaonly.safetensors"
DIFFUSERS_DIR = "./models/sd15_diffusers"

if os.path.exists(CHECKPOINT):
    size_gb = os.path.getsize(CHECKPOINT) / 1024**3
    print(f"Checkpoint    : FOUND ({size_gb:.2f} GB)")
else:
    print(f"Checkpoint    : NOT FOUND at {CHECKPOINT}")
    print("  -> Download v1-5-pruned-emaonly.safetensors from Hugging Face manually")

if os.path.exists(os.path.join(DIFFUSERS_DIR, "model_index.json")):
    print(f"Diffusers dir : FOUND ({DIFFUSERS_DIR})")
else:
    print(f"Diffusers dir : not converted yet (optional if checkpoint exists)")
    print("  -> To convert run:")
    print("     python tools\\convert_sd.py --checkpoint_path models\\sd15\\v1-5-pruned-emaonly.safetensors --dump_path models\\sd15_diffusers --from_safetensors")

print()
print("=== Ready to run? ===")
ok = (
    "torch" in sys.modules
    and __import__("torch").cuda.is_available()
    and (os.path.exists(CHECKPOINT) or os.path.exists(os.path.join(DIFFUSERS_DIR, "model_index.json")))
)
if ok:
    print("YES — run:  python baseline.py")
    print("Then run:   python optimized.py")
    print("Then run:   python plot_results.py")
else:
    print("NOT YET — fix the issues above first.")
