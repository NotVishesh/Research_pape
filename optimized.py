"""
EMOD Optimized Experiments
Runs two memory-optimization variants and logs results alongside baseline.
"""
import time
import os
import csv
import torch
from diffusers import StableDiffusionPipeline

os.makedirs("results", exist_ok=True)

CHECKPOINT = "./models/sd15/v1-5-pruned-emaonly.safetensors"
DIFFUSERS_DIR = "./models/sd15_diffusers"
PROMPT = "A futuristic cyberpunk city at night, highly detailed, cinematic lighting"
STEPS = 20
CFG = 7.5
HEIGHT = 512
WIDTH = 512

EXPERIMENTS = [
    {
        "name": "attn_slicing",
        "label": "Attention Slicing",
        "apply": lambda p: p.enable_attention_slicing(),
        "to_cuda": True,
    },
    {
        "name": "cpu_offload",
        "label": "Sequential CPU Offload",
        "apply": lambda p: p.enable_sequential_cpu_offload(),
        "to_cuda": False,
    },
]


def load_pipeline():
    print("[load_pipeline] Checking for diffusers directory...")
    if os.path.exists(os.path.join(DIFFUSERS_DIR, "model_index.json")):
        print(f"[load_pipeline] Loading from pretrained diffusers dir: {DIFFUSERS_DIR}")
        return StableDiffusionPipeline.from_pretrained(
            DIFFUSERS_DIR,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    print(f"[load_pipeline] Loading from single file checkpoint: {CHECKPOINT}")
    return StableDiffusionPipeline.from_single_file(
         "models/sd15/v1-5-pruned-emaonly.safetensors",
         original_config_file="configs/v1-inference.yaml",   # add this line
        # CHECKPOINT,
        torch_dtype=torch.float16,
    )


print("=== EMOD Optimized Experiments ===")
print(f"[startup] Current working directory: {os.getcwd()}")
print(f"[startup] Results directory ready: {os.path.abspath('results')}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

csv_path = "results/results.csv"
write_header = not os.path.exists(csv_path)
print(f"[startup] CSV path: {csv_path} | write_header={write_header}")

for exp in EXPERIMENTS:
    print(f"--- {exp['label']} ---")
    print(f"[experiment] Name: {exp['name']}")
    print("[experiment] Loading pipeline...")
    pipe = load_pipeline()
    print("[experiment] Pipeline loaded")
    print("[experiment] Applying optimization...")
    exp["apply"](pipe)
    print(f"[experiment] Optimization applied: {exp['name']}")
    if exp["to_cuda"]:
        print("[experiment] Moving pipeline to CUDA...")
        pipe = pipe.to("cuda")
        print("[experiment] Pipeline moved to CUDA")
    else:
        print("[experiment] Skipping explicit CUDA move (handled by optimization)")

    print("[experiment] Resetting peak memory stats...")
    torch.cuda.reset_peak_memory_stats()
    print("[experiment] Synchronizing CUDA before generation...")
    torch.cuda.synchronize()

    print("[experiment] Starting generation...")
    start = time.time()
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        height=HEIGHT,
        width=WIDTH,
    ).images[0]
    print("[experiment] Generation complete")
    print("[experiment] Synchronizing CUDA after generation...")
    torch.cuda.synchronize()
    elapsed = time.time() - start

    gen_time = round(elapsed, 2)
    peak_vram = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    print(f"  Time: {gen_time} s  |  Peak VRAM: {peak_vram} GB")

    out_img = f"results/{exp['name']}_output.png"
    print(f"[experiment] Saving image to: {out_img}")
    image.save(out_img)
    print(f"  Saved: {out_img}")

    print(f"[experiment] Writing CSV row to: {csv_path}")
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            print("[experiment] Writing CSV header")
            w.writerow(["experiment", "gen_time_s", "peak_vram_gb", "steps", "resolution"])
            write_header = False
        w.writerow([exp["name"], gen_time, peak_vram, STEPS, f"{WIDTH}x{HEIGHT}"])
    print("[experiment] CSV write complete")

    print("[experiment] Cleaning up pipeline and cache...")
    del pipe
    torch.cuda.empty_cache()
    print()

print(f"All results saved to {csv_path}")
