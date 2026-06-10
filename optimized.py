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
    if os.path.exists(os.path.join(DIFFUSERS_DIR, "model_index.json")):
        return StableDiffusionPipeline.from_pretrained(
            DIFFUSERS_DIR,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    return StableDiffusionPipeline.from_single_file(
        CHECKPOINT,
        torch_dtype=torch.float16,
    )


print("=== EMOD Optimized Experiments ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

csv_path = "results/results.csv"
write_header = not os.path.exists(csv_path)

for exp in EXPERIMENTS:
    print(f"--- {exp['label']} ---")
    pipe = load_pipeline()
    exp["apply"](pipe)
    if exp["to_cuda"]:
        pipe = pipe.to("cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.time()
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        height=HEIGHT,
        width=WIDTH,
    ).images[0]
    torch.cuda.synchronize()
    elapsed = time.time() - start

    gen_time = round(elapsed, 2)
    peak_vram = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    print(f"  Time: {gen_time} s  |  Peak VRAM: {peak_vram} GB")

    out_img = f"results/{exp['name']}_output.png"
    image.save(out_img)
    print(f"  Saved: {out_img}")

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["experiment", "gen_time_s", "peak_vram_gb", "steps", "resolution"])
            write_header = False
        w.writerow([exp["name"], gen_time, peak_vram, STEPS, f"{WIDTH}x{HEIGHT}"])

    del pipe
    torch.cuda.empty_cache()
    print()

print(f"All results saved to {csv_path}")
