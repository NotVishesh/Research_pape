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


def load_pipeline():
    if os.path.exists(os.path.join(DIFFUSERS_DIR, "model_index.json")):
        print("Loading from converted diffusers folder...")
        return StableDiffusionPipeline.from_pretrained(
            DIFFUSERS_DIR,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    print("Loading from safetensors checkpoint (first run may download configs ~200 MB)...")
    return StableDiffusionPipeline.from_single_file(
        CHECKPOINT,
        torch_dtype=torch.float16,
    )


print("=== EMOD Baseline Experiment ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

pipe = load_pipeline().to("cuda")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start = time.time()
image = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=CFG, height=HEIGHT, width=WIDTH).images[0]
torch.cuda.synchronize()
elapsed = time.time() - start

gen_time = round(elapsed, 2)
peak_vram = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

print(f"\nGeneration time : {gen_time} s")
print(f"Peak VRAM       : {peak_vram} GB")

out_img = "results/baseline_output.png"
image.save(out_img)
print(f"Image saved     : {out_img}")

csv_path = "results/results.csv"
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["experiment", "gen_time_s", "peak_vram_gb", "steps", "resolution"])
    w.writerow(["baseline", gen_time, peak_vram, STEPS, f"{WIDTH}x{HEIGHT}"])

print(f"Results logged  : {csv_path}")
