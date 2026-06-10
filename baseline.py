import time
import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
print(f"Using model: {model_id}")



prompt = "A futuristic cyberpunk city at night, highly detailed, cinematic lighting"

device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
print("Pipeline loaded successfully")

pipe.set_progress_bar_config(disable=False)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
print("VRAM stats reset, starting generation...")

start = time.time()
print("Starting image generation...")

image = pipe(
    prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

torch.cuda.synchronize()
end = time.time()
print("Image generation completed")
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"Generation time: {end - start:.2f} seconds")
print(f"Peak VRAM: {peak_vram:.2f} GB")

image.save("baseline_output.png")
print("Saved as baseline_output.png")