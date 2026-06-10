# EMOD: Efficient Memory Optimization for Diffusion Models
### An Empirical Study of VRAM Reduction Techniques for Stable Diffusion Inference on Consumer GPUs

**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Model:** Stable Diffusion v1.5  
**Resolution:** 512 × 512 pixels  
**Inference Steps:** 20  

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Background and Terminology](#3-background-and-terminology)
4. [Experimental Setup](#4-experimental-setup)
5. [Optimization Techniques Explained](#5-optimization-techniques-explained)
6. [Results](#6-results)
7. [Analysis and Discussion](#7-analysis-and-discussion)
8. [Conclusion](#8-conclusion)
9. [Glossary](#9-glossary)

---

## 1. Abstract

Large generative AI image models such as Stable Diffusion require significant GPU memory (VRAM) to run, making them inaccessible on many consumer-grade machines. This paper presents **EMOD** (Efficient Memory Optimization for Diffusion models), an empirical evaluation of two widely-used memory optimization strategies — **Attention Slicing** and **Sequential CPU Offloading** — applied to Stable Diffusion v1.5 on a 6 GB laptop GPU. We measure the trade-off between peak VRAM consumption and image generation time for each strategy. Our results show that Sequential CPU Offloading reduces peak VRAM by **76.9%** (from 2.626 GB to 0.606 GB) at the cost of a **3.77× slowdown** in generation time, while Attention Slicing provides no measurable VRAM reduction at 512×512 resolution but adds modest latency overhead. These findings provide practical guidance for deploying diffusion models on memory-constrained consumer hardware.

---

## 2. Introduction

### The Problem

Generative AI image models have become extremely powerful, but they come with a heavy cost: **they require a lot of GPU memory (VRAM) to run**. Stable Diffusion v1.5, one of the most popular open-source image generation models, needs a minimum of around 4–6 GB of VRAM just to load. For context, many consumer laptops ship with GPUs that have only 4–8 GB of VRAM. This creates a real barrier for students, researchers, and developers who want to work with these models but do not have access to high-end hardware.

The challenge is not just loading the model — it is running inference (i.e., generating an image) efficiently. During inference, multiple large tensors (multi-dimensional arrays of numbers) must live in VRAM simultaneously: the model weights, intermediate feature maps, attention matrices, and the evolving image itself. All of these compete for the same limited memory pool.

### Why This Matters

- **Accessibility:** If a model only runs on expensive datacenter GPUs, it is not accessible to most people.
- **Edge deployment:** Many real-world applications (mobile apps, embedded devices, offline tools) need to run AI models on hardware with limited memory.
- **Research:** Understanding memory-latency trade-offs helps researchers make informed decisions about which optimization to apply in different scenarios.

### What This Paper Does

We take Stable Diffusion v1.5 and run it three ways on a 6 GB laptop GPU:
1. **Baseline** — no memory optimizations, just the raw model.
2. **Attention Slicing** — a technique that reduces peak memory during the attention computation phase.
3. **Sequential CPU Offloading** — a technique that moves model components off the GPU and onto regular RAM when they are not in use.

We measure two things for each: how much VRAM was used at peak, and how long it took to generate one image.

---

## 3. Background and Terminology

This section explains every concept you need to understand before reading the rest of the paper.

### 3.1 What is a Diffusion Model?

A **diffusion model** is a type of generative AI model that learns to create images by learning how to reverse a "noising" process. Here is the intuition:

- **Forward process (training time):** Take a real image and gradually add random noise to it over many steps until it becomes pure static (like a TV with no signal). The model watches this happen thousands of times.
- **Reverse process (inference time):** Start with pure random noise and ask the model to slowly "denoise" it — removing a little noise at each step — until a clean image emerges.

The model learns what realistic images look like by learning how to undo noise. At inference time, you start from a random noise image and run the reverse process for a fixed number of **timesteps** (in our experiments, 20 steps). After 20 rounds of denoising, you have a generated image.

This is fundamentally different from older generative models like GANs (Generative Adversarial Networks), which generate images in a single forward pass.

### 3.2 What is Stable Diffusion?

**Stable Diffusion** is a specific family of open-source diffusion models released by Stability AI. The version used in this paper, **v1.5**, was trained on a large dataset of image-text pairs and can generate photorealistic images from text descriptions (called **prompts**).

Stable Diffusion v1.5 has four main components:

| Component | What it does | Approximate size |
|---|---|---|
| **Text Encoder** (CLIP) | Converts your text prompt into a list of numbers the model can understand | ~340 MB |
| **U-Net** | The core denoising network. Does the heavy lifting at every timestep | ~3.4 GB |
| **VAE** (Variational Autoencoder) | Compresses images into a smaller "latent space" and decompresses them back | ~160 MB |
| **Scheduler** | Controls how much noise is removed at each of the 20 timesteps | negligible |

The total model size is roughly 4 GB on disk.

### 3.3 What is VRAM?

**VRAM** (Video RAM) is the dedicated memory built into your GPU. It is separate from your computer's regular RAM (which is on the CPU side). When you run a neural network:

- The **model weights** (the learned numbers that define the model) must be loaded into VRAM.
- During computation, **intermediate activations** (temporary results from each layer) also live in VRAM.
- The **image being generated** (stored as a latent tensor) lives in VRAM throughout.

If your VRAM fills up, the generation either crashes with an out-of-memory (OOM) error or becomes extremely slow because data must be swapped to slower memory.

**Peak VRAM** is the single highest memory reading recorded during a generation run — the worst-case moment when the most data was simultaneously in VRAM.

In our experiments, the GPU has **6 GB of VRAM total**, and the baseline run uses about **2.626 GB peak** for a 512×512 image.

### 3.4 What is the Attention Mechanism?

The **attention mechanism** is one of the core building blocks of modern neural networks, introduced in the "Attention is All You Need" paper (Vaswani et al., 2017). In diffusion models, attention is used so that different parts of the image can "look at" and influence each other during denoising.

Mathematically, attention computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
```

where:
- **Q** (Query), **K** (Key), **V** (Value) are matrices derived from the image features
- The `QK^T` operation creates an attention matrix showing how much each part of the image should attend to every other part
- `d` is the dimension of the features (a scaling factor for numerical stability)

The problem: for large images or large batch sizes, `QK^T` produces a **very large matrix** (size grows quadratically with image resolution). At 512×512, this matrix is already sizable. At higher resolutions it becomes enormous and is the main cause of memory spikes during inference.

### 3.5 What is Latent Space?

Rather than working directly on 512×512 pixel images (which are large), Stable Diffusion compresses images into a **latent space** — a smaller 64×64×4 representation. The VAE encoder compresses the image; the denoising process runs in this compressed space (which is 8× smaller in each spatial dimension); then the VAE decoder expands it back to the full pixel image.

This is why Stable Diffusion is called a **Latent Diffusion Model** — the diffusion happens in latent space, not pixel space.

### 3.6 What is Inference vs. Training?

- **Training** is when the model learns from data. This requires storing gradients (extra numbers used to update weights) and is extremely memory-hungry.
- **Inference** is when you use an already-trained model to generate new outputs. No gradients are needed, so it is significantly cheaper — but still expensive at scale.

All experiments in this paper are **inference only**. We never modify the model weights; we only run it to generate images.

### 3.7 What is float16 (half precision)?

Neural network weights are normally stored as **float32** (32-bit floating point numbers, 4 bytes each). By switching to **float16** (16-bit, 2 bytes each), you cut VRAM usage roughly in half with minimal quality loss. All experiments in this paper use float16. This is standard practice for consumer GPU inference and is why our 4 GB model fits comfortably in a 6 GB GPU.

---

## 4. Experimental Setup

### 4.1 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Total VRAM | 6.0 GB GDDR6 |
| GPU Architecture | Ada Lovelace |
| Operating System | Windows 11 |

### 4.2 Software

| Package | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework; handles GPU tensors |
| Diffusers (HuggingFace) | Library that provides the Stable Diffusion pipeline with optimization hooks |
| Transformers (HuggingFace) | Provides the CLIP text encoder |
| Accelerate (HuggingFace) | Backend for CPU offloading logic |
| Safetensors | Efficient, safe model weight file format |

### 4.3 Model

- **Model:** Stable Diffusion v1.5
- **Checkpoint:** `v1-5-pruned-emaonly.safetensors` (the pruned EMA-only weights, ~3.97 GB)
- **Precision:** float16 (half precision) for all components
- **Safety checker:** Disabled (not needed for controlled research experiments)

> **What is EMA?** During training, two sets of weights are maintained: the "live" weights that are updated each step, and an **Exponential Moving Average** (EMA) of those weights, which is a smoothed version that tends to produce higher quality outputs. The "emaonly" checkpoint keeps only the EMA weights, making the file smaller without sacrificing quality.

### 4.4 Generation Parameters

| Parameter | Value | Explanation |
|---|---|---|
| Prompt | "A futuristic cyberpunk city at night, highly detailed, cinematic lighting" | The text description of the image to generate |
| Inference steps | 20 | How many denoising iterations to run. More steps = higher quality but slower |
| Guidance scale | 7.5 | How strictly to follow the prompt vs. be creative. 7.5 is the standard default |
| Resolution | 512 × 512 px | The output image size. SD v1.5 was trained at this resolution |

### 4.5 Measurement Methodology

For each experiment:
1. Load a fresh pipeline (no shared state between runs)
2. Call `torch.cuda.reset_peak_memory_stats()` to clear the VRAM counter
3. Call `torch.cuda.synchronize()` to ensure all GPU operations have finished before starting the timer
4. Run one image generation, recording wall-clock time
5. Call `torch.cuda.synchronize()` again before stopping the timer
6. Read `torch.cuda.max_memory_allocated()` for peak VRAM

**Why synchronize?** CUDA operations are **asynchronous** by default — the CPU queues up work for the GPU but does not wait for it to finish before moving on. Calling `synchronize()` forces the CPU to wait until all GPU work is complete, ensuring we measure true GPU time rather than just the time to enqueue work.

---

## 5. Optimization Techniques Explained

### 5.1 Baseline (No Optimization)

The baseline runs Stable Diffusion exactly as it comes out of the box: all model components (text encoder, U-Net, VAE) are loaded onto the GPU in float16, and stay there for the entire generation. This is the simplest setup and gives us our reference point for both speed and memory usage.

### 5.2 Attention Slicing

**The problem it solves:** As mentioned in Section 3.4, the attention mechanism creates large intermediate matrices. When generating images at 512×512 or higher, these matrices can cause memory spikes — brief moments where VRAM usage jumps high.

**How it works:** Instead of computing the full attention matrix in one shot, Attention Slicing splits it into smaller chunks ("slices") and processes them one at a time. Each slice is computed, its contribution is added to the running total, and then it is discarded from memory before the next slice is computed. The final result is mathematically identical to computing the full matrix at once.

```
Full attention (baseline):
  [Q] [K] [V] → compute full QK^T matrix → all in VRAM at once → [output]

Sliced attention:
  [Q] [K] [V] → slice 1 of QK^T → add to output → discard slice
              → slice 2 of QK^T → add to output → discard slice
              → ...  → [output]
```

**The trade-off:** Processing slices sequentially rather than in parallel means more GPU kernel launches and some overhead. This slightly increases generation time. The benefit is lower peak VRAM — but only when attention is the dominant memory consumer. At 512×512 with float16, the model weights themselves (not attention) are the main VRAM occupant.

**Enabled with:** `pipe.enable_attention_slicing()`

### 5.3 Sequential CPU Offloading

**The problem it solves:** The GPU does not need all four model components simultaneously. At any given moment during generation, only one component is active:
- Step 1 of generation: the Text Encoder runs (then is idle)
- Steps 1–20 of denoising: the U-Net runs repeatedly (text encoder and VAE are idle)
- Final step: the VAE decoder runs (U-Net is now idle)

Keeping all components loaded in VRAM the entire time wastes memory on idle components.

**How it works:** Sequential CPU Offloading uses HuggingFace Accelerate to hook into the model's execution. Before each component (text encoder, U-Net, VAE) is about to run, its weights are moved from CPU RAM → GPU VRAM. When that component finishes, its weights are moved back from GPU VRAM → CPU RAM. This way, only the currently-active component occupies VRAM at any time.

```
Baseline memory layout:
  VRAM: [Text Encoder] [U-Net] [VAE]   ← all loaded, all the time (2.6 GB)

CPU Offload memory layout:
  Step: Text Encode  →  VRAM: [Text Encoder]        (others in RAM)
  Step: Denoise      →  VRAM: [U-Net]               (others in RAM)
  Step: Decode       →  VRAM: [VAE]                 (others in RAM)
  Peak VRAM ≈ size of largest component (U-Net, ~0.6 GB in float16)
```

**The trade-off:** Moving data between CPU RAM and VRAM over the PCIe bus (the connection between CPU and GPU) is slow — much slower than the GPU's internal memory bandwidth. With 20 denoising steps, the U-Net must be loaded and unloaded 20 times, accumulating significant transfer overhead.

**Enabled with:** `pipe.enable_sequential_cpu_offload()` (this also handles moving to CUDA internally, so we do not call `.to("cuda")` separately)

---

## 6. Results

### 6.1 Raw Data

| Experiment | Generation Time (s) | Peak VRAM (GB) | Steps | Resolution |
|---|---|---|---|---|
| Baseline (run 1) | 3.44 | 2.626 | 20 | 512×512 |
| Baseline (run 2) | 3.31 | 2.626 | 20 | 512×512 |
| **Baseline (avg)** | **3.38** | **2.626** | 20 | 512×512 |
| Attention Slicing | 4.41 | 2.626 | 20 | 512×512 |
| CPU Offload | 12.73 | 0.606 | 20 | 512×512 |

### 6.2 Comparison vs. Baseline

| Optimization | Time Change | VRAM Change | Time (× baseline) | VRAM saved |
|---|---|---|---|---|
| Attention Slicing | +1.03 s | 0.000 GB | 1.31× slower | 0% |
| CPU Offload | +9.35 s | −2.020 GB | 3.77× slower | **76.9%** |

### 6.3 Summary Chart Description

The results chart (generated by `plot_results.py`) shows two side-by-side bar charts:

- **Left chart (Peak VRAM):** Baseline and Attention Slicing both sit at 2.626 GB — their bars are identical in height. CPU Offload drops dramatically to 0.606 GB.
- **Right chart (Generation Time):** Baseline averages ~3.38 s. Attention Slicing is slightly higher at 4.41 s. CPU Offload towers at 12.73 s — nearly 4× the baseline time.

This visual immediately communicates the core trade-off: the only technique that meaningfully reduces VRAM comes at a steep latency cost.

---

## 7. Analysis and Discussion

### 7.1 Why Attention Slicing Did Not Reduce VRAM

This is the most interesting finding in our results. Attention Slicing is documented as a memory-saving technique, yet our measurements show **zero VRAM reduction** compared to baseline (both 2.626 GB).

The explanation lies in understanding what dominates VRAM usage at 512×512:

- The U-Net alone is ~3.4 GB on disk. In float16 it occupies roughly **1.7–2.0 GB** of VRAM.
- The attention matrices at 512×512 latent space (64×64) are relatively small — the quadratic scaling of attention (`O(n²)` where n = 64×64 = 4096 tokens) is not yet problematic at this resolution.
- Therefore, **model weights are the bottleneck**, not attention computation.

Attention Slicing becomes beneficial at much higher resolutions (768×768, 1024×1024 or larger) where the attention matrix itself grows to occupy significant VRAM. At 512×512 — the native training resolution of SD v1.5 — it is overhead without benefit.

The 1.31× slowdown from Attention Slicing is consistent with the cost of breaking parallel computation into serial slices.

> **Key takeaway:** Attention Slicing is the right tool for high-resolution generation on memory-constrained GPUs. At the native 512×512 resolution, it adds latency without saving memory.

### 7.2 CPU Offloading: Dramatic VRAM Savings at a Real Cost

CPU Offloading reduced peak VRAM from 2.626 GB to 0.606 GB — a **76.9% reduction**. This is a striking result. With only 0.606 GB of peak VRAM, Stable Diffusion v1.5 could theoretically run on a GPU with as little as 1–2 GB of VRAM, making it accessible to far more hardware.

The 0.606 GB figure represents the VRAM footprint of the single largest component at peak — which in float16 is approximately the U-Net's largest intermediate activation during one denoising step, not its full weight set (since the rest of the U-Net is swapped back to RAM between sub-operations).

However, the **3.77× slowdown** (3.38s → 12.73s) is a real penalty. This comes from:

1. **PCIe bandwidth bottleneck:** Moving model weights between CPU RAM and GPU VRAM over PCIe 4.0 x16 offers ~32 GB/s theoretical bandwidth. In practice, this is 10–50× slower than the GPU's internal VRAM bandwidth (~576 GB/s for RTX 4050).
2. **20 repeated offloads:** The U-Net is loaded and offloaded at every denoising step. With 20 steps, the transfer penalty compounds.
3. **Sequential nature:** Because components are loaded one at a time, there is no overlap between computation and data transfer.

> **Key takeaway:** CPU Offloading enables Stable Diffusion to run on very low-VRAM GPUs at the cost of ~4× slower generation. For use cases where latency is acceptable (batch generation, overnight jobs, offline tools), this trade-off is worthwhile. For interactive real-time use, it is too slow.

### 7.3 Baseline Reproducibility

The baseline was run twice (3.44s and 3.31s, both 2.626 GB VRAM). The time variation of ~0.13s (3.9%) is expected — it reflects normal system noise such as GPU clock speed variation, background OS tasks, and CUDA kernel launch overhead. The VRAM reading is perfectly consistent, which makes sense since peak VRAM is determined by model size and is not affected by random timing noise.

### 7.4 Practical Implications

| Use Case | Recommended Approach | Reason |
|---|---|---|
| 512×512 generation, latency matters | Baseline or Attention Slicing | Fastest; VRAM is sufficient anyway |
| High-resolution (≥768×768), has 6+ GB VRAM | Attention Slicing | Reduces attention memory spikes at high res |
| Very low VRAM GPU (2–4 GB) | CPU Offloading | Only option that makes the model fit |
| Batch generation overnight | CPU Offloading | Latency is irrelevant; VRAM savings allow other processes to run |
| Real-time interactive use | Baseline | CPU offloading's 12+ s latency is unacceptable |

### 7.5 Limitations and Future Work

1. **Single image per experiment:** We generated one image per configuration. A more rigorous study would average over 10–20 runs to reduce timing variance.

2. **Only two optimizations tested:** The HuggingFace Diffusers library offers additional techniques not tested here:
   - **xFormers memory-efficient attention:** A library (xFormers) that uses a fused CUDA kernel for attention, reducing both memory and time simultaneously. Likely the best of both worlds for attention.
   - **VAE tiling:** Breaks the VAE decode into tiles for very high-resolution output.
   - **Model CPU offload (non-sequential):** A smarter offloading that keeps the entire U-Net in VRAM during denoising and only offloads the text encoder and VAE. Less VRAM savings than sequential, but faster.
   - **Quantization (int8 / int4):** Reducing weights from float16 to 8-bit or 4-bit integers, cutting VRAM by 2–4× with some quality trade-off.

3. **Single prompt and resolution:** Testing across multiple prompts and resolutions would give a fuller picture of how each technique scales.

4. **No image quality measurement:** This study focuses entirely on speed and memory. A complete study would include a perceptual quality metric (FID score, CLIP score) to confirm that optimizations do not degrade output quality.

---

## 8. Conclusion

This paper presented EMOD, an empirical evaluation of memory optimization strategies for Stable Diffusion v1.5 inference on a consumer 6 GB laptop GPU. We compared three configurations — baseline, Attention Slicing, and Sequential CPU Offloading — measuring peak VRAM and generation time.

Our key findings are:

1. **Attention Slicing does not reduce VRAM at 512×512 resolution** because model weights, not attention matrices, are the dominant memory consumer at this resolution. It adds a ~31% latency overhead with no memory benefit at native SD v1.5 resolution.

2. **Sequential CPU Offloading reduces peak VRAM by 76.9%** (2.626 GB → 0.606 GB), enabling Stable Diffusion to run on GPUs with as little as 1–2 GB VRAM. The trade-off is a 3.77× slowdown in generation time (3.38s → 12.73s) due to PCIe bandwidth bottlenecks from repeated CPU-GPU weight transfers.

3. **There is a clear memory-latency trade-off:** no tested technique simultaneously reduces both. The choice of optimization depends entirely on the deployment context — interactive applications should avoid CPU offloading, while memory-constrained or batch-processing scenarios benefit significantly from it.

These results provide concrete empirical guidance for practitioners deploying diffusion models on consumer hardware, and establish a baseline for comparing more advanced techniques such as xFormers attention, model quantization, and hybrid offloading strategies in future work.

---

## 9. Glossary

| Term | Definition |
|---|---|
| **Activation** | Intermediate numerical output of a neural network layer, temporarily stored in VRAM during inference |
| **Attention mechanism** | A mathematical operation that lets a neural network weigh the importance of different input positions relative to each other |
| **Batch size** | Number of images generated simultaneously. We use batch size 1 throughout |
| **CFG / Guidance scale** | Classifier-Free Guidance scale — controls how closely the model follows the text prompt. Higher = more faithful to prompt but less diverse |
| **Checkpoint** | A saved file containing a trained model's weights |
| **CPU offloading** | Technique of moving model components to CPU RAM when not actively in use, freeing GPU VRAM |
| **Denoising** | The reverse diffusion process of iteratively removing noise from an image to produce the final output |
| **Diffusion model** | A generative AI model that learns to reverse a noise-addition process to synthesize new data |
| **EMA (Exponential Moving Average)** | A smoothed average of model weights maintained during training; typically produces higher quality outputs |
| **float16 / half precision** | A 16-bit floating-point number format that uses half the memory of the standard float32, with minimal quality impact |
| **float32 / full precision** | The standard 32-bit floating-point number format used in most numerical computing |
| **GPU** | Graphics Processing Unit — a processor with thousands of small cores optimized for parallel matrix math, the core operation of neural networks |
| **Inference** | Using a trained model to produce predictions or outputs (vs. training, which modifies the model) |
| **Latent space** | A compressed mathematical representation of data; SD v1.5 compresses 512×512 images to 64×64×4 latent tensors |
| **Latent Diffusion Model** | A diffusion model that operates in compressed latent space rather than full pixel space |
| **OOM (Out of Memory)** | Error that occurs when a GPU runs out of VRAM during a computation |
| **PCIe** | PCI Express — the physical slot and bus connecting the CPU and GPU on a motherboard; data moving over PCIe is much slower than data movement within VRAM |
| **Peak VRAM** | The single highest VRAM usage recorded during a generation run |
| **Pipeline** | In HuggingFace Diffusers, the top-level object that bundles all model components and orchestrates the generation process |
| **Prompt** | The text description given to a text-to-image model describing what to generate |
| **Quantization** | Reducing the precision of model weights (e.g., float16 → int8) to reduce memory footprint |
| **Safetensors** | A safe, efficient file format for storing model weights, developed by HuggingFace |
| **Scheduler** | The algorithm that determines how much noise to remove at each denoising timestep |
| **Stable Diffusion** | An open-source family of latent diffusion models developed by Stability AI |
| **Tensor** | A multi-dimensional array of numbers (generalization of vectors and matrices); the fundamental data structure in deep learning |
| **Timestep** | One iteration of the denoising process; with 20 inference steps there are 20 timesteps |
| **U-Net** | The neural network architecture used for denoising in Stable Diffusion; shaped like the letter U with encoder (downsampling) and decoder (upsampling) paths |
| **VAE (Variational Autoencoder)** | Neural network that compresses images into latent space (encoder) and decompresses them back (decoder) |
| **VRAM (Video RAM)** | Dedicated memory on a GPU, separate from the computer's main RAM |
| **xFormers** | A library by Meta Research providing memory-efficient attention kernels; reduces attention memory use and latency simultaneously |
