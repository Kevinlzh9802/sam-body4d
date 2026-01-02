<!-- <h1 align="center">🏂 SAM-Body4D</h1> -->

# 🏂 SAM-Body4D

[**Mingqi Gao**](https://mingqigao.com), [**Yunqi Miao**](https://yoqim.github.io/), [**Jungong Han**](https://jungonghan.github.io/)

**SAM-Body4D** is a **training-free** method for **temporally consistent** and **robust** 4D human mesh recovery from videos.
By leveraging **pixel-level human continuity** from promptable video segmentation **together with occlusion recovery**, it reliably preserves identity and full-body geometry in challenging in-the-wild scenes.

[ 📄 [`Paper`](https://arxiv.org/pdf/2512.08406)] [ 🌐 [`Project Page`](https://mingqigao.com/projects/sam-body4d/index.html)] [ 📝 [`BibTeX`](#-citation)]


### ✨ Key Features

- **Temporally consistent human meshes across the entire video**
<div align=center>
<img src="./assets/demo1.gif" width="99%"/>
</div>

- **Robust multi-human recovery under heavy occlusions**
<div align=center>
<img src="./assets/demo2.gif" width="99%"/>
</div>

- **Robust 4D reconstruction under camera motion**
<div align=center>
<img src="./assets/demo3.gif" width="99%"/>
</div>

<!-- Training-Free 4D Human Mesh Recovery from Videos, based on [SAM-3](https://github.com/facebookresearch/sam3), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas), and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). -->

## 🕹️ Gradio Demo

https://github.com/user-attachments/assets/07e49405-e471-40a0-b491-593d97a95465


## 📊 Resource & Profiling Summary

For detailed GPU/CPU resource usage, peak memory statistics, and runtime profiling, please refer to:

👉 **[resources.md](assets/doc/resources.md)**  


## 🖥️ Installation

#### 1. Create and Activate Environment
```
conda create -n body4d python=3.12 -y
conda activate body4d
```
#### 2. Install PyTorch (choose the version that matches your CUDA), Detectron, and SAM3
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install -e models/sam3
```
If you are using a different CUDA version, please select the matching PyTorch build from the official download page:
https://pytorch.org/get-started/previous-versions/

#### 3. Install Dependencies
```
pip install -e .
```

## 🧪 Apptainer / Singularity (cluster)

If you want a self-contained `.sif` for running on a cluster, make sure **SAM3 is installed inside the image**.
Even though this repo imports SAM3 via `models.sam3...`, the SAM3 code itself imports `sam3.*` internally; if `sam3` is not installed, you will see:
`ModuleNotFoundError: No module named 'sam3'`.

#### Build the SIF

```bash
apptainer build body4d.sif apptainer/body4d.def
```

#### Quick import check inside the image (recommended)

```bash
apptainer exec body4d.sif python scripts/doctor_imports.py
```

#### Run on GPU nodes

Bind your checkpoint directory into the container and pass the normal commands:

```bash
apptainer exec --nv \
  --bind /path/to/checkpoints:/checkpoints \
  body4d.sif \
  python infer_video.py --video /path/to/input.mp4 --config configs/body4d.yaml
```

**Debug tips if you still see `sam3` import errors**
- **Use the same python/pip**: inside the image, always prefer `python -m pip ...` (never plain `pip ...`).
- **Verify install location**: `apptainer exec body4d.sif python -c "import sam3; print(sam3.__file__)"`.
- **If you bind-mount the repo** over the image’s `/opt/sam-body4d`, you can accidentally “hide” the code the editable install points to. Either don’t bind-mount the repo, or bind it to a different path.

#### Fast repro for the BF16 sparse CUDA error (SAM-3D-Body / MHR)

If you see:
`RuntimeError: "addmm_sparse_cuda" not implemented for 'BFloat16'`
you can reproduce (and verify the fix) without running the full pipeline:

```bash
apptainer exec --nv --bind /path/to/checkpoints:/checkpoints body4d.sif \
  python scripts/test_mhr_bf16_issue.py --mhr /checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt --device cuda
```

This should show FP32 succeeding and BF16 failing. The typical fix is to force MHR (or all SAM-3D-Body inference) to run in FP32.

#### SAM3-only smoke test

To confirm SAM3 is installed and can build a model from checkpoint (without running SAM-Body4D):

```bash
apptainer exec --nv --bind /path/to/checkpoints:/checkpoints body4d.sif \
  python scripts/test_sam3_only.py --ckpt /checkpoints/sam3/sam3.pt --device cuda
```


## 🚀 Run the Demo

#### 1. Setup checkpoints & config (recommended)

We provide an automated setup script that:
- generates `configs/body4d.yaml` from a release template,
- downloads all required checkpoints (existing files will be skipped).

Some checkpoints (**[SAM 3](https://huggingface.co/facebook/sam3)** and **[SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3)**) require prior access approval on Hugging Face.
Before running the setup script, please make sure you have **accepted access**
on their Hugging Face pages.

If you plan to use these checkpoints, login once:
```bash
huggingface-cli login
```
Then run the setup script:
```bash
python scripts/setup.py --ckpt-root /path/to/checkpoints
```
#### 2. Run
```bash
python app.py
```
#### Manual checkpoint setup (optional)

If you prefer to download checkpoints manually ([SAM 3](https://huggingface.co/facebook/sam3), [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3), [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl-normal), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas?tab=readme-ov-file#download-checkpoints), [Depth-Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)), please place them under the directory with the following structure:
```
${CKPT_ROOT}/
├── sam3/                                
│   └── sam3.pt
├── sam-3d-body-dinov3/
│   ├── model.ckpt
│   └── assets/
│       └── mhr_model.pt
├── moge-2-vitl-normal/
│   └── model.pt
├── diffusion-vas-amodal-segmentation/
│   └── (directory contents)
├── diffusion-vas-content-completion/
│   └── (directory contents)
└── depth_anything_v2_vitl.pth
```
After placing the files correctly, you can run the setup script again.
Existing files will be detected and skipped automatically.

## 🎯 Inference Scripts

We provide multiple ways to run inference:

### 1. Interactive Gradio App (`app.py`)
**Best for**: Manual annotation and exploration
```bash
python app.py
```
- Upload videos and interactively click to annotate people
- Visualize results in real-time

### 2. Command-Line Tool (`infer_video.py`)
**Best for**: Batch processing with known object locations
```bash
# Step 1: Get bounding boxes interactively
python tools/get_bbox_interactive.py --video your_video.mp4 --frame 0

# Step 2: Run inference with the boxes
python infer_video.py \
    --video your_video.mp4 \
    --config configs/body4d.yaml \
    --output results/ \
    --boxes "1,0,150,200,350,600" "2,0,400,150,600,550"
```

**Important**: You must provide initial prompts (bounding boxes or points) for SAM-3 to track objects. Use `--boxes` or `--points` arguments.

### 3. Python Library (`scripts/offline_app.py`)
**Best for**: Integration into other scripts
```python
from scripts.offline_app import offline_app
app = offline_app(refine_occlusion=True)
# See INFERENCE_GUIDE.md for details
```

### 📖 Detailed Documentation

For comprehensive usage instructions, see:
- **[ANSWERS.md](ANSWERS.md)** - Quick answers about initialization and prompts
- **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** - Complete usage guide
- **[example_usage.sh](example_usage.sh)** - Working examples
- **[SUMMARY.md](SUMMARY.md)** - Overview of all scripts

## 📝 Citation
If you find this repository useful, please consider giving a star ⭐ and citation.
```
@article{gao2025sambody4d,
  title   = {SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos},
  author  = {Gao, Mingqi and Miao, Yunqi and Han, Jungong},
  journal = {arXiv preprint arXiv:2512.08406},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.08406}
}
```

## 👏 Acknowledgements

The project is built upon [SAM-3](https://github.com/facebookresearch/sam3), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas) and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). We sincerely thank the original authors for their outstanding work and contributions. 
