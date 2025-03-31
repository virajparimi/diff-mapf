# Diffusion-Guided Multi-Arm Motion Planning

[Viraj Parimi](https://people.csail.mit.edu/vparimi/), [Brian Williams](https://www.csail.mit.edu/person/brian-williams)  
Massachusetts Institute of Technology  
**[CoRL 2025](https://www.corl.org/)**

**Project:** [diff-mapf-mers.csail.mit.edu](https://diff-mapf-mers.csail.mit.edu/) • **Paper:** [arXiv:2509.08160](https://arxiv.org/abs/2509.08160)

![Overview](docs/static/images/overview.png)

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Pre-Trained Models & Data](#pre-trained-models-and-data)
- [Evaluate](#evaluate)
- [Summarize Results](#summarize-results)
- [Train From Scratch](#train-from-scratch)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [Credits](#credits)
- [Troubleshooting](#troubleshooting)

## Overview
This repository implements a diffusion-guided motion planner for **coordinated multi-arm manipulation**.  
We use single-arm diffusion models for proposal generation and a dual-arm diffusion model (and Diffusion-QL variants) to resolve inter-arm conflicts, enabling scalable planning across team sizes and task difficulty.

## Requirements
- **Python:** 3.9+ (managed via Conda)  
- **Dependencies:** Specified in `environment.yml`
- **GPU:** Recommended for evaluation and training  
- **OS:** Linux (tested on recent Ubuntu 22.04.5 LTS and 24.04.3 LTS)  

## Setup
```bash
conda env create -f environment.yml
conda activate multiarm
# Set your repo root so imports work everywhere:
export PYTHONPATH=<path-to-diff-mapf>:$PYTHONPATH
```

## Pre-Trained Models and Data

### Option 1: Fetch Everything Automatically
Use the helper script (located at repo root) to download and unpack all required artifacts:

```bash
./fetch_assets.sh all --outdir .
```

- Replace `.` with another directory if you want the assets outside the repo.
- Add `--list` to preview the download plan without saving anything.
- Swap `all` for `models`, `datasets`, or `benchmarks` to grab a single category.

### Option 2: Download Assets Manually
- **Pre-trained planners**  
  - [Diffusion models](https://www.dropbox.com/scl/fo/daah2lixb3digjcp5i7ti/AIJ3po1nKmUSPKTBJvJn5qs?rlkey=wsq4b705kx8qi0sxq5j9ipzmz&st=pay2cq62&dl=0) → extract into `application/runs/plain_diffusion/`  
  - [Diffusion-QL models](https://www.dropbox.com/scl/fo/tmrnoz8sticj660oj5v2i/APZdOkl6FX-gXjLSeSxVG9s?rlkey=488l482qmr5kq4y5776j1k2qz&st=xkvfv1nq&dl=0) → extract into `application/runs/diffusion_ql/`
- **Expert datasets** (place the extracted folders under `datasets/`)  
  - [Single-Agent](https://www.dropbox.com/scl/fi/e2mnzqsrh9wf96bhbthb7/single_agent.zip?rlkey=a8uf0gukb04te46zu4164o196&st=hpa9rwh0&dl=0)  
  - [Dual-Agent](https://www.dropbox.com/scl/fi/w9o3c05ndyeavbiu3r1vu/dual_agent.zip?rlkey=u0bqnedvzlvwta8xfpzih382k&st=lzg49q8u&dl=0)  
  - [Single-QL-Agent](https://www.dropbox.com/scl/fi/i71hnenw21oxc9uqpbcy7/ql_single_agent.zip?rlkey=uxnewd77fl9tsfpj1o9sbh1ed&st=wonrl3nj&dl=0)  
  - [Dual-QL-Agent](https://www.dropbox.com/scl/fi/l3601jzsouu1pl3c6lv5w/ql_dual_agent.zip?rlkey=tgra17py4fh2qthj5m7tpcedh&st=qqq1jwex&dl=0)
- **Benchmark tasks (Ha et al.)**
  ```bash
  wget -qO- https://multiarm.cs.columbia.edu/downloads/data/benchmark.tar.xz | tar xvfJ -
  mv benchmark application/tasks/
  ```

## Evaluate
Evaluate the motion planner on **multi-arm pick-and-place**:
```bash
python application/demo.py --single_agent_model "<path-to-mini-custom-diffusion-1.pth>" --dual_agent_model "<path-to-mini-custom-diffusion-2.pth>"
```

**Typical outputs** (logs, metrics, videos) are stored under `application/runs/[plain_diffusion/diffusion_ql]/...`.

## Summarize Results
Aggregate and print benchmark statistics from a run directory:
```bash
python application/evaluate_results.py --result_dir <result-dir>
```

## Train From Scratch

### Download Expert Datasets
Use the dataset links provided in [Pre-Trained Models and Data](#pre-trained-models-and-data). Extract each archive into `datasets/` so the training scripts can find the zarr files.

### Training command
```bash
mkdir -p runs
python -u core/agent_manager.py          \
    --config configs/diffusion.json      \     # or configs/diffusionQL.json
    --offline_dataset <path-to-dataset>  \
    --num_agents <number-of-agents>      \
    --name <training-run-name>
```

To **resume** from a checkpoint, pass:
```bash
--load <path-to-checkpoint>
```
See additional flags in `core/utils.py`.

## Repository Layout
```text
application/
  demo.py              # main evaluation entrypoint
  evaluate_results.py  # result aggregation / reporting
configs/
  diffusion.json       # default training config
core/
  agent_manager.py     # training driver
  utils.py             # common CLI/config helpers
```

## Citation
If you use our work or codebase in your research, please cite our paper.
```bibtex
@InProceedings{pmlr-v305-parimi25a,
  title = {Diffusion-Guided Multi-Arm Motion Planning},
  author = {Parimi, Viraj and Williams, Brian C.},
  booktitle = {Proceedings of The 9th Conference on Robot Learning},
  pages = {4684--4696},
  year = {2025},
  editor = {Lim, Joseph and Song, Shuran and Park, Hae-Won},
  volume = {305},
  series = {Proceedings of Machine Learning Research},
  month = {27--30 Sep},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v305/main/assets/parimi25a/parimi25a.pdf},
  url = {https://proceedings.mlr.press/v305/parimi25a.html},
}
```

## Credits
Portions of code and datasets are adapted from:
- **Decentralized MultiArm**: <https://github.com/real-stanford/decentralized-multiarm>  
- **PyBullet–Blender Recorder** (visualization): <https://github.com/huy-ha/pybullet-blender-recorder>

## License
This project is licensed under the [Apache License 2.0](LICENSE); portions of the code are adapted from credited works released under the same license.

## Troubleshooting
- **`ModuleNotFoundError` after install:** Double-check `PYTHONPATH` is set to your repo root.  
- **Slow or OOM on training:** Use a smaller batch size in `configs/diffusion.json`, or ensure CUDA is available.  
- **No runs appearing:** Confirm write permissions in `runs/` and `application/runs/`.  
- **Benchmark missing:** Re-run the `wget | tar` commands and verify `application/tasks/benchmark/` exists.
