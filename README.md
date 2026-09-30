# 🌮 TACO: Training-free Sound Prompted Segmentation via Semantically Constrained Audio-visual CO-factorization

Here is the code to reproduce the evaluations of _TACO_ 🌮. Dataset and checkpoint locations are set in [`TACO/configs/paths.yaml`](TACO/configs/paths.yaml).

## Environment

Python 3.8. CUDA 12.1 and GCC 12 need to be available (`module load cuda/12.1 gcc/12.5.0` on this cluster). From the repository root:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST="8.0;9.0" FORCE_CUDA=1 pip install --no-build-isolation -r requirements-detectron2.txt
export PYTHONPATH="$(pwd)/TACO:$(pwd):${PYTHONPATH}"
```

The second install builds Detectron2 and the MultiScaleDeformableAttention operator used by FC-CLIP.

## Evaluation

Each command runs the benchmark three times and writes the mean and standard deviation. Full test sets take hours on one GPU. `--limit N` evaluates the first N samples only.

```bash
python -m taco.eval --benchmark s4 --output s4.json
python -m taco.eval --benchmark ms3 --output ms3.json
python -m taco.eval --benchmark ade_sp --output ade_sp.json
python -m taco.eval --benchmark avss --output avss.json
python -m taco.eval --benchmark ade_sp_semantic --output ade_sp_semantic.json
```

- `s4`: AVSBench single-source
- `ms3`: AVSBench multi-source
- `ade_sp`: ADE Sound Prompted, binary
- `avss`: AVSBench-semantic
- `ade_sp_semantic`: ADE Sound Prompted, semantic