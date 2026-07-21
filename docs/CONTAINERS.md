# Container Workflow

## Overview

Apptainer containers are built automatically via GitHub Actions and stored on HuggingFace Hub at [`openeurollm/evaluation_singularity_images`](https://huggingface.co/datasets/openeurollm/evaluation_singularity_images).

## How It Works

1. Definition files live in `containers/<cluster>.def`
2. On push to `main` (when `.def` files change), GitHub Actions provisions [Lambda Labs](https://lambdalabs.com/) GPU instances via [SkyPilot](https://skypilot.readthedocs.io/) and builds all containers in parallel
3. Built `.sif` images are uploaded to HuggingFace Hub
4. Clusters pull the image specified in `oellm/resources/clusters.yaml` via `EVAL_CONTAINER_IMAGE`

Images are compressed with zstd (level 3) via mksquashfs for a good balance of size and build speed.

## Adding a New Cluster

1. Create `containers/<cluster>.def` with the appropriate base image:
   - NVIDIA: `nvcr.io/nvidia/pytorch:25.06-py3` (or newer)
   - AMD/ROCm: `rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1` (or newer)

2. Add the cluster to the matrix in `.github/workflows/build-and-push-apptainer.yml`:
   ```yaml
   matrix:
     include:
       - image: <new-cluster>
         arch: arm64  # omit for default x86_64
   ```

3. Add cluster configuration to `oellm/resources/clusters.yaml`:
   ```yaml
   <cluster>:
     hostname_pattern: "<pattern>"
     EVAL_BASE_DIR: "<path>"
     PARTITION: "<partition>"
     ACCOUNT: "<account>"
     QUEUE_LIMIT: <limit>
     EVAL_CONTAINER_IMAGE: "eval_env-<cluster>.sif"
     SINGULARITY_ARGS: "--nv"  # or "--rocm" for AMD
   ```

4. Push to `main` to trigger the build.

## JudgeArena (generate + judge)

The `judgearena-suite` task group runs [JudgeArena](https://github.com/OpenEuroLLM/JudgeArena) generate+judge benchmarks (`alpaca-eval`, `arena-hard-v2.0`, `mt-bench`). A model under test generates completions with vLLM, a local vLLM judge scores them pairwise against each task's native baseline, and the metric is a win-rate.

The suite runs `judgearena` **inside `EVAL_CONTAINER_IMAGE`**, so that image must have JudgeArena installed alongside vLLM. Point `EVAL_CONTAINER_IMAGE` at a JudgeArena image and set `JUDGEARENA_CONFIG` to a judge config:

```bash
export EVAL_CONTAINER_IMAGE=/path/to/judgearena-<cluster>.sif
export JUDGEARENA_CONFIG=/path/to/judge.yaml
oellm-eval schedule --models VLLM/<model> --task_groups judgearena-suite \
    --slurm_template_var '{"GPUS_PER_NODE":"4"}'
```

`--task`, `--model.name`, and `--run.result_folder` are supplied by oellm-eval; everything else (judge model, GPU-memory split, generation limits) lives in the judge config:

```yaml
judge:
  model: VLLM/google/gemma-4-12b-it
  engine_kwargs:
    tensor_parallel_size: 4
    gpu_memory_utilization: 0.45
model:
  engine_kwargs:
    gpu_memory_utilization: 0.45
generation:
  n_instructions: 20
```

`collect` reads the resulting `BattleReport` JSON and records the win-rate per task.

### Binding the config and data

The container binds `EVAL_BASE_DIR`, `HF_HOME`, and `HF_DATASETS_CACHE`, so the judge config (`JUDGEARENA_CONFIG`) and JudgeArena's data dir (`JUDGEARENA_DATA`, holding the prefetched task datasets) must live under one of those — the simplest is to put them under `EVAL_BASE_DIR`. To keep them elsewhere, set `JUDGEARENA_EXTRA_BINDS` to a comma-separated list of extra host paths to mount, e.g.:

```bash
export JUDGEARENA_EXTRA_BINDS=/scratch/<proj>/users/<you>
export JUDGEARENA_DATA=/scratch/<proj>/users/<you>/openjury-eval-data
export JUDGEARENA_CONFIG=/projappl/<proj>/<you>/judge.yaml
```

### Building a JudgeArena image

Bake JudgeArena on top of the cluster's vLLM base image:

```
Bootstrap: docker
From: vllm/vllm-openai-rocm:latest   # or vllm/vllm-openai:latest for CUDA
%post
    pip install --no-cache-dir "judgearena @ git+https://github.com/OpenEuroLLM/JudgeArena@main"
%runscript
    exec judgearena "$@"
```
