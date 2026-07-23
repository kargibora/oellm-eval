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

## JudgeArena

The `judgearena-suite` and `judgearena-elo` task groups run [JudgeArena](https://github.com/OpenEuroLLM/JudgeArena) **inside `EVAL_CONTAINER_IMAGE`**, so that image must have JudgeArena installed alongside vLLM. Point `EVAL_CONTAINER_IMAGE` at a JudgeArena image and `JUDGEARENA_CONFIG` at a judge config:

```bash
export EVAL_CONTAINER_IMAGE=/path/to/judgearena-<cluster>.sif
export JUDGEARENA_CONFIG=/path/to/judge.yaml
oellm-eval schedule --models VLLM/<model> --task_groups judgearena-suite
```

`oellm-eval` injects `--task`/`--model.name`/`--run.result_folder`; `collect` records the win-rate (`judgearena-suite`) or ELO rating (`judgearena-elo`) per model. If the config or prefetched data live outside the bound dirs (`EVAL_BASE_DIR`, `HF_HOME`, `HF_DATASETS_CACHE`), add them via `JUDGEARENA_EXTRA_BINDS`.

Building the JudgeArena image and writing judge configs are covered in JudgeArena's [`docs/CONTAINER.md`](https://github.com/OpenEuroLLM/JudgeArena/blob/main/docs/CONTAINER.md).
