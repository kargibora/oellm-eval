# Container Workflow

## Overview

Apptainer containers are built automatically via GitHub Actions and stored on HuggingFace Hub at [`openeurollm/evaluation_singularity_images`](https://huggingface.co/datasets/openeurollm/evaluation_singularity_images).

## How It Works

1. Definition files live in `containers/<cluster>.def`
2. On push to `main` (when `.def` files change), GitHub Actions provisions [Lambda Labs](https://lambdalabs.com/) GPU instances via [SkyPilot](https://skypilot.readthedocs.io/) and builds all containers in parallel
3. Built `.sif` images are uploaded to HuggingFace Hub
4. Clusters pull the image specified in `oellm/resources/clusters.yaml` via `EVAL_CONTAINER_IMAGE`

Images are compressed with zstd (level 3) via mksquashfs for a good balance of size and build speed.

## JudgeArena image

The JudgeArena task groups (`judgearena-alpaca`, `judgearena-arena-hard`, `judgearena-mt-bench`, `judgearena-elo`) run inside a separate image (vLLM + [JudgeArena](https://github.com/OpenEuroLLM/JudgeArena)), not the per-cluster `eval_env-*` image. Its definition is `containers/judgearena.def`, but it is intentionally **not** in the build matrix above, so CI does not build it — the image is published manually for now at [`kbora/judgearena-container`](https://huggingface.co/datasets/kbora/judgearena-container). To wire it into CI later, add `- image: judgearena` to the matrix.

Point `EVAL_CONTAINER_IMAGE` at this `.sif` when scheduling JudgeArena tasks (see the JudgeArena tasks section in the [README](../README.md#judgearena-tasks)). Run it **without** `--rocm` (`SINGULARITY_ARGS`): the flag injects host ROCm libs built against a newer glibc than the image ships and breaks the `torch` import; `--gpus`/`--device` plus the default `/dev` mount still provide GPU access.

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
