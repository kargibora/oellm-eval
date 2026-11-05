"""The generated job script must not report a failed evaluation as finished.

Every eval is a subprocess whose exit status the script used to ignore, so a
container that fails to start, a missing venv or a crashed harness all ended
with "Evaluation finished" and a zero exit code. The array element was then
COMPLETED with no result file, and only a later `collect --check` noticed.

These tests render a real script through `schedule_evals` and run it against a
stub harness, so they exercise the template rather than a copy of its logic.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from oellm.main import schedule_evals

# Stub `python` implementations placed on PATH by the fake venv below.
HARNESS_OK = """#!/bin/bash
# lm_eval creates the results directory itself and appends a timestamp to the
# path it is given; mimic both so the script's result check sees a real file.
for arg in "$@"; do
    if [ "$prev" = "--output_path" ]; then
        mkdir -p "$(dirname "$arg")"
        touch "${arg%.json}_2026.json"
    fi
    prev="$arg"
done
exit 0
"""
HARNESS_CRASHES = """#!/bin/bash
echo "boom" >&2
exit 1
"""
HARNESS_SILENT = """#!/bin/bash
exit 0
"""
JUDGEARENA_OK = """#!/bin/bash
for arg in "$@"; do
    if [ "$prev" = "--run.result_folder" ]; then
        output_dir="$arg"
    fi
    prev="$arg"
done
mkdir -p "$output_dir/run"
echo '{"schema_version":"judgearena-run-metadata/v1"}' > "$output_dir/run/run-metadata.v1.json"
exit 0
"""


def _fake_venv(tmp_path: Path, harness: str, command: str = "python") -> Path:
    """A venv whose `activate` puts a stub harness ahead of the real one."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "activate").write_text(f'export PATH="{venv}/bin:$PATH"\n')
    executable = venv / "bin" / command
    executable.write_text(harness)
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    return venv


def _render(
    tmp_path: Path,
    venv: Path | None,
    *,
    task: str = "hellaswag",
    backend: str | None = None,
) -> Path:
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path), "GPUS_PER_NODE": "1"}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks=task,
            n_shot=0,
            skip_checks=True,
            venv_path=str(venv) if venv else None,
            dry_run=True,
            **({"lm_eval_backend": backend} if backend is not None else {}),
        )
    scripts = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(scripts) == 1
    return scripts[0]


def _run(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "1",
            "SLURM_JOB_ID": "1",
        },
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "harness, expected_error",
    [
        pytest.param(HARNESS_CRASHES, "exited with status 1", id="harness-crashes"),
        pytest.param(HARNESS_SILENT, "wrote no result", id="harness-writes-nothing"),
    ],
)
def test_failed_evaluation_fails_the_job(tmp_path, harness, expected_error):
    """A crash and a silent no-op are both failures, not finished evaluations."""
    script = _render(tmp_path, _fake_venv(tmp_path, harness))
    result = _run(script)

    assert result.returncode == 1
    assert expected_error in result.stdout
    assert "1 failed evaluation" in result.stdout
    assert "Evaluation finished" not in result.stdout


def test_successful_evaluation_passes(tmp_path):
    """The happy path still exits 0 and reports the evaluation as finished."""
    script = _render(tmp_path, _fake_venv(tmp_path, HARNESS_OK))
    result = _run(script)

    assert result.returncode == 0
    assert "Evaluation finished" in result.stdout
    assert "[error]" not in result.stdout


@pytest.mark.parametrize(
    "backend, expected_python, expected_model",
    [
        (None, "python", "hf"),
        ("hf", "python", "hf"),
        ("vllm", "/opt/vllm/bin/python", "vllm"),
    ],
)
def test_container_uses_the_selected_lm_eval_runtime(
    tmp_path, monkeypatch, backend, expected_python, expected_model
):
    """Changing the backend must also select the Python containing that backend."""
    args_file = tmp_path / "args"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    singularity = bin_dir / "singularity"
    singularity.write_text(
        HARNESS_OK.replace(
            'for arg in "$@";', 'printf "%s\\n" "$@" > "$RUNTIME_ARGS"\nfor arg in "$@";'
        )
    )
    singularity.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("RUNTIME_ARGS", str(args_file))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("EVAL_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EVAL_CONTAINER_IMAGE", "eval.sif")
    monkeypatch.setenv("SINGULARITY_ARGS", "--nv --contain")

    result = _run(_render(tmp_path, None, backend=backend))

    assert result.returncode == 0, result.stdout + result.stderr
    args = args_file.read_text().splitlines()
    assert args[args.index("env") + 1].startswith("CUDA_VISIBLE_DEVICES=")
    assert args[args.index("env") + 2 : args.index("-m")] == [expected_python]
    assert args[args.index("--model") + 1] == expected_model


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_explicit_venv_keeps_its_own_lm_eval_runtime(tmp_path, monkeypatch, backend):
    args_file = tmp_path / "args"
    monkeypatch.setenv("RUNTIME_ARGS", str(args_file))
    harness = HARNESS_OK.replace(
        'for arg in "$@";',
        'printf "%s\\n" "$@" > "$RUNTIME_ARGS"\nfor arg in "$@";',
    )
    venv = _fake_venv(tmp_path, harness)

    result = _run(_render(tmp_path, venv, backend=backend))

    assert result.returncode == 0, result.stdout + result.stderr
    args = args_file.read_text().splitlines()
    assert args[:2] == ["-m", "lm_eval"]
    assert args[args.index("--model") + 1] == backend


def test_missing_shared_runtime_fails_instead_of_using_hf(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    singularity = bin_dir / "singularity"
    singularity.write_text(
        HARNESS_OK.replace(
            'for arg in "$@";',
            'for arg in "$@"; do\n'
            '    if [ "$arg" = "/opt/vllm/bin/python" ]; then exit 127; fi\n'
            'done\nfor arg in "$@";',
        )
    )
    singularity.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("EVAL_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EVAL_CONTAINER_IMAGE", "old.sif")

    result = _run(_render(tmp_path, None, backend="vllm"))

    assert result.returncode == 1
    assert "exited with status 127" in result.stdout
    assert "Evaluation finished" not in result.stdout


def test_failure_count_survives_the_read_loop(tmp_path):
    """Both rows of a two-eval job are counted.

    The loop reads the CSV by redirection rather than a pipe; a pipeline would
    run the body in a subshell and lose the counter at `done`.
    """
    venv = _fake_venv(tmp_path, HARNESS_CRASHES)
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path), "GPUS_PER_NODE": "1"}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag,arc_easy",
            n_shot=0,
            skip_checks=True,
            venv_path=str(venv),
            max_array_len=1,
            dry_run=True,
        )
    script = next(iter(tmp_path.glob("**/submit_evals.sbatch")))
    result = _run(script)

    assert result.returncode == 1
    assert "2 failed evaluation" in result.stdout


@pytest.mark.parametrize(
    "harness, expected_error",
    [
        pytest.param(HARNESS_CRASHES, "exited with status 1", id="crashes"),
        pytest.param(HARNESS_SILENT, "wrote no result", id="writes-nothing"),
    ],
)
def test_failed_judgearena_evaluation_fails_the_job(tmp_path, harness, expected_error):
    venv = _fake_venv(tmp_path, harness, command="judgearena")
    script = _render(tmp_path, venv, task="arena-hard-v2.0")
    result = _run(script)

    assert result.returncode == 1
    assert expected_error in result.stdout
    assert "Evaluation finished" not in result.stdout


def test_successful_judgearena_evaluation_passes(tmp_path):
    venv = _fake_venv(tmp_path, JUDGEARENA_OK, command="judgearena")
    script = _render(tmp_path, venv, task="arena-hard-v2.0")
    result = _run(script)

    assert result.returncode == 0
    assert "Evaluation finished" in result.stdout


def test_sbatch_failure_exits_non_zero(tmp_path):
    """A rejected submission must not look like a successful schedule."""
    error = subprocess.CalledProcessError(1, ["sbatch"], stderr="AssocMaxSubmitJobLimit")
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch("oellm.main.subprocess.run", side_effect=error),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
        pytest.raises(SystemExit) as excinfo,
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
        )
    assert excinfo.value.code == 1
