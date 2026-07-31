import os
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from oellm.main import schedule_evals

_config = yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text())
ALL_TASK_GROUPS = list(_config["task_groups"].keys())


@pytest.mark.parametrize("n_shot", [None, 0])
@pytest.mark.parametrize("task_groups", ALL_TASK_GROUPS)
def test_schedule_evals(tmp_path, n_shot, task_groups):
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups=task_groups,
            n_shot=n_shot,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )


def test_schedule_evals_slurm_template_var_overrides(tmp_path):
    """Verify --slurm_template_var JSON overrides appear in the generated sbatch."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {
                "EVAL_OUTPUT_DIR": str(tmp_path),
                "PARTITION": "default_partition",
                "ACCOUNT": "test_account",
            },
        ),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            slurm_template_var='{"PARTITION":"dev-g","ACCOUNT":"myproject","TIME":"02:15:00","GPUS_PER_NODE":2}',
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "#SBATCH --partition=dev-g" in sbatch_content
    assert "#SBATCH --account=myproject" in sbatch_content
    assert "#SBATCH --time=02:15:00" in sbatch_content
    assert "#SBATCH --gres=gpu:2" in sbatch_content


def test_schedule_evals_nodelist(tmp_path):
    """Verify --nodelist adds an #SBATCH --nodelist directive to the sbatch."""
    env = {k: v for k, v in os.environ.items() if k != "NODELIST"}
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {**env, "EVAL_OUTPUT_DIR": str(tmp_path)}, clear=True),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
            nodelist="tdll-3gpu4",
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "#SBATCH --nodelist=tdll-3gpu4" in sbatch_content


def test_schedule_evals_no_nodelist(tmp_path):
    """Without --nodelist the directive is stripped from the sbatch."""
    env = {k: v for k, v in os.environ.items() if k != "NODELIST"}
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {**env, "EVAL_OUTPUT_DIR": str(tmp_path)}, clear=True),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            tasks="hellaswag",
            n_shot=0,
            skip_checks=True,
            venv_path=str(Path(sys.prefix)),
            dry_run=True,
        )

    sbatch_files = list(tmp_path.glob("**/submit_evals.sbatch"))
    assert len(sbatch_files) == 1
    sbatch_content = sbatch_files[0].read_text()
    assert "--nodelist" not in sbatch_content


def test_generated_sbatch_has_judgearena_case(tmp_path):
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        schedule_evals(
            models="EleutherAI/pythia-70m",
            task_groups="judgearena-alpaca",
            n_shot=0,
            skip_checks=True,
            dry_run=True,
        )
    script = next(tmp_path.rglob("submit_evals.sbatch")).read_text()
    assert "judgearena)" in script
    assert "singularity exec" in script
    assert "$EVAL_SIF_PATH" in script
    # config_path defaults to the task name (resolves the bundled config); the
    # scheduled task is always re-applied via --task so it stays per-job even
    # when JUDGEARENA_ARGS overrides --config_path.
    assert 'judgearena --config_path "$task_path"' in script
    assert '--task "$task_path"' in script
    assert '--model.name "VLLM/$model_path"' in script
    # JudgeArena writes the lm-eval envelope (opt-in) into the run folder collect scans
    assert '--run.result_folder "$RESULTS_SUBDIR"' in script
    assert "--run.emit_envelope true" in script
    # free-form JudgeArena flags can be injected via the JUDGEARENA_ARGS env var
    assert "$JUDGEARENA_ARGS" in script


def test_schedule_evals_slurm_template_var_invalid_json(tmp_path):
    """Verify invalid slurm_template_var raises ValueError."""
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(os.environ, {"EVAL_OUTPUT_DIR": str(tmp_path)}),
    ):
        with pytest.raises(ValueError, match="valid JSON object"):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
                slurm_template_var="not valid json",
            )
        with pytest.raises(ValueError, match="must be a JSON object"):
            schedule_evals(
                models="EleutherAI/pythia-70m",
                tasks="hellaswag",
                n_shot=0,
                skip_checks=True,
                venv_path=str(Path(sys.prefix)),
                dry_run=True,
                slurm_template_var='["partition", "dev-g"]',
            )
