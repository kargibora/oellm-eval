from unittest.mock import Mock

from oellm.utils import (
    _expand_local_model_paths,
    _num_jobs_in_queue,
    _pre_download_judge_arena_tasks,
)


class TestExpandLocalModelPaths:
    def test_directory_with_safetensors(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()
        assert _expand_local_model_paths(model_dir) == [model_dir]

    def test_hf_checkpoint_structure(self, tmp_path):
        model_dir = tmp_path / "model"
        iter1 = model_dir / "hf" / "iter_0001000"
        iter2 = model_dir / "hf" / "iter_0002000"
        for d in [iter1, iter2]:
            d.mkdir(parents=True)
            (d / "model.safetensors").touch()

        result = _expand_local_model_paths(model_dir)
        assert set(result) == {iter1, iter2}

    def test_multiple_models_in_subdirs(self, tmp_path):
        base_dir = tmp_path / "models"
        model1 = base_dir / "pythia-70m"
        model2 = base_dir / "pythia-160m"
        for d in [model1, model2]:
            d.mkdir(parents=True)
            (d / "model.safetensors").touch()

        result = _expand_local_model_paths(base_dir)
        assert set(result) == {model1, model2}

    def test_no_safetensors_returns_empty(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()
        assert _expand_local_model_paths(model_dir) == []


class TestNumJobsInQueue:
    def test_counts_jobs(self, monkeypatch):
        class Result:
            returncode = 0
            stdout = "12345\n12346\n12347\n"

        monkeypatch.setattr("oellm.utils.subprocess.run", lambda *a, **kw: Result())
        assert _num_jobs_in_queue() == 3

    def test_returns_zero_on_error(self, monkeypatch):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "error"

        monkeypatch.setattr("oellm.utils.subprocess.run", lambda *a, **kw: Result())
        assert _num_jobs_in_queue() == 0


def test_pre_download_judge_arena_tasks_uses_container_and_shared_paths(
    tmp_path, monkeypatch
):
    image = tmp_path / "judgearena.sif"
    data_root = tmp_path / "judgearena-data"
    hf_home = tmp_path / "hf-cache"
    run = Mock()
    monkeypatch.setattr("oellm.utils.subprocess.run", run)
    for key, value in {
        "EVAL_BASE_DIR": tmp_path,
        "JUDGEARENA_CONTAINER_IMAGE": image,
        "JUDGEARENA_DATA": data_root,
        "HF_HOME": hf_home,
    }.items():
        monkeypatch.setenv(key, str(value))

    _pre_download_judge_arena_tasks(
        ["mt-bench", "alpaca-eval", "mt-bench"],
        venv_path=None,
    )

    command = run.call_args.args[0]
    assert command[:3] == ["singularity", "exec", "--bind"]
    assert str(image) in command
    assert command[-4:] == ["tasks", "download", "alpaca-eval", "mt-bench"]
    assert run.call_args.kwargs["env"]["JUDGEARENA_DATA"] == str(data_root)
