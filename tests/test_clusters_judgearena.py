import os
from unittest.mock import patch

from oellm.utils import _load_cluster_env


def test_lumi_sets_judgearena_container_vars():
    with (
        patch("oellm.utils.socket.gethostname", return_value="uan01"),
        patch.dict(os.environ, {}, clear=False),  # restore env; _load_cluster_env leaks keys
    ):
        for k in (
            "JUDGEARENA_CONTAINER_IMAGE",
            "JUDGEARENA_VENV",
            "JUDGEARENA_SINGULARITY_ARGS",
        ):
            os.environ.pop(k, None)
        _load_cluster_env()
        assert os.environ["JUDGEARENA_CONTAINER_IMAGE"].endswith("vllm-openai-rocm.sif")
        assert os.environ["JUDGEARENA_VENV"].endswith("judgearena-container")
        assert "--rocm" not in os.environ["JUDGEARENA_SINGULARITY_ARGS"]
