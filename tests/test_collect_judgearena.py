import json

import pandas as pd

from oellm.main import collect_results


def test_collect_results_parses_judgearena_envelope(tmp_path):
    # The judgearena sbatch case converts its report into this lm-eval envelope;
    # collect then reads it through the same generic extractor as every suite.
    envelope = {
        "config_general": {"model_name": "openeurollm/prelude"},
        "results": {"alpaca-eval": {"winrate": 0.9}},
        "n-shot": {"alpaca-eval": 0},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(envelope))

    output_csv = tmp_path / "eval_results.csv"
    collect_results(str(tmp_path), str(output_csv))

    df = pd.read_csv(output_csv)
    row = df[df["metric_name"] == "winrate"]
    assert len(row) == 1
    assert row.iloc[0]["performance"] == 0.9
    assert row.iloc[0]["model_name"] == "openeurollm/prelude"
    assert row.iloc[0]["task"] == "alpaca-eval"


def test_collect_results_skips_non_envelope_json(tmp_path):
    # JudgeArena writes native artifacts next to the envelope; some have a
    # `results` value that isn't a {metric: value} dict. collect must skip
    # those instead of crashing, and still read the real envelope.
    (tmp_path / "envelope.json").write_text(
        json.dumps(
            {
                "config_general": {"model_name": "m"},
                "results": {"alpaca-eval": {"winrate": 0.5}},
                "n-shot": {"alpaca-eval": 0},
            }
        )
    )
    (tmp_path / "native.json").write_text(
        json.dumps({"results": {"alpaca-eval": 5}})  # scalar, not a dict
    )

    output_csv = tmp_path / "eval_results.csv"
    collect_results(str(tmp_path), str(output_csv))  # must not raise

    df = pd.read_csv(output_csv)
    row = df[df["metric_name"] == "winrate"]
    assert len(row) == 1
    assert row.iloc[0]["performance"] == 0.5
