import json

import pandas as pd

from oellm.main import collect_results


def test_collect_results_parses_judgearena_battle_report(tmp_path):
    battle_report = {
        "schema_version": "1",
        "report_type": "BattleReport",
        "task": "alpaca-eval",
        "model_A": "VLLM/openeurollm/prelude",
        "model_B": "gpt4_1106_preview",
        "judge_model": "VLLM/google/gemma-4-31b-it",
        "winrate": 0.9,
        "num_wins": 9,
        "num_losses": 1,
        "num_ties": 0,
        "num_missing": 0,
    }
    (tmp_path / "results-prelude.json").write_text(json.dumps(battle_report))

    output_csv = tmp_path / "eval_results.csv"
    collect_results(str(tmp_path), str(output_csv))

    df = pd.read_csv(output_csv)
    row = df[df["metric_name"] == "winrate"]
    assert len(row) == 1
    assert row.iloc[0]["performance"] == 0.9
    assert row.iloc[0]["model_name"] == "VLLM/openeurollm/prelude"
    assert row.iloc[0]["task"] == "alpaca-eval"
