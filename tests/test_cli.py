"""End to end on a synthetic benchmark: run with plugins, merge shards, evaluate."""

import json
from pathlib import Path

from interactionbench.cli import main
from interactionbench.mcq import make_mcq_scorer

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _lines(fp):
    return [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]


def test_run_and_eval(bench, tmp_path, capsys):
    out = tmp_path / "run"
    main(["run", "--plugin", str(EXAMPLES / "custom_model.py"), "--model", "scripted:always",
          "--protocol", "sliding", "--blind", "--data", str(bench), "--mcq", "--out", str(out)])
    preds = _lines(out / "preds.jsonl")
    assert len(preds) == 4
    by = {f"{p['video_id']}#{p['item_index']}": p for p in preds}
    assert by["vidAAAAAAAA#1"]["n_polls"] == 20 and len(by["vidAAAAAAAA#1"]["emissions"]) == 20
    assert by["vidAAAAAAAA#0"]["n_polls"] == 6
    assert (out / "raw" / "vidAAAAAAAA#0.json").exists() and (out / "config.json").exists()

    main(["run", "--plugin", str(EXAMPLES / "custom_model.py"), "--model", "scripted:always",
          "--protocol", "sliding", "--blind", "--data", str(bench), "--mcq", "--out", str(out)])
    assert len(_lines(out / "preds.jsonl")) == 4

    ev = tmp_path / "eval"
    main(["eval", str(out / "preds.jsonl"), "--data", str(bench), "--mcq-key", "--out", str(ev)])
    s = json.loads((ev / "summary.json").read_text())
    assert s["overall"]["n_items"] == 4
    assert s["config"]["pre_tol_s"] == 1.0 and s["config"]["use_gate"] is False
    neg = [r for r in _lines(ev / "records.jsonl") if r["family"] == "negative"][0]
    assert neg["false_alarm"] is True and neg["total_score"] == 0.0
    assert "InteractionBench eval" in capsys.readouterr().out


def test_silent_system_gets_no_credit_on_positive_items(bench, tmp_path):
    out = tmp_path / "run"
    main(["run", "--plugin", str(EXAMPLES / "custom_model.py"), "--model", "always-wait",
          "--plugin", str(EXAMPLES / "custom_protocol.py"), "--protocol", "single-look",
          "--blind", "--data", str(bench), "--out", str(out)])
    ev = tmp_path / "eval"
    main(["eval", str(out / "preds.jsonl"), "--data", str(bench), "--mcq-key", "--out", str(ev),
          "--plugin", str(EXAMPLES / "custom_judge.py"), "--judge", "exact"])
    recs = {r["item_id"]: r for r in _lines(ev / "records.jsonl")}
    assert recs["vidBBBBBBBB#1"]["total_score"] == 100.0
    assert recs["vidAAAAAAAA#1"]["total_score"] == 0.0
    assert recs["vidAAAAAAAA#1"]["silence_compliance"] is None


def test_shards_and_merge(bench, tmp_path):
    outs = []
    for i in range(2):
        out = tmp_path / f"s{i}"
        main(["run", "--plugin", str(EXAMPLES / "custom_model.py"), "--model", "always-wait",
              "--blind", "--data", str(bench), "--num-shards", "2", "--shard-index", str(i),
              "--out", str(out)])
        outs.append(str(out / "preds.jsonl"))
    merged = tmp_path / "merged.jsonl"
    main(["merge", str(merged), *outs, outs[0]])
    assert len(_lines(merged)) == 4


def test_items_filter_and_missing_predictions(bench, tmp_path):
    items = tmp_path / "items.txt"
    items.write_text("vidBBBBBBBB#0\n")
    preds = tmp_path / "p.jsonl"
    preds.write_text(json.dumps({"video_id": "vidBBBBBBBB", "item_index": 0, "emissions": [
        {"t": 3.0, "content": "1"}, {"t": 6.2, "content": "2"}, {"t": 9.0, "content": "3"}]}) + "\n")
    ev = tmp_path / "e"
    main(["eval", str(preds), "--data", str(bench), "--items", str(items), "--out", str(ev)])
    r = _lines(ev / "records.jsonl")
    assert len(r) == 1 and r[0]["accuracy"] == 100.0 and r[0]["v_redundant"] == 0
    assert abs(r[0]["timing_accuracy"] - round(100 * (1 + (1 - 0.2 / 5) + 1) / 3, 1)) < 1e-9
    main(["eval", str(preds), "--data", str(bench), "--out", str(ev)])
    assert len(_lines(ev / "records.jsonl")) == 4
    main(["eval", str(preds), "--data", str(bench), "--skip-missing", "--out", str(ev)])
    assert len(_lines(ev / "records.jsonl")) == 1


def test_mcq_scorer():
    s = make_mcq_scorer(["red", "blue", "green"], 1)
    for good in ("B", "b.", "(B)", "Answer: B", "blue", "It is blue."):
        assert s("", "", good) == 1.0, good
    for bad in ("A", "C.", "red", "", "purple"):
        assert s("", "", bad) == 0.0, bad
