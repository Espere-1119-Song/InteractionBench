"""Helpers shared by the analysis scripts: common arguments, run lists, evaluation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

DEFAULT_RUNS_ROOT = "results/runs"
DEFAULT_DATA = "data/interactionbench"
DEFAULT_OUT = "results/analysis"
MCQ_KEY = "mcq/mcq_key_v4.jsonl"
EVAL_NOJUDGE = "eval_nojudge"
EVAL_JUDGE = "eval_judge"


def add_common_args(ap: argparse.ArgumentParser, multi_root: bool = False) -> None:
    if multi_root:
        ap.add_argument("--runs-root", nargs="+", default=[DEFAULT_RUNS_ROOT],
                        help="one or more directories that hold <run_name>/preds.jsonl")
    else:
        ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT,
                        help="directory that holds <run_name>/preds.jsonl")
    ap.add_argument("--data", default=DEFAULT_DATA, help="benchmark root directory")
    ap.add_argument("--mcq-key", default=None,
                    help="multiple-choice answer key (default <data>/" + MCQ_KEY + ")")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")


def resolve_common(args) -> None:
    if getattr(args, "mcq_key", None) is None:
        args.mcq_key = f"{args.data}/{MCQ_KEY}"
    os.makedirs(args.out, exist_ok=True)


def parse_runs(values: list[str] | None, default: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not values:
        return list(default)
    if len(values) == 1 and os.path.isfile(values[0]):
        values = [l.strip("\n") for l in Path(values[0]).read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    out = []
    for v in values:
        if "\t" in v:
            label, run = v.split("\t", 1)
        elif "=" in v:
            label, run = v.rsplit("=", 1)
        else:
            label, run = v, v
        out.append((label.strip(), run.strip()))
    return out


def run_eval(predictions: str, data: str, mcq_key: str | None, out_dir: str | None = None,
             quiet: bool = True, judge_obj=None, **kw):
    from interactionbench.evaluate import EvalConfig, evaluate, write_outputs

    cfg = EvalConfig(predictions=str(predictions), data=data, mcq_key=mcq_key, **kw)
    if quiet:
        with contextlib.redirect_stderr(io.StringIO()):
            agg, records = evaluate(cfg, judge=judge_obj)
    else:
        agg, records = evaluate(cfg, judge=judge_obj)
    if out_dir:
        write_outputs(out_dir, agg, records)
    return agg, records


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
