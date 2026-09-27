"""Helpers shared by the analysis scripts: common arguments, run lists, evaluation.

Layout assumed by every script:
  <runs_root>/<run_name>/preds.jsonl                          predictions of one run
  <runs_root>/<run_name>/<eval_name>/{summary.json,records.jsonl}   one evaluation of it

Evaluation directories are written by ``python -m interactionbench eval ... --out DIR``
or by ``python scripts/reproduce_paper.py eval --write-eval NAME``.
"""

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
EVAL_NOJUDGE = "eval_nojudge"   # evaluation without a judge (free-form content scored lexically)
EVAL_JUDGE = "eval_judge"       # evaluation with the paper judge


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
    """Fill the defaults that depend on other arguments and create the output directory."""
    if getattr(args, "mcq_key", None) is None:
        args.mcq_key = f"{args.data}/{MCQ_KEY}"
    os.makedirs(args.out, exist_ok=True)


def parse_runs(values: list[str] | None, default: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Run list as [(label, run_name)].

    ``values`` is either a list of entries or one path of a file with one entry per
    line. An entry is ``run_name``, ``label<TAB>run_name`` or ``label=run_name`` (split
    at the last ``=``); without a label the run name is the label."""
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
    """Score one predictions file with the library evaluator.

    ``kw`` holds EvalConfig fields (pre_tol, delta, gate_thresh, items, judge, ...).
    ``judge_obj`` is an already built judge, used in place of the ``judge`` spec.
    With ``out_dir`` the records and the summary are also written there.
    Returns (summary, records)."""
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
