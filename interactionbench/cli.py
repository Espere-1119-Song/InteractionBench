"""Command line interface.

  ibench run    generate predictions for one system under one protocol
  ibench eval   score a predictions file
  ibench merge  merge prediction files from parallel shards
  ibench list   show the registered models, protocols and judges

Run ``ibench <command> --help`` for the options of each command.
"""

from __future__ import annotations

import argparse
import json
import sys

from .registry import load_plugins

DEFAULT_DATA = "data/interactionbench"
MCQ_OPTIONS = "mcq/mcq_options_v4.jsonl"
MCQ_KEY = "mcq/mcq_key_v4.jsonl"


def parse_kv(pairs: list[str] | None) -> dict:
    """``key=value`` pairs; values are parsed as JSON when possible, else kept as text."""
    out = {}
    for pair in pairs or []:
        k, sep, v = pair.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {pair!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=DEFAULT_DATA, help="benchmark root directory")
    p.add_argument("--plugin", action="append", default=[], metavar="FILE_OR_MODULE",
                   help="Python file or module that registers models, protocols or "
                        "judges; repeatable")


def _resolve_auto(value: str | None, data: str, rel: str) -> str | None:
    return f"{data}/{rel}" if value == "auto" else value


# ------------------------------------------------------------------ run

def add_run_parser(sub) -> None:
    p = sub.add_parser("run", help="generate predictions",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(p)
    g = p.add_argument_group("system under test")
    g.add_argument("--model", default="qwen3vl-8b",
                   help="zoo name, hf:<repo>, hf-text:<repo>, api:<model>, "
                        "<adapter>:<id>, or module:Class")
    g.add_argument("--model-path", default=None,
                   help="local checkpoint directory used in place of the repo id")
    g.add_argument("--model-config", default=None,
                   help="JSON or YAML file with extra named model configurations")
    g.add_argument("--model-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="adapter keyword argument; repeatable")
    g.add_argument("--api-base", default=None, help="endpoint root for API models")
    g.add_argument("--api-key-env", default=None,
                   help="environment variable that holds the API key")

    g = p.add_argument_group("test method")
    g.add_argument("--protocol", default="sliding",
                   help="sliding, cumulative, interleaved, offline, a registered name, "
                        "or module:Class")
    g.add_argument("--interval", type=float, default=1.0,
                   help="stream seconds between decision steps")
    g.add_argument("--a-window", type=float, default=10.0,
                   help="seconds of polling after an A-type question is revealed")
    g.add_argument("--sample-fps", type=float, default=2.0, help="frame sampling rate")
    g.add_argument("--max-frames", type=int, default=16, help="frames visible per step")
    g.add_argument("--max-new-per-turn", type=int, default=8,
                   help="interleaved: new frames appended per step")
    g.add_argument("--max-long-side", type=int, default=512, help="frame resize bound")
    g.add_argument("--max-new-tokens", type=int, default=96)
    g.add_argument("--hint-set", default="default",
                   help="capability-hint paraphrase set: default, v2, v3")
    g.add_argument("--blind", action="store_true",
                   help="remove all frames: language-prior baseline")
    g.add_argument("--protocol-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="extra option passed to a custom protocol; repeatable")

    g = p.add_argument_group("items")
    g.add_argument("--mcq", default=None, nargs="?", const="auto",
                   help="options file; without a value uses <data>/" + MCQ_OPTIONS)
    g.add_argument("--video-dir", default=None,
                   help="video root, default <data>/videos (use an H.264 proxy directory "
                        "when the originals are AV1)")
    g.add_argument("--items", default=None, help="file with one item_id per line")
    g.add_argument("--videos", nargs="*", default=None, help="only these video ids")
    g.add_argument("--capabilities", nargs="*", default=None)
    g.add_argument("--limit", type=int, default=0, help="max items, 0 = all")
    g.add_argument("--num-shards", type=int, default=1)
    g.add_argument("--shard-index", type=int, default=0)

    g = p.add_argument_group("output")
    g.add_argument("--out", default=None, help="default results/runs/<model>_<protocol>...")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--no-skip-oom", action="store_true",
                   help="stop on GPU out-of-memory instead of recording the item as silent")
    g.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_run)


def cmd_run(args) -> None:
    from .models import load_model_configs
    from .protocols import ProtocolConfig
    from .run import RunConfig, run_benchmark

    load_plugins(args.plugin)
    if args.model_config:
        load_model_configs(args.model_config)
    model_args = parse_kv(args.model_arg)
    if args.api_base:
        model_args["base_url"] = args.api_base
    if args.api_key_env:
        model_args["api_key_env"] = args.api_key_env

    pcfg = ProtocolConfig(
        interval=args.interval, a_window=args.a_window, max_frames=args.max_frames,
        max_new_per_turn=args.max_new_per_turn, max_new_tokens=args.max_new_tokens,
        hint_set=args.hint_set, blind=args.blind, verbose=args.verbose,
        extra=parse_kv(args.protocol_arg))
    cfg = RunConfig(
        data=args.data, model=args.model, model_path=args.model_path,
        model_args=model_args, protocol=args.protocol, protocol_config=pcfg,
        sample_fps=args.sample_fps, max_long_side=args.max_long_side,
        mcq=_resolve_auto(args.mcq, args.data, MCQ_OPTIONS), video_dir=args.video_dir,
        items=args.items, videos=args.videos, capabilities=args.capabilities,
        limit=args.limit, num_shards=args.num_shards, shard_index=args.shard_index,
        out=args.out, overwrite=args.overwrite, skip_oom=not args.no_skip_oom)
    run_benchmark(cfg)


# ------------------------------------------------------------------ eval

def add_eval_parser(sub) -> None:
    p = sub.add_parser("eval", help="score predictions",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("predictions", help="predictions .jsonl")
    _add_common(p)
    p.add_argument("--out", metavar="DIR", default=None,
                   help="write records.jsonl and summary.json to DIR")
    p.add_argument("--mcq-key", default=None, nargs="?", const="auto",
                   help="answer key; without a value uses <data>/" + MCQ_KEY)
    p.add_argument("--items", default=None,
                   help="file of item_ids to score, e.g. benchmark/splits/subset103.txt")
    p.add_argument("--skip-missing", action="store_true",
                   help="ignore items without a prediction (default: score as silent)")
    g = p.add_argument_group("judge")
    g.add_argument("--judge", default=None,
                   help="hf:<model_id>, api:<model>, cache:<files>, ensemble:<slugs>, "
                        "vdc:<judge>, a registered name, or module:Class. "
                        "Without a judge, free-form content is scored lexically.")
    g.add_argument("--judge-cache", default="results/judge_cache/judge_cache.jsonl",
                   help="verdict cache; keep one file per judge model and prompt version")
    g.add_argument("--judge-prompt", default="v2", help="grading prompt version: v1, v2")
    g.add_argument("--judge-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="judge keyword argument, e.g. base_url=...; repeatable")
    g = p.add_argument_group("metric parameters (defaults are the paper protocol)")
    g.add_argument("--delta", type=float, default=None,
                   help="acceptable-delay bound of the timing decay, seconds (5.0)")
    g.add_argument("--pre-tol", type=float, default=None,
                   help="pre-anchor tolerance, seconds (1.0); 0.0 disables it")
    g.add_argument("--use-gate", action="store_true",
                   help="ablation: enable the content gate on trigger matching")
    g.add_argument("--gate-thresh", type=float, default=None,
                   help="content-gate threshold (0.3), only with --use-gate")
    p.set_defaults(func=cmd_eval)


def cmd_eval(args) -> None:
    from .evaluate import EvalConfig, evaluate, print_report, write_outputs

    load_plugins(args.plugin)
    cfg = EvalConfig(
        predictions=args.predictions, data=args.data,
        mcq_key=_resolve_auto(args.mcq_key, args.data, MCQ_KEY), items=args.items,
        skip_missing=args.skip_missing, judge=args.judge, judge_cache=args.judge_cache,
        judge_prompt=args.judge_prompt, judge_args=parse_kv(args.judge_arg),
        delta=args.delta, pre_tol=args.pre_tol, gate_thresh=args.gate_thresh,
        use_gate=True if args.use_gate else None)
    agg, records = evaluate(cfg)
    print_report(agg, len(records), args.predictions)
    if args.out:
        write_outputs(args.out, agg, records)
        print(f"\nwrote {args.out}/records.jsonl and {args.out}/summary.json")


# ------------------------------------------------------------------ merge / list

def add_merge_parser(sub) -> None:
    p = sub.add_parser("merge", help="merge prediction files (first line per item wins)")
    p.add_argument("output")
    p.add_argument("inputs", nargs="+")
    p.set_defaults(func=cmd_merge)


def cmd_merge(args) -> None:
    from .run import merge_predictions
    n = merge_predictions(args.inputs, args.output)
    print(f"{n} items -> {args.output}")


def add_list_parser(sub) -> None:
    p = sub.add_parser("list", help="show registered models, protocols and judges")
    p.add_argument("what", nargs="?", default="all",
                   choices=["all", "models", "adapters", "protocols", "judges"])
    p.add_argument("--plugin", action="append", default=[])
    p.add_argument("--model-config", default=None)
    p.set_defaults(func=cmd_list)


def cmd_list(args) -> None:
    from .judges import list_judges
    from .models import ADAPTERS, list_models, load_model_configs
    from .protocols import list_protocols

    load_plugins(args.plugin)
    if args.model_config:
        load_model_configs(args.model_config)
    if args.what in ("all", "models"):
        print("models:")
        for name, cfg in list_models().items():
            target = cfg.get("repo") or cfg.get("model") or "(set at run time)"
            print(f"  {name:<20} {cfg['adapter']:<9} {target}")
    if args.what in ("all", "adapters"):
        print("model adapters:  " + ", ".join(ADAPTERS.names()))
    if args.what in ("all", "protocols"):
        print("protocols:       " + ", ".join(list_protocols()))
    if args.what in ("all", "judges"):
        print("judge kinds:     " + ", ".join(list_judges()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ibench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    add_run_parser(sub)
    add_eval_parser(sub)
    add_merge_parser(sub)
    add_list_parser(sub)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
