#!/usr/bin/env python3
"""Oracle diagnostics: restraint, timing and perception.

Every oracle uses the reference annotations of the benchmark. The results are upper
bounds for diagnosis, not scores of a system that could be deployed.

  restraint    Post-processing of stored predictions, no model. Emissions that count
               as timing violations are deleted; no reply is rewritten. On an item
               that requires silence every emission is deleted. On any other item
               only the first matched response of each reference window is kept
               (reference times: the question time for time type A, the merged
               answer times otherwise; matching as in the scorer, pre-anchor
               tolerance 1.0 s). Text and timestamps of the kept emissions are
               unchanged. The scorer pools all text inside a free-form segment, so
               the deletion can lower content accuracy of free-form items.
  timing       The model is queried only at the reference response times (question
               time for time type A, the distinct answer times otherwise, never on
               an item that requires silence) and is told to answer at each of them.
               No reference answer text is sent to the model.
  perception   At every polling step the images are removed and replaced by the
               reference facts of the item that are already available: a fact is
               released at its annotated evidence time, or at its answer time when
               no evidence time exists. The model still decides when to speak. This
               is a content oracle derived from the annotations, not a transcript of
               the whole video.

The timing and the perception oracle need a model on a GPU. They are implemented as
protocols and registered when this file is loaded as a plugin:

  python -m interactionbench run --plugin analysis/oracles.py \
      --protocol oracle-timing --model qwen3vl-8b --interval 1 --max-frames 16 \
      --sample-fps 2 --mcq --items benchmark/splits/subset103.txt \
      --out results/runs/qwen3vl-8b_oracle_timing
  python -m interactionbench run --plugin analysis/oracles.py \
      --protocol oracle-perception --model qwen3vl-8b --interval 1 --max-frames 16 \
      --sample-fps 2 --mcq --items benchmark/splits/subset103.txt \
      --out results/runs/qwen3vl-8b_oracle_perception

Both use the sliding context regime; ``--protocol-arg mode=cumulative`` or
``mode=interleaved`` selects another regime.

Sub-commands (no model):

  restraint   write <target_root>/<run><target_suffix>/{preds.jsonl,config.json}
  manifest    write oracle_manifest.json into the run directory of a timing or
              perception run
  score       score a finished run on an item subset with the paper protocol
              (pre-anchor tolerance 1.0 s, Delta 5.0 s, stored judge verdicts); a run
              that does not cover every item of the subset is refused
  collect     status of a list of runs: recorded items, failed items, summary

Upstream checkpoint of the GPU part: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
(model name ``qwen3vl-8b``). Judge of ``score``: stored verdicts of
https://huggingface.co/Qwen/Qwen3-14B, or the model itself with ``--gpu-judge``.
Environment: Python >= 3.10 and the ``interactionbench`` package; torch and
transformers for the GPU part.

Commands used for the paper numbers:

  python analysis/oracles.py restraint --preset subset103
  python analysis/oracles.py restraint --preset all --target-suffix _oracle_restraint_all
  python analysis/oracles.py manifest --oracle timing --run qwen3vl-8b_oracle_timing
  python analysis/oracles.py score qwen3vl-8b_oracle_timing --gpu-judge
  python analysis/oracles.py collect

Outputs:
  <target_root>/<run><target_suffix>/preds.jsonl, config.json      (restraint)
  <runs_root>/<run>/oracle_manifest.json                           (manifest)
  <runs_root>/<run>/<eval_name>/{summary.json,records.jsonl}       (score)
  <out>/status.json                                                (collect)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from interactionbench.frames import frames_up_to, subsample
from interactionbench.parsing import parse_decision
from interactionbench.prompts import (FORMAT, INTERLEAVED_SUFFIX, INTERLEAVED_TURN,
                                      REVEAL_TEMPLATE, STANDING_TEMPLATE, SYSTEM, hint_for)
from interactionbench.protocols import (Protocol, make_poll, poll_ticks, register_protocol,
                                        trim_context_images, user_turn)

RESTRAINT_PRESETS = {
    # the two traces compared with the agents, on the 103-item subset
    "subset103": {"runs": ["qwen3vl-8b_sliding_iv1_mcqv4_full", "joyai_streaming_4fps_mcqv4"],
                  "items": "benchmark/splits/subset103.txt"},
    # the four traces of the thinning curves, on all items
    "all": {"runs": ["qwen3vl-8b_sliding_win64_iv1_mcqv4_full", "llava-ov2-8b_sliding_iv1_mcqv4_full",
                     "joyai_streaming_4fps_mcqv4", "mmduet2-3b_streaming_4fps_mcq_MERGED"],
            "items": None},
}
COLLECT_RUNS = [
    ("grid/qwen/polling", "qwen3vl-8b_sliding_iv1_mcqv4_full"),
    ("grid/claude/polling", "claude-bare_sliding_sub103"),
    ("grid/qwen/joyai", "joyai-qwen3vl8b-sub103_streaming_4fps_mcqv4"),
    ("grid/claude/joyai", "joyai-claude-opus5-sub103_streaming_4fps_mcqv4"),
    ("threshold/0.5", "videollm_threshold0.5"),
    ("threshold/0.725", "videollm-online-8b_streaming_8fps_mcq"),
    ("threshold/0.9", "videollm_threshold0.9"),
    ("oracle/reference_facts", "qwen3vl-8b_oracle_perception"),
    ("oracle/timing", "qwen3vl-8b_oracle_timing"),
    ("oracle/baseline/joyai", "joyai_streaming_4fps_mcqv4"),
    ("oracle/restraint/qwen", "qwen3vl-8b_sliding_iv1_mcqv4_full_oracle_restraint"),
    ("oracle/restraint/joyai", "joyai_streaming_4fps_mcqv4_oracle_restraint"),
]
ORACLE_DOC = """Privileged-input diagnostics on an unchanged Qwen3-VL polling backbone.

perception: replace images with the item's reference facts, released only at
annotation evidence/answer times. This is an annotation-derived content oracle,
not an independently annotated full-video transcript or a pure vision ablation.
timing: query the visual model only at reference answer times and force a reply.
No reference answer text is sent to the timing model.
"""


# ------------------------------------------------------------------ GPU oracles (protocols)

class _OracleProtocol(Protocol):
    """Fixed-interval polling loop with two hooks: the tick schedule and an edit of
    the messages of each step before generation."""

    def ticks(self, item, cfg, is_reveal):
        return poll_ticks(item, cfg.interval, cfg.a_window)

    def edit(self, messages, item, t):
        raise NotImplementedError

    def run_item(self, model, frames, item, question, cfg):
        mode = cfg.extra.get("mode", "sliding")
        is_reveal = item.time_type == "A"
        q_t = item.question_time_s if is_reveal else 0.0
        ticks = self.ticks(item, cfg, is_reveal)
        template = REVEAL_TEMPLATE if is_reveal else STANDING_TEMPLATE
        hint = hint_for(item.capability, cfg.hint_set)

        convo: list[dict] = []
        if mode == "interleaved":
            convo = [{"role": "system", "content":
                      SYSTEM + "\n\n" + template.format(t=q_t, question=question,
                                                        hint=hint, fmt=FORMAT)
                      + INTERLEAVED_SUFFIX}]

        polls = []
        prev_t = 0.0 if not is_reveal else max(0.0, q_t - 1e-6)
        for t in ticks:
            history = frames_up_to(frames, t) or frames[:1]
            if cfg.blind:
                history = []
            if mode == "cumulative":
                window = subsample(history, cfg.max_frames)
                messages = [{"role": "system", "content": SYSTEM},
                            user_turn(window, template.format(
                                t=t, question=question, hint=hint, fmt=FORMAT))]
            elif mode == "sliding":
                window = history[-cfg.max_frames:]
                messages = [{"role": "system", "content": SYSTEM},
                            user_turn(window, template.format(
                                t=t, question=question, hint=hint, fmt=FORMAT))]
            else:  # interleaved
                new = [f for f in history if prev_t < f.time <= t] or history[-1:]
                new = subsample(new, cfg.max_new_per_turn)
                convo.append(user_turn(new, INTERLEAVED_TURN.format(t=t)))
                trim_context_images(convo, cfg.max_frames)
                messages = convo

            self.edit(messages, item, t)
            gen = model.timed_chat(messages, max_new_tokens=cfg.max_new_tokens)
            spoke, response = parse_decision(gen.text)
            if mode == "interleaved":
                convo.append({"role": "assistant", "content": gen.text})
            polls.append(make_poll(t, spoke, response, gen))
            if cfg.verbose:
                tag = f"SPEAK: {response}" if spoke else "wait"
                print(f"    t={t:7.2f}s [{gen.n_images:2d} frm {gen.latency_s:5.2f}s] {tag}",
                      flush=True)
            prev_t = t
        return polls


class OracleTimingProtocol(_OracleProtocol):
    name = "oracle-timing"

    def ticks(self, item, cfg, is_reveal):
        # Keep visual content generation; supply only an ideal trigger schedule.
        return ([] if item.should_remain_silent else
                [item.question_time_s] if is_reveal else
                sorted({a.time_s for a in item.timed_answers}))

    def edit(self, messages, item, t):
        messages[-1]["content"].append({"type": "text", "text":
            "An oracle controller requires a response at this instant. "
            "Output DECISION: SPEAK and RESPONSE: followed by your answer "
            "using only the video frames. Do not output WAIT."})


class OraclePerceptionProtocol(_OracleProtocol):
    name = "oracle-perception"

    def edit(self, messages, item, t):
        facts = []
        for answer in item.timed_answers:
            release = answer.evidence_time_s if answer.evidence_time_s is not None else answer.time_s
            # Never reveal the reference before its annotated evidence exists.
            if release <= t:
                facts.append(f"[observed at {release:.3f}s] {answer.content}")
        observed = "\n".join(facts) or "No reference fact has become available yet."
        for message in messages:
            if isinstance(message.get("content"), list):
                message["content"] = [part for part in message["content"] if part.get("type") != "image"]
        messages[-1]["content"].append({"type": "text", "text":
            "Privileged reference facts replace the images at this tick:\n" + observed +
            "\nDecide whether to speak now. Fact availability alone does not require repeated speech."})


register_protocol(OracleTimingProtocol.name, OracleTimingProtocol, overwrite=True)
register_protocol(OraclePerceptionProtocol.name, OraclePerceptionProtocol, overwrite=True)


# ------------------------------------------------------------------ restraint

def cmd_restraint(args) -> None:
    from interactionbench.data import iter_items, load_benchmark
    from interactionbench.metrics import MetricConfig, decision_timing, merge_coincident

    preset = RESTRAINT_PRESETS[args.preset]
    runs = [r for _, r in parse_runs(args.runs, [(r, r) for r in preset["runs"]])]
    items_fp = args.items if args.items is not None else preset["items"]
    items = {i.item_id: i for i in iter_items(load_benchmark(args.data))}
    keep = None
    if items_fp:
        keep = set(l for l in Path(items_fp).read_text().splitlines() if l.strip())
    target_root = Path(args.target_root or args.runs_root)
    for baseline in runs:
        directory = Path(args.runs_root) / baseline
        target = target_root / (baseline + args.target_suffix)
        target.mkdir(parents=True, exist_ok=True)
        output = []
        removed = 0
        for line in (directory / 'preds.jsonl').read_text().splitlines():
            row = json.loads(line)
            iid = f"{row['video_id']}#{row['item_index']}"
            if keep is not None and iid not in keep:
                continue
            item = items[iid]
            old = sorted([e for e in row['emissions'] if e.get('content', '').strip()], key=lambda e: e['t'])
            if item.should_remain_silent:
                new = []
            else:
                times = ([item.question_time_s] if item.time_type == 'A' else
                         [t for t, _ in merge_coincident(item.timed_answers)])
                new = list(decision_timing(times, old, MetricConfig())['matched_events'].values())
                # Validate the claimed intervention using the scorer itself.
                assert decision_timing(times, new, MetricConfig())['n_violations'] == 0
            assert all(e in old for e in new)
            removed += len(old) - len(new)
            row['emissions'] = new
            row['run'] = target.name
            row['oracle'] = 'restraint: retain only first matched response per reference window'
            output.append(row)
        if keep is not None:
            assert len(output) == len(keep)
        (target / 'preds.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output))
        (target / 'config.json').write_text(json.dumps({'baseline': baseline, 'n_items': len(output),
            'removed_emissions': removed, 'pre_tol_s': 1.0, 'privileged_postprocessing': True,
            'preserves_retained_text_and_timestamps': True}, indent=2))
        print(target.name, len(output), 'deleted', removed, flush=True)


# ------------------------------------------------------------------ manifest

def cmd_manifest(args) -> None:
    out = Path(args.runs_root) / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / 'oracle_manifest.json').write_text(json.dumps({
        'oracle': args.oracle, 'privileged_diagnostic': True,
        'reference_content_in_model_input': args.oracle == 'perception',
        'facts_release': 'evidence_time_s if present, otherwise answer time_s',
        'timing_schedule': 'question time for A; reference times for B/C; none for negatives',
        'shared_baseline': args.baseline,
        'interpretation': ORACLE_DOC}, indent=2))
    print("wrote", out / 'oracle_manifest.json')


# ------------------------------------------------------------------ score

def cmd_score(args) -> None:
    r = Path(args.runs_root) / args.run
    ids = set(l for l in Path(args.items).read_text().splitlines() if l.strip())
    preds = [json.loads(l) for l in (r / 'preds.jsonl').read_text().splitlines() if l.strip()]
    assert ids <= {f"{x['video_id']}#{x['item_index']}" for x in preds}, 'Unfinished run: do not report missing jobs as measured silence'
    out = r / args.eval_name
    kw = dict(items=args.items, pre_tol=args.pre_tol, delta=args.delta,
              judge_prompt=args.judge_prompt)
    judge_args = {"cache_dir": args.judge_cache_dir} if args.judge.startswith("ensemble:") else None
    run_eval(r / 'preds.jsonl', args.data, args.mcq_key, out_dir=out,
             judge=args.judge, judge_args=judge_args, **kw)
    s = json.loads((out / 'summary.json').read_text())
    if s['config']['judge_cache_misses'] and args.gpu_judge:
        run_eval(r / 'preds.jsonl', args.data, args.mcq_key, out_dir=out,
                 judge=args.gpu_judge_spec, judge_cache=str(r / 'judge_cache.jsonl'), **kw)
        s = json.loads((out / 'summary.json').read_text())
    s['experiment_status'] = {'failed_items': sum(bool(x.get('failed') or x.get('error')) for x in preds if f"{x['video_id']}#{x['item_index']}" in ids),
                              'complete': True, 'final': s['config']['judge_cache_misses'] == 0}
    (out / 'summary.json').write_text(json.dumps(s, indent=2))
    print(args.run, s['overall'], 'status', s['experiment_status'], flush=True)


# ------------------------------------------------------------------ collect

def cmd_collect(args) -> None:
    from datetime import datetime, timezone

    ids = set(l for l in Path(args.items).read_text().splitlines() if l.strip())
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'target_items': len(ids), 'runs': {}}
    for key, name in parse_runs(args.runs, COLLECT_RUNS):
        p = Path(args.runs_root) / name
        rows = [json.loads(l) for l in (p / 'preds.jsonl').read_text().splitlines() if l.strip()] if (p / 'preds.jsonl').exists() else []
        rows = [r for r in rows if f"{r['video_id']}#{r['item_index']}" in ids]
        entry = {'run': name, 'recorded_items': len(rows), 'failed_items': sum(bool(r.get('failed') or r.get('error')) for r in rows)}
        f = p / args.eval_name / 'summary.json'
        if f.exists():
            s = json.loads(f.read_text()); entry['summary'] = s
            entry['final'] = len(rows) == len(ids) and s.get('experiment_status', {}).get('final', False)
        else:
            entry['final'] = False
        report['runs'][key] = entry
    out = Path(args.out) / 'status.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {'recorded': v['recorded_items'], 'failed': v['failed_items'], 'final': v['final']} for k, v in report['runs'].items()}, indent=2))
    print("wrote", out)


# ------------------------------------------------------------------ cli

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("restraint", help="delete the emissions a restraint oracle would suppress")
    add_common_args(p)
    p.add_argument("--preset", choices=sorted(RESTRAINT_PRESETS), default="subset103",
                   help="run list and item subset (default: %(default)s)")
    p.add_argument("--runs", nargs="*", default=None,
                   help="run names, or a file with one name per line; replaces the "
                        "list of the preset")
    p.add_argument("--items", default=None,
                   help="file of item_ids; replaces the subset of the preset "
                        "(pass an empty string for all items)")
    p.add_argument("--target-root", default=None,
                   help="directory that receives the filtered runs (default: --runs-root)")
    p.add_argument("--target-suffix", default="_oracle_restraint",
                   help="suffix of the name of the filtered run (default: %(default)s)")
    p.set_defaults(func=cmd_restraint)

    p = sub.add_parser("manifest", help="write oracle_manifest.json of a timing or perception run")
    add_common_args(p)
    p.add_argument("--oracle", choices=["perception", "timing"], required=True)
    p.add_argument("--run", required=True, help="run name under --runs-root")
    p.add_argument("--baseline", default="qwen3vl-8b_sliding_iv1_mcqv4_full",
                   help="run the oracle is compared with (default: %(default)s)")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("score", help="score a finished run with the paper protocol")
    add_common_args(p)
    p.add_argument("run", help="run name under --runs-root")
    p.add_argument("--items", default="benchmark/splits/subset103.txt",
                   help="file of item_ids the run has to cover (default: %(default)s)")
    p.add_argument("--eval-name", default="eval_paper",
                   help="evaluation directory name (default: %(default)s)")
    p.add_argument("--pre-tol", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--judge", default="ensemble:qwen_qwen3-14b",
                   help="judge spec of the first pass (default: %(default)s)")
    p.add_argument("--judge-cache-dir", default="results/judge_cache",
                   help="directory of the stored verdicts of an ensemble judge")
    p.add_argument("--judge-prompt", default="v2", help="grading prompt version")
    p.add_argument("--gpu-judge", action="store_true",
                   help="when verdicts are missing, score again with --gpu-judge-spec "
                        "(verdict file <run>/judge_cache.jsonl)")
    p.add_argument("--gpu-judge-spec", default="hf:Qwen/Qwen3-14B")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("collect", help="status of a list of runs")
    add_common_args(p)
    p.add_argument("--runs", nargs="*", default=None,
                   help="entries 'key=run_name' or 'run_name', or a file with one entry "
                        "per line (default: the twelve runs of the paper diagnostics)")
    p.add_argument("--items", default="benchmark/splits/subset103.txt",
                   help="file of item_ids (default: %(default)s)")
    p.add_argument("--eval-name", default="eval_paper",
                   help="evaluation directory name (default: %(default)s)")
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    resolve_common(args)
    args.func(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import add_common_args, parse_runs, resolve_common, run_eval  # noqa: E402
    main()
