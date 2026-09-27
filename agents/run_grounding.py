#!/usr/bin/env python3
"""Setting 2: offline temporal grounding evaluation of a visual agent.

What it does
  The agent receives the path of the FULL video and the question, and lists the
  moments at which it would have spoken, one line per moment: "[t=12.3] <utterance>".
  The claimed times become emission times, so `ibench eval` applies the same timing,
  silence and accuracy scores as for the polling runs. This setting is not real time:
  the agent sees the whole video, including the part after each answer. Its results
  are reported separately from the polling results.
  For items of time type A the agent returns one line with the evidence time and the
  answer; the emission is placed at question_time_s and the claimed evidence time is
  kept in the field "claimed_t".
  An item whose output contains neither a parsable line nor NO_RESPONSE is written to
  raw/<item>.flake.json and not to preds.jsonl; the next invocation runs it again.

Harnesses (--harness)
  claude   claude -p PROMPT --mcp-config <agent-home>/.mcp.json
           --allowedTools <seven Qwen-MM-Plugins tools> --model sonnet
  gemini   gemini -p PROMPT --yolo   (GEMINI_CLI_TRUST_WORKSPACE=true; MCP server and
           excluded tools come from <agent-home>/.gemini/settings.json)
  openai   agents/openai_agent.py PROMPT  (OpenAI-compatible tool-calling loop)

Upstream code
  Qwen-MM-Plugins, https://github.com/QwenLM/Qwen-MM-Plugins (capability "core",
  started through `uvx`); the path of the checkout is read from QWEN_MM_PLUGINS_ROOT.
  The checkout used with this script was version 1.0.8 (commit ab339d2).

Environment
  Python 3.10 or later with the interactionbench package (repository root), uv (for
  uvx), ffmpeg (used by the toolbox), and the CLI of the chosen harness or the
  variables of agents/openai_agent.py. API keys are read from the environment by the
  CLI tools themselves (ANTHROPIC_API_KEY, GEMINI_API_KEY).

Commands used for the paper (run from the repository root; the command was repeated
until preds.jsonl held all 1060 items)
  claude_qwenmm_grounding:
    python agents/run_grounding.py --harness claude --jobs 4 --timeout 1800
  openai_qwenmm_grounding:
    python agents/run_grounding.py --harness openai --jobs 12 --timeout 900
    (see agents/README.md for the 17 items of this run that came from a later pass)

Output
  <out>/preds.jsonl            one line per item (evaluator schema), appended
  <out>/raw/<item>.json        last 600 characters of the agent output, wall time
  <out>/raw/<item>.flake.json  the same for an item without a usable output
  <out>/config.json            configuration of the last invocation
  Default <out>: results/runs/<harness>_qwenmm_grounding
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Set in main(): HE is the agent home directory (working directory of the agent
# sessions), VID the directory of the H.264 videos.
HE = VID = ""
LETTERS = "ABCDEF"
CLAUDE_TOOLS = ",".join(
    f"mcp__qwen-mm-plugins-core__{t}" for t in
    ("read_video", "read_image", "media_info", "crop", "draw_bbox", "visualize", "save_view"))

CAP_HINT = {
    "PTR": "Respond only at the exact moment the requested event happens; never if it never happens.",
    "TOA": "Answer at the earliest moment the question becomes decidable.",
    "CST": "Each time the tracked state changes (e.g. the count increases), respond with the update (counts: just the new number).",
    "BRC": "Respond whenever the correct answer changes, correcting yourself if needed.",
    "LCG": "One short line for each new step or scene as it happens, like a live guide.",
}

TPL_STREAM = """Watch the video file {video} with the read_video tool — it is {dur:.0f} seconds long. Locate times precisely: scan coarsely first, then re-read around each candidate moment to pin the time down to about a second.

Imagine this request was given at the START of a LIVE stream of this video:
"{question}"

{hint}

Report every moment you would have spoken if watching live. Anchor each timestamp to the exact video second where the triggering content is VISIBLE on screen — not earlier (no anticipation), not later (no added reaction delay). Your FINAL output must contain ONLY lines in EXACTLY this format (time in seconds on the video timeline):
[t=12.3] <the exact thing you would say at that moment>

Rules: increasing time order; one line per distinct moment; no other text. If you would never respond, output exactly: NO_RESPONSE"""

TPL_A = """Watch the video file {video} with the read_video tool — it is {dur:.0f} seconds long; sample as many frames as you need, and re-read around candidate moments to pin times down to about a second.

At t={qt:.1f}s of the video, the user asks:
"{question}"

Answer using ONLY content visible up to t={qt:.1f}s, and give the video second where the KEY EVIDENCE for your answer is visible (must be <= {qt:.1f}). Your FINAL output must be EXACTLY one line (evidence time in seconds, then your answer{letter_hint}):
[t=12.3] <your answer>"""


def load_items(subset, data, mcq_path):
    from interactionbench.data import iter_items, load_benchmark
    from interactionbench.mcq import load_mcq_options
    keep = {l.strip() for l in open(subset) if l.strip()}
    mcq = load_mcq_options(mcq_path)
    items = []
    for x in iter_items(load_benchmark(data), only_valid=False):
        iid = x.item_id
        if iid not in keep:
            continue
        gt = " ".join(str((t.content or "")) for t in x.answers)
        items.append({
            "item_id": iid, "video_id": x.video_id, "item_index": x.item_index,
            "capability": x.capability,
            "time_type": x.time_type,
            "interaction_type": x.interaction_type,
            "question": x.question,
            "question_time_s": x.question_time_s,
            "duration_s": x.duration_s,
            "is_negative": x.is_negative
                          or "SHOULD_REMAIN_SILENT" in gt,
            "mcq": mcq.get(iid)})
    return items


def build_prompt(it, video):
    q = it["question"]
    letter_hint = ""
    if it["mcq"]:
        opts = "\n".join(f"{LETTERS[k]}. {o}"
                         for k, o in enumerate(it["mcq"]["options"]))
        # The question line is the annotation question. The "stem" field of the
        # options file is not used in this setting: the paper runs were produced with
        # the annotation question, and the two texts differ for part of the items.
        stem = q
        q = f"{stem}\n{opts}\n(Respond with just the option letter.)"
        letter_hint = " — just the option letter"
    if it["time_type"] == "A":
        return TPL_A.format(video=video, dur=it["duration_s"],
                            qt=it["question_time_s"], question=q,
                            letter_hint=letter_hint)
    hint = CAP_HINT.get(it["capability"],
                        "Respond only when the moment calls for it.")
    return TPL_STREAM.format(video=video, dur=it["duration_s"],
                             question=q, hint=hint)


_LINE = re.compile(r"\[?\s*t\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*\]?\s*(.+)")


TEMPLATE_ECHOES = ("the exact thing you would say", "<your answer>")


def parse_emissions(text, it):
    out = []
    for line in (text or "").splitlines():
        m = _LINE.match(line.strip())
        if m and m.group(2).strip() and not any(
                x in m.group(2) for x in TEMPLATE_ECHOES):
            out.append({"t": float(m.group(1)), "content": m.group(2).strip()})
    if it["time_type"] == "A" and out:
        out = [{"t": it["question_time_s"], "content": out[0]["content"],
                "claimed_t": out[0]["t"]}]
    return sorted(out, key=lambda e: e["t"])


def run_one(harness, prompt, timeout):
    if harness == "claude":
        cmd = ["claude", "-p", prompt, "--mcp-config", f"{HE}/.mcp.json",
               "--allowedTools", CLAUDE_TOOLS, "--model", "sonnet"]
        env = dict(os.environ)
    elif harness == "openai":
        cmd = [sys.executable, os.path.join(HERE, "openai_agent.py"), prompt]
        env = dict(os.environ)
    else:
        cmd = ["gemini", "-p", prompt, "--yolo"]
        env = dict(os.environ, GEMINI_CLI_TRUST_WORKSPACE="true")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HE, env=env, capture_output=True,
                           text=True, timeout=timeout)
        out, err = (r.stdout or "").strip(), (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        out, err = "", "TIMEOUT"
    return out, err, round(time.time() - t0, 1)


def main():
    global HE, VID
    ap = argparse.ArgumentParser(
        description="Setting 2: offline temporal grounding evaluation of a visual agent.")
    ap.add_argument("--harness", required=True, choices=["claude", "gemini", "openai"])
    ap.add_argument("--items", default="benchmark/splits/all1060.txt",
                    help="file with one item_id per line "
                         "(default: benchmark/splits/all1060.txt)")
    ap.add_argument("--data", default=os.environ.get("IBENCH_DATA", "data/interactionbench"),
                    help="benchmark root directory (default: $IBENCH_DATA or "
                         "data/interactionbench)")
    ap.add_argument("--mcq", default="auto",
                    help="multiple-choice options file "
                         "(default: <data>/mcq/mcq_options_v4.jsonl)")
    ap.add_argument("--video-dir", default=os.environ.get("IBENCH_VIDEOS_H264", ""),
                    help="directory with <video_id>.mp4 in H.264 (default: "
                         "$IBENCH_VIDEOS_H264 or <data>/videos_h264)")
    ap.add_argument("--agent-home", default=os.environ.get("IBENCH_AGENT_HOME", "agent_home"),
                    help="working directory of the agent sessions; holds .mcp.json and "
                         ".gemini/settings.json (default: $IBENCH_AGENT_HOME or "
                         "agent_home)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=480)
    ap.add_argument("--out", default="",
                    help="override results/runs/<harness>_qwenmm_grounding")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from common import capability_roundrobin, check_agent_setup

    # Absolute paths: the agent runs in <agent-home> and receives the video path in
    # the prompt.
    HE = os.path.abspath(args.agent_home)
    VID = os.path.abspath(args.video_dir or f"{args.data}/videos_h264")
    check_agent_setup(args.harness, HE)
    mcq_path = (f"{args.data}/mcq/mcq_options_v4.jsonl" if args.mcq == "auto"
                else args.mcq)

    items = load_items(args.items, args.data, mcq_path)
    run_tag = f"{args.harness}_qwenmm_grounding"
    out_dir = args.out or f"results/runs/{run_tag}"
    raw_dir = f"{out_dir}/raw"
    os.makedirs(raw_dir, exist_ok=True)
    preds_fp = f"{out_dir}/preds.jsonl"
    done = set()
    if os.path.exists(preds_fp):
        done = {f"{json.loads(l)['video_id']}#{json.loads(l)['item_index']}"
                for l in open(preds_fp) if l.strip()}
    todo = [it for it in items if it["item_id"] not in done
            and os.path.exists(f"{VID}/{it['video_id']}.mp4")]
    todo = capability_roundrobin(todo)
    if args.limit:
        todo = todo[: args.limit]

    json.dump({
        "harness": args.harness, "run": run_tag,
        "track": "offline temporal grounding (not real time; reported "
                 "separately from the polling runs)",
        "backend": "qwen-mm-plugins-core (local MCP)",
        "prompt_version": "v2 time-precision (timestamps anchored to visible evidence, coarse-to-fine localization, A-type evidence time <= question time)",
        "agent_model": "claude-sonnet (CLI)" if args.harness == "claude"
                       else "gemini CLI default",
        "n_total": len(items), "n_this_pass": len(todo),
        "timeout_s": args.timeout, "jobs": args.jobs,
        "item_set_sha1": hashlib.sha1(json.dumps(
            sorted(x["item_id"] for x in items)).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, open(f"{out_dir}/config.json", "w"), indent=1)
    print(f"{run_tag}: {len(todo)} to run ({len(done)} done)", flush=True)

    def work(it):
        iid = it["item_id"]
        video = f"{VID}/{it['video_id']}.mp4"
        prompt = build_prompt(it, video)
        out, err, wall = run_one(args.harness, prompt, args.timeout)
        emissions = parse_emissions(out, it)
        if it["is_negative"] and "NO_RESPONSE" in (out or ""):
            emissions = []
        for e in emissions:
            e["latency_s"] = wall
        row = {"video_id": it["video_id"], "item_index": it["item_index"],
               "model": f"{args.harness}-agent", "run": run_tag,
               "emissions": emissions, "n_polls": 1,
               "poll_latencies": [wall]}
        if emissions or "NO_RESPONSE" in (out or ""):
            with open(preds_fp, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            open(f"{raw_dir}/{iid.replace('#','_i')}.json", "w").write(
                json.dumps({"item_id": iid, "raw_tail": out[-600:],
                            "err": err, "wall_s": wall},
                           ensure_ascii=False, indent=1))
            mark = f"{len(emissions)}em"
        else:
            open(f"{raw_dir}/{iid.replace('#','_i')}.flake.json", "w").write(
                json.dumps({"item_id": iid, "raw_tail": (out or "")[-600:],
                            "err": err, "wall_s": wall},
                           ensure_ascii=False, indent=1))
            mark = "flake"
        print(f"  {iid} {mark} {wall}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(work, todo))
    print("PASS DONE", flush=True)


if __name__ == "__main__":
    main()
