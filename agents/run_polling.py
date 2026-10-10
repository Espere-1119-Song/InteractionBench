#!/usr/bin/env python3
"""Setting 1: online polling evaluation of a visual agent."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HE = VID = TRIM = ""
LETTERS = "ABCDEF"
CLAUDE_TOOLS = ",".join(
    f"mcp__qwen-mm-plugins-core__{t}" for t in
    ("read_video", "read_image", "media_info", "crop", "draw_bbox", "visualize", "save_view"))
CLAUDE_BUILTIN_TOOLS = "Bash,Read,Glob,Grep,Write,Edit"

SYSTEM = ("You are a real-time visual assistant watching a LIVE video stream. "
          "The video file contains ONLY the stream so far — nothing after the "
          "current moment exists yet. Decide RIGHT NOW whether to speak or "
          "keep watching silently. Only speak when this exact moment calls "
          "for it; do not speak early; never repeat what you already said.")
FORMAT = ("Reply in EXACTLY this format, nothing else:\n"
          "DECISION: SPEAK or WAIT\n"
          "RESPONSE: <if SPEAK, one short line; if WAIT, leave blank>")
CAP_HINT = {
    "PTR": "Speak only at the exact moment the requested event happens; if it never happens, stay silent for the whole stream.",
    "TOA": "Answer at the earliest moment the question becomes decidable; before that, wait.",
    "CST": "Each time the tracked state changes (e.g. the count increases), speak the update (counts: just the new number); otherwise wait.",
    "BRC": "Speak whenever the correct answer changes, correcting your earlier answer if needed.",
    "LCG": "Speak one short line for each new step or scene as it happens, like a live guide.",
    "IVQA": "Answer the question as soon as you are asked.",
    "CIR": "Answer as soon as it becomes answerable.",
    "LVM": "Answer from what you saw earlier in the stream, immediately.",
}


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
        items.append({
            "item_id": iid, "video_id": x.video_id, "item_index": x.item_index,
            "capability": x.capability,
            "time_type": x.time_type,
            "question": x.question,
            "question_time_s": x.question_time_s,
            "duration_s": x.duration_s,
            "mcq": mcq.get(iid)})
    return items


def q_text(it):
    if not it["mcq"]:
        return it["question"]
    opts = "\n".join(f"{LETTERS[k]}. {o}"
                     for k, o in enumerate(it["mcq"]["options"]))
    stem = it["question"]
    return f"{stem}\n{opts}\n(When you speak, answer with just the option letter.)"


def trim(src, t, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    r = subprocess.run(["ffmpeg", "-y", "-i", src, "-t", f"{t:.2f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
                        "-an", "--", dst], capture_output=True)
    return r.returncode == 0 and os.path.getsize(dst) > 0


DEC_RE = re.compile(r"DECISION:\s*(SPEAK|WAIT)", re.I)
PLACEHOLDER = "if SPEAK, one short line"
RESP_RE = re.compile(r"RESPONSE:\s*(.*)", re.I | re.S)


def agent_call(harness, prompt, timeout, cwd=None, mcp=True, model="sonnet"):
    cwd = cwd or HE
    if harness == "claude":
        if mcp:
            cmd = ["claude", "-p", prompt, "--mcp-config", f"{HE}/.mcp.json",
                   "--allowedTools", CLAUDE_TOOLS, "--model", model]
        else:
            cmd = ["claude", "-p", prompt, "--strict-mcp-config",
                   "--tools", CLAUDE_BUILTIN_TOOLS,
                   "--allowedTools", CLAUDE_BUILTIN_TOOLS, "--model", model]
        env = dict(os.environ)
    elif harness == "openai":
        cmd = [sys.executable, os.path.join(HERE, "openai_agent.py"), prompt]
        env = dict(os.environ)
    else:
        cmd = ["gemini", "-p", prompt, "--yolo"]
        env = dict(os.environ, GEMINI_CLI_TRUST_WORKSPACE="true")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return (out or "").strip()
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
        proc.wait()
        return ""


def main():
    global HE, VID, TRIM
    ap = argparse.ArgumentParser(
        description="Setting 1: online polling evaluation of a visual agent.")
    ap.add_argument("--harness", required=True, choices=["claude", "gemini", "openai"])
    ap.add_argument("--items", "--subset", dest="subset",
                    default="benchmark/splits/subset103.txt",
                    help="file with one item_id per line "
                         "(default: benchmark/splits/subset103.txt)")
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
                    help="working directory of the agent sessions; holds .mcp.json, "
                         ".gemini/settings.json and the tick clips (default: "
                         "$IBENCH_AGENT_HOME or agent_home)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--a-window", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=6,
                    help="parallel ITEMS (each item's ticks stay sequential)")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", "--out-dir", dest="out_dir", default="",
                    help="override results/runs/<harness>_qwenmm_polling_iv<interval>")
    ap.add_argument("--no-mcp", action="store_true",
                    help="claude only: no qwen-mm MCP server, built-in tools only")
    ap.add_argument("--item-cwd", action="store_true",
                    help="run each item's sessions in its own scratch cwd under "
                         "<agent-home>/work/ (deleted afterwards), so frames "
                         "an agent extracts never leak across items")
    ap.add_argument("--model", default="sonnet", help="claude --model alias")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from common import capability_roundrobin, check_agent_setup

    HE = os.path.abspath(args.agent_home)
    VID = os.path.abspath(args.video_dir or f"{args.data}/videos_h264")
    TRIM = f"{HE}/videos_tick"
    check_agent_setup(args.harness, HE, mcp=not args.no_mcp)
    os.makedirs(TRIM, exist_ok=True)
    mcq_path = (f"{args.data}/mcq/mcq_options_v4.jsonl" if args.mcq == "auto"
                else args.mcq)

    items = load_items(args.subset, args.data, mcq_path)
    run_tag = f"{args.harness}_qwenmm_polling_iv{args.interval:g}"
    out_dir = args.out_dir or f"results/runs/{run_tag}"
    run_tag = os.path.basename(out_dir.rstrip("/"))
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
        "protocol": ("polling: one full agent session every "
                     f"{args.interval}s of stream time; video "
                     "trimmed to now; own prior utterances in prompt; silence "
                     "instructed by default"),
        "subset": args.subset, "n_subset": len(items),
        "interval_s": args.interval, "a_window_s": args.a_window,
        "timeout_s": args.timeout, "jobs": args.jobs,
        "mcp": not args.no_mcp, "model": args.model if args.harness == "claude" else None,
        "claude_tools": (CLAUDE_BUILTIN_TOOLS if args.no_mcp else CLAUDE_TOOLS)
                        if args.harness == "claude" else None,
        "item_cwd": args.item_cwd, "agent_home": HE,
        "item_set_sha1": hashlib.sha1(json.dumps(
            sorted(x["item_id"] for x in items)).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, open(f"{out_dir}/config.json", "w"), indent=1)
    print(f"{run_tag}: {len(todo)} items ({len(done)} done)", flush=True)

    def work(it):
        iid = it["item_id"]
        src = f"{VID}/{it['video_id']}.mp4"
        is_A = it["time_type"] == "A"
        q_t = it["question_time_s"]
        dur = it["duration_s"] or 60.0
        end_s = min(dur, q_t + args.a_window) if is_A else dur
        start_s = q_t if is_A else args.interval
        hint = CAP_HINT.get(it["capability"],
                            "Speak only when it is the right moment.")
        question = q_text(it)
        emissions, polls = [], []
        cwd = HE
        if args.item_cwd:
            cwd = f"{HE}/work/{iid.replace('#','_i')}"
            os.makedirs(cwd, exist_ok=True)
        t = start_s
        while t <= end_s + 1e-6:
            tv = f"{TRIM}/{iid.replace('#','_i')}_{t:07.1f}.mp4"
            if not trim(src, max(t, 1.0), tv):
                t += args.interval
                continue
            said = "".join(f'\n- [t={e["t"]:.1f}s] you said: "{e["content"]}"'
                           for e in emissions[-3:])
            prior = (f"\nYou have already spoken during this stream:{said}\n"
                     if emissions else "")
            asked = (f'The user JUST asked: "{question}"' if is_A else
                     f'The user\'s standing request (given at stream start): "{question}"')
            prompt = (f"{SYSTEM}\n\nThe live stream so far is the video file "
                      f"{tv} (current stream time: {t:.1f}s — the file ends at "
                      f"the current moment). Watch it with the read_video tool."
                      f"\n\n{asked}\n{hint}{prior}\n{FORMAT}")
            t0 = time.monotonic()
            raw = agent_call(args.harness, prompt, args.timeout, cwd=cwd,
                             mcp=not args.no_mcp, model=args.model)
            lat = round(time.monotonic() - t0, 3)
            m = DEC_RE.search(raw)
            spoke = bool(m and m.group(1).upper() == "SPEAK")
            resp = ""
            if spoke:
                rm = RESP_RE.search(raw)
                resp = (rm.group(1).strip().splitlines()[0].strip()
                        if rm and rm.group(1).strip() else "")
            if resp and PLACEHOLDER in resp:
                resp = ""
            polls.append({"t": round(t, 2), "spoke": spoke,
                          "latency_s": lat, "raw": raw[-150:]})
            if spoke and resp:
                emissions.append({"t": round(t, 2), "content": resp,
                                  "latency_s": lat})
            try:
                os.remove(tv)
            except OSError:
                pass
            t += args.interval
        if args.item_cwd:
            shutil.rmtree(cwd, ignore_errors=True)
        lats = [p["latency_s"] for p in polls]
        row = {"video_id": it["video_id"], "item_index": it["item_index"],
               "model": f"{args.harness}-agent", "run": run_tag,
               "emissions": emissions, "n_polls": len(polls),
               "poll_latencies": lats,
               "poll_latency_mean": round(sum(lats) / len(lats), 3) if lats else None}
        with open(preds_fp, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        open(f"{raw_dir}/{iid.replace('#','_i')}.json", "w").write(
            json.dumps({"item_id": iid, "question": question, "polls": polls},
                       ensure_ascii=False, indent=1))
        print(f"  {iid}: {len(polls)} ticks, {len(emissions)} emissions, "
              f"mean latency {row['poll_latency_mean']}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(work, todo))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
