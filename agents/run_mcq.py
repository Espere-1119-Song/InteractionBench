#!/usr/bin/env python3
"""Setting 3: offline multiple-choice accuracy of a visual agent."""
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
HE = VID = TRIM = ""
LETTERS = "ABCDEF"

CLAUDE_TOOLS = ",".join(
    f"mcp__qwen-mm-plugins-core__{t}" for t in
    ("read_video", "read_image", "media_info", "crop", "draw_bbox", "visualize", "save_view"))


def load_items(subset, data, mcq_path, key_path):
    from interactionbench.data import iter_items, load_benchmark
    from interactionbench.mcq import load_mcq_options
    keep = {l.strip() for l in open(subset) if l.strip()}
    ann = {x.item_id: x for x in iter_items(load_benchmark(data), only_valid=False)}
    key = {}
    for l in open(key_path, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            key[d["item_id"]] = d
    items, g = [], {}
    for iid, o in load_mcq_options(mcq_path).items():
        if iid not in keep:
            continue
        x = ann[iid]
        items.append({
            "item_id": iid, "video_id": x.video_id,
            "capability": x.capability,
            "interaction_type": x.interaction_type,
            "question": x.question,
            "question_time_s": x.question_time_s,
            "options": o["options"], "mcq_stem": o.get("stem"),
            "answer_letter": key[iid]["correct_letter"]})
        g[iid] = {
            "times": sorted(t.time_s for t in x.answers
                            if t.time_s is not None),
            "duration": x.duration_s or 0}
    return items, g


def cutoff_s(it, g):
    times, dur = g.get("times") or [], g.get("duration") or 0
    q = it.get("question_time_s") or 0
    if it["interaction_type"] == "INS" and times:
        c = times[0] + 2.0
    else:
        c = max([q] + times[-1:]) or dur
    if dur:
        c = min(max(c, 5.0), dur)
    return c or 60.0


def trim(vid_path, cut, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", vid_path, "-t", f"{cut:.2f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-an",
         "--", out_path], capture_output=True)
    return r.returncode == 0 and os.path.getsize(out_path) > 0


def build_prompt(it, video):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"]))
    stem = it.get("mcq_stem") or it["question"]
    return (f"Watch the video file {video} (use the read_video tool; sample "
            f"frames as needed), then answer this question about it.\n\n"
            f"{stem}\n{opts}\n\n"
            f"Reply with ONLY the option letter (one character).")


def run_one(harness, it, video, timeout):
    prompt = build_prompt(it, video)
    if harness == "claude":
        cmd = ["claude", "-p", prompt, "--mcp-config", f"{HE}/.mcp.json",
               "--allowedTools", CLAUDE_TOOLS, "--model", "sonnet"]
        env = dict(os.environ)
    elif harness == "openai":
        cmd = [sys.executable, os.path.join(HERE, "openai_agent.py"), prompt]
        env = dict(os.environ)
    elif harness == "gemini":
        cmd = ["gemini", "-p", prompt, "--yolo"]
        env = dict(os.environ, GEMINI_CLI_TRUST_WORKSPACE="true")
    else:
        raise ValueError(harness)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HE, env=env, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or "").strip()
        err = (r.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        out, err = "", "TIMEOUT"
    wall = time.time() - t0
    n = len(it["options"])
    m = re.findall(rf"\b([{LETTERS[:n]}])\b", out[-200:] or "")
    letter = m[-1] if m else None
    return {"letter": letter, "raw_tail": out[-300:], "err": err,
            "wall_s": round(wall, 1)}


def main():
    global HE, VID, TRIM
    ap = argparse.ArgumentParser(
        description="Setting 3: offline multiple-choice accuracy of a visual agent.")
    ap.add_argument("--harness", required=True, choices=["claude", "gemini", "openai"])
    ap.add_argument("--items", default="benchmark/splits/mcq688.txt",
                    help="file with one item_id per line "
                         "(default: benchmark/splits/mcq688.txt)")
    ap.add_argument("--data", default=os.environ.get("IBENCH_DATA", "data/interactionbench"),
                    help="benchmark root directory (default: $IBENCH_DATA or "
                         "data/interactionbench)")
    ap.add_argument("--mcq", default="auto",
                    help="multiple-choice options file, shown to the agent "
                         "(default: <data>/mcq/mcq_options_v4.jsonl)")
    ap.add_argument("--mcq-key", default="auto",
                    help="answer key, used for scoring only "
                         "(default: <data>/mcq/mcq_key_v4.jsonl)")
    ap.add_argument("--video-dir", default=os.environ.get("IBENCH_VIDEOS_H264", ""),
                    help="directory with <video_id>.mp4 in H.264 (default: "
                         "$IBENCH_VIDEOS_H264 or <data>/videos_h264)")
    ap.add_argument("--agent-home", default=os.environ.get("IBENCH_AGENT_HOME", "agent_home"),
                    help="working directory of the agent sessions; holds .mcp.json, "
                         ".gemini/settings.json and the trimmed clips (default: "
                         "$IBENCH_AGENT_HOME or agent_home)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default=None,
                    help="override results/runs/<harness>_qwenmm_mcqv4")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    from common import capability_roundrobin, check_agent_setup

    HE = os.path.abspath(args.agent_home)
    VID = os.path.abspath(args.video_dir or f"{args.data}/videos_h264")
    TRIM = f"{HE}/videos_trim"
    check_agent_setup(args.harness, HE)
    os.makedirs(TRIM, exist_ok=True)
    mcq_path = (f"{args.data}/mcq/mcq_options_v4.jsonl" if args.mcq == "auto"
                else args.mcq)
    key_path = (f"{args.data}/mcq/mcq_key_v4.jsonl" if args.mcq_key == "auto"
                else args.mcq_key)

    items, g = load_items(args.items, args.data, mcq_path, key_path)
    run_tag = f"{args.harness}_qwenmm_mcqv4"
    out_dir = args.out or f"results/runs/{run_tag}"
    os.makedirs(out_dir, exist_ok=True)
    res_fp = f"{out_dir}/results.jsonl"
    done = set()
    if os.path.exists(res_fp):
        done = {json.loads(l)["item_id"] for l in open(res_fp) if l.strip()}

    todo = []
    for it in items:
        if it["item_id"] in done:
            continue
        src = f"{VID}/{it['video_id']}.mp4"
        if not os.path.exists(src):
            continue
        todo.append(it)
    todo = capability_roundrobin(todo)
    if args.limit:
        todo = todo[: args.limit]

    json.dump({
        "harness": args.harness, "backend": "qwen-mm-plugins-core (local MCP)",
        "agent_model": "claude-sonnet (CLI)" if args.harness == "claude"
                       else "gemini CLI default",
        "run": run_tag, "items_file": args.items,
        "mcq_options": mcq_path, "mcq_key": key_path,
        "video_protocol": "per-item trim to cutoff (INS first_event+2s; "
                          "QA max(q_time,last_event)); H.264, no audio",
        "n_total": len(items), "n_this_pass": len(todo),
        "timeout_s": args.timeout, "jobs": args.jobs,
        "item_set_sha1": hashlib.sha1(json.dumps(
            sorted(x["item_id"] for x in items)).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, open(f"{out_dir}/config.json", "w"), indent=1)

    print(f"{run_tag}: {len(todo)} to run ({len(done)} done)", flush=True)

    def work(it):
        iid = it["item_id"]
        cut = cutoff_s(it, g.get(iid, {}))
        tv = f"{TRIM}/{iid.replace('#', '_i')}.mp4"
        if not trim(f"{VID}/{it['video_id']}.mp4", cut, tv):
            row = {"item_id": iid, "error": "trim_failed"}
        else:
            r = run_one(args.harness, it, tv, args.timeout)
            row = {"item_id": iid, "gt": it["answer_letter"],
                   "cutoff_s": round(cut, 2), **r,
                   "correct": r["letter"] == it["answer_letter"]}
        with open(res_fp, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        mark = "✓" if row.get("correct") else ("✗" if row.get("letter") else "∅")
        print(f"  {mark} {iid} {row.get('letter')}/{it['answer_letter']} "
              f"{row.get('wall_s')}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(work, todo))

    for rp in range(2):
        rows = [json.loads(l) for l in open(res_fp)]
        bad = {r["item_id"] for r in rows
               if not r.get("letter") and not r.get("error")}
        if not bad:
            break
        keep = [r for r in rows if r["item_id"] not in bad]
        with open(res_fp, "w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        redo = [it for it in items if it["item_id"] in bad]
        print(f"retry pass {rp+1}: {len(redo)} unanswered", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(work, redo))

    rows = [json.loads(l) for l in open(res_fp)]
    ok = [r for r in rows if r.get("letter")]
    acc = sum(r["correct"] for r in ok) / len(ok) if ok else 0
    print(f"DONE {len(rows)} rows | answered {len(ok)} | acc {acc:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
