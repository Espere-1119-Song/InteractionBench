#!/usr/bin/env python3
"""Rebuild the Flash-VStream emissions from the raw poll dumps with a lenient parser.

The strict regular expression of ``baselines/flash_vstream/run.py``
(DECISION: SPEAK / RESPONSE: ...) matched none of the replies of the model. A census
over the raw dumps of the paper run shows three variants instead: 'SPEAk\n<content>'
(27167 polls), a bare option letter 'C.' (25764), bare 'WAIT' (2678). This script
derives the emission stream of each item again from raw/*.json without touching the
original predictions, and writes a parallel run directory with the suffix
_lenientparse for side-by-side scoring.

Lenient rule per poll (first match wins):
  1. strict template  DECISION: SPEAK|WAIT (+ RESPONSE: ...)
  2. reply starts with the WAIT token  -> silence
  3. reply starts with the SPEAK token -> emission = text after that token/line
  4. bare option letter [A-E][.)]? -> emission = the letter
  5. empty -> silence
  6. anything else -> emission = first line verbatim (the model spoke content
     without any protocol token; counting it as silence is what set the
     original scores to zero)

A prediction line that carries an "error" field, or that has no raw dump, is copied
unchanged.

Upstream system: https://github.com/IVGSZ/Flash-VStream, checkpoint
zhang9302002/Flash-VStream-Qwen-7b (needed by run.py only).
Environment: Python >= 3.8, standard library only.

Command used for the paper numbers (run directory
fvstream-7b_polling_iv1_8fps_mcq_lenientparse):
  python baselines/flash_vstream/reparse.py \
      --src results/runs/fvstream-7b_polling_iv1_8fps_mcq

Input: <src>/preds.jsonl and the raw dumps <src>_*shard*/raw/*.json and
<src>/raw/*.json (a dump under <src>/raw takes precedence).
Output: <dst>/preds.jsonl, default <dst> = <src>_lenientparse.
"""
import argparse
import glob
import json
import os
import re

DEC = re.compile(r"DECISION:\s*(SPEAK|WAIT)", re.I)
RESP = re.compile(r"RESPONSE:\s*(.*)", re.I | re.S)
SPEAK0 = re.compile(r"^\s*SPEAK\w*\b[:.]?\s*", re.I)
WAIT0 = re.compile(r"^\s*WAIT\b", re.I)
LETTER = re.compile(r"^[A-E][\.\)]?$")

SRC = "results/runs/fvstream-7b_polling_iv1_8fps_mcq"


def parse(raw):
    """-> (spoke, content)"""
    raw = (raw or "").strip()
    m = DEC.search(raw)
    if m:
        if m.group(1).upper() == "WAIT":
            return False, ""
        rm = RESP.search(raw)
        content = rm.group(1).strip().splitlines()[0].strip() if rm and rm.group(1).strip() else ""
        return True, content
    if WAIT0.match(raw):
        return False, ""
    if SPEAK0.match(raw):
        rest = SPEAK0.sub("", raw, count=1).strip()
        return True, rest.splitlines()[0].strip() if rest else ""
    if LETTER.match(raw):
        return True, raw
    if not raw:
        return False, ""
    return True, raw.splitlines()[0].strip()


def main():
    ap = argparse.ArgumentParser(
        description="Re-parse the raw Flash-VStream replies with the lenient parser.")
    ap.add_argument("--src", default=SRC,
                    help="run directory written by baselines/flash_vstream/run.py "
                         "(merged preds.jsonl)")
    ap.add_argument("--dst", default=None, help="output run directory, "
                                                 "default <src>_lenientparse")
    ap.add_argument("--shard-glob", default="{src}_*shard*",
                    help="pattern of the shard run directories whose raw/ dumps are "
                         "read in addition to <src>/raw; {src} is replaced by --src")
    args = ap.parse_args()
    src = args.src.rstrip("/")
    dst = args.dst or src + "_lenientparse"

    os.makedirs(dst, exist_ok=True)
    raw_index = {}
    shards = args.shard_glob.format(src=src)
    for fp in glob.glob(f"{shards}/raw/*.json") + glob.glob(f"{src}/raw/*.json"):
        raw_index[os.path.basename(fp)[:-5]] = fp

    n_re, n_kept, n_noraw = 0, 0, 0
    with open(f"{dst}/preds.jsonl", "w") as out:
        for l in open(f"{src}/preds.jsonl"):
            d = json.loads(l)
            key = f"{d['video_id']}#{d['item_index']}"
            fp = raw_index.get(key)
            if d.get("error") or not fp:
                if not d.get("error"):
                    n_noraw += 1
                n_kept += 1
                out.write(l)
                continue
            polls = json.load(open(fp)).get("polls", [])
            emissions = []
            for p in polls:
                spoke, content = parse(p.get("raw"))
                if spoke and content:
                    emissions.append({"t": p["t"], "content": content,
                                      "latency_s": p.get("latency_s")})
            d["emissions"] = emissions
            d["run"] = os.path.basename(dst)
            d["reparse"] = "lenient"
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_re += 1
    print(f"reparsed {n_re}, kept as-is {n_kept} (error-marked or no raw; no-raw={n_noraw})")


if __name__ == "__main__":
    main()
