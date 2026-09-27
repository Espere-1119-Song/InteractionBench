"""Example: score a system that does not fit the chat interface.

Natively streaming systems, agents, and human annotators are scored from a predictions
file. Produce one JSON line per item in the schema below, then run

    ibench eval my_preds.jsonl --data data/interactionbench --mcq-key

An emission is one thing the system said: the stream time at which it spoke and the
text. Silence is the absence of emissions. Items without a line are scored as silent.
"""

import json
import sys

from interactionbench import iter_items, load_benchmark
from interactionbench.mcq import load_mcq_options
from interactionbench.prompts import format_question


def my_system(video_path, question, item):
    """Replace with your system. Return [(stream_time_seconds, text), ...]."""
    return []


def main(data="data/interactionbench", out="my_preds.jsonl"):
    options = load_mcq_options(f"{data}/mcq/mcq_options_v4.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for item in iter_items(load_benchmark(data)):
            # A-type items: the question must not be shown before item.question_time_s
            question = format_question(item, options.get(item.item_id))
            video = f"{data}/videos/{item.domain}/{item.video_id}.mp4"
            emissions = my_system(video, question, item)
            f.write(json.dumps({
                "video_id": item.video_id,
                "item_index": item.item_index,
                "model": "my-system",
                "emissions": [{"t": float(t), "content": text} for t, text in emissions],
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
