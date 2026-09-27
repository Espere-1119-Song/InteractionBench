# Data

The dataset is on the Hugging Face Hub:
[InteractionBench/InteractionBench](https://huggingface.co/datasets/InteractionBench/InteractionBench).
It contains the videos, the questions and the ground truth.

## Layout

```
<data>/
  annotations/<domain>/<video_id>.json     one file per video
  items.jsonl                              all items in one table (not read by the code)
  mcq/mcq_options_v4.jsonl                 shown to the system
  mcq/mcq_key_v4.jsonl                     answer key, used by the scorer only
  videos/<domain>/<video_id>.mp4
  videos_h264/<video_id>.mp4               optional H.264 proxies, built locally
```

`<data>` defaults to `data/interactionbench`; pass `--data` to use another location.

```bash
python scripts/prepare_data.py --data <data> download [--no-videos]
python scripts/prepare_data.py --data <data> h264       # only if needed, see below
python scripts/prepare_data.py --data <data> check
```

Scoring needs the annotations and the answer key. Generating predictions also needs the
videos and the options file.

## Annotation schema

```json
{
  "video_id": "...", "domain": "...", "video": "videos/<domain>/<video_id>.mp4",
  "duration_s": 183.4,
  "items": [
    {
      "capability": "PTR",
      "time_type": "B",
      "interaction_type": "INS",
      "range_length": "1-5min",
      "sub_tag": null,
      "is_negative": false,
      "auto_number": false,
      "question": "Tell me when ...",
      "question_time_s": 0,
      "answers": [ {"time_s": 41.2, "content": "..."} ]
    }
  ]
}
```

An item is identified by `<video_id>#<index in items>`.

| Field | Meaning |
|---|---|
| `domain` | collection directory of the video |
| `time_type` | `A`: the question is revealed at `question_time_s` and answered at once. `B`: standing request from t = 0, the answer becomes determinable at `answers[i].time_s`. `C`: standing request with many timed answers over the stream. |
| `interaction_type` | `QA` question, `INS` standing instruction |
| `range_length` | `0-1min`, `1-5min`, `5-20min` |
| `sub_tag` | `counting`, `goal`, `narration`, or null |
| `is_negative` | the requested event never happens; the system must stay silent. The answer content is `SHOULD_REMAIN_SILENT`. |
| `auto_number` | the answers are a running count |
| `answers[i].time_s` | earliest moment at which response `i` is warranted |
| `answers[i].evidence_time_s` | for retrospective questions, when the evidence was visible |

## Tasks

| Task in the paper | Annotation labels | Items | Response obligation |
|---|---|---|---|
| Look | `IVQA` | 159 | answer when asked, from the current view |
| Recall | `LVM`, `CIR` | 163 | answer when asked, from earlier evidence |
| Time | `TOA` | 126 | answer when an order or temporal condition resolves |
| Alert | `PTR` | 313 | speak when the requested event occurs |
| Track | `CST`, `BRC` | 192 | speak when a count or state changes |
| Commentate | `LCG` | 107 | speak one line for each new step or scene |

The scorer reports every label separately (`by_capability`). Its merged groups
(`by_capability_group`) are `IVQA+CIR` and `CST+BRC`; the first differs from the task
grouping above, where the 7 `CIR` items belong to Recall.

## Item lists

| File | Items | Used for |
|---|---|---|
| `benchmark/splits/all1060.txt` | 1,060 | the full set |
| `benchmark/splits/subset103.txt` | 103 | stratified subset for systems that are expensive to run (agents, API models, prompt ablations) |
| `benchmark/splits/mcq688.txt` | 688 | the items that have a multiple-choice form |
| `benchmark/splits/frozen218.txt` | 218 | an earlier frozen subset; one of its items is no longer in the annotations |

## Video codecs

Frame extraction in `ibench run` uses `ffmpeg` and reads every codec your `ffmpeg` build
supports. Several runners in `baselines/` decode with `decord`, which cannot read AV1.
For those, build H.264 proxies with `prepare_data.py h264` and pass
`--video-dir <data>/videos_h264`.
