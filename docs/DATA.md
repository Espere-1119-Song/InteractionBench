# Data

## Layout

```
<data>/
  results/<domain>/<video_id>/annotation.json     one file per video
  videos/<domain>/<video_id>.mp4
  videos_h264/<video_id>.mp4                       optional H.264 proxies
  mcq/mcq_options_v4.jsonl                         shown to the system
  mcq/mcq_key_v4.jsonl                             answer key, used by the scorer only
```

`<data>` defaults to `data/interactionbench`; pass `--data` to use another location.

```bash
python scripts/prepare_data.py --data <data> download [--no-videos] [--repo <hub dataset id>]
python scripts/prepare_data.py --data <data> mcq <mcq_items.jsonl>    # only if mcq/ is absent
python scripts/prepare_data.py --data <data> h264                     # only if needed, see below
python scripts/prepare_data.py --data <data> check
```

Scoring needs the annotations and the answer key. Generating predictions also needs the
videos and the options file.

## Annotation schema

```json
{
  "video_id": "...", "category": "<domain>", "duration_s": 183.4,
  "items": [
    {
      "capability": "PTR",
      "time_type": "B",
      "interaction_type": "INS",
      "range_length": "...",
      "is_negative": false,
      "auto_number": false,
      "question": "Tell me when ...",
      "question_time_s": 0.0,
      "answers": [ {"time_s": 41.2, "content": "...", "evidence_time_s": null} ]
    }
  ]
}
```

An item is identified by `<video_id>#<index in items>`.

| Field | Meaning |
|---|---|
| `time_type` | `A`: the question is revealed at `question_time_s` and answered at once. `B`: standing request from t = 0, the answer becomes determinable at `answers[i].time_s`. `C`: standing request with many timed answers over the stream. |
| `interaction_type` | `QA` question, `INS` standing instruction |
| `is_negative` | the requested event never happens; the system must stay silent. The answer content is `SHOULD_REMAIN_SILENT`. |
| `auto_number` | the answers are a running count |
| `answers[i].time_s` | earliest moment at which response `i` is warranted |
| `answers[i].evidence_time_s` | for retrospective questions, when the evidence was visible |

## Tasks

The annotation labels and the task names used in the paper:

| Label | Task in the paper | Response obligation |
|---|---|---|
| `IVQA` (with `CIR`, causal questions) | Look | answer when asked, from the current view |
| `LVM` | Recall | answer when asked, from earlier evidence |
| `TOA` | Time | answer when an order or temporal condition resolves |
| `PTR` | Alert | speak when the requested event occurs |
| `CST` (with `BRC`, belief revision) | Track | speak when a count or state changes |
| `LCG` | Commentate | speak one line for each new step or scene |

The scorer reports `IVQA+CIR` and `CST+BRC` as merged groups (`by_capability_group`)
and every label separately (`by_capability`).

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
