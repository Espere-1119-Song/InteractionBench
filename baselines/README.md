# Streaming video systems

Runners for systems that are not driven through the chat interface of `ibench run`.
Each runner feeds the stream to the system through the system's own interface, records
when the system speaks, and writes the same `preds.jsonl` as `ibench run`. Scoring is
the same for every system:

```bash
ibench eval <out>/preds.jsonl --mcq-key --judge hf:Qwen/Qwen3-14B --out <out>/eval
```

| Directory | System | Interface used | Paper run |
|---|---|---|---|
| `joyai/` | JoyAI-VL-Interaction | streaming service with its controller; also with other kernels behind the same controller | `joyai_streaming_4fps_mcqv4` and the kernel swaps |
| `mmduet2/` | MMDuet2 | native proactive streaming | `mmduet2-3b_streaming_4fps_mcq_MERGED` |
| `videollm_online/` | VideoLLM-online | native streaming with its trigger threshold | `videollm-online-8b_streaming_8fps_mcq` |
| `flash_vstream/` | Flash-VStream | streaming encoder, polled once per second | `fvstream-7b_polling_iv1_8fps_mcq_lenientparse` |
| `videochat3/` | VideoChat3 | native streaming rounds | `videochat3-4b_streaming_iv1_mcq` |
| `moss/` | MOSS-Video | native real-time generation | `moss-video-preview_streaming_rt_mcq` |
| `livecc/` | LiveCC | streaming commentary | `livecc-7b_streaming_iv1_mcq` |
| `dispider/` | Dispider | offline temporal grounding | `dispider_offline_mcq` |
| `vispeak/` | ViSpeak | native proactive streaming | `vispeak-s3_streaming_native_mcq` |

Each directory has a `README.md` with the upstream repository, the environment, the
command of the paper run and the item set that run used.

## Common points

- Every system needs its own Python environment. The version pins differ and several
  conflict with each other.
- The runners import `interactionbench` for the data loader and the question rendering.
  Install this repository into each environment (`pip install -e .`) or set
  `PYTHONPATH` to the repository root.
- Upstream repositories and checkpoints are located through arguments and environment
  variables. The defaults point to `external/<name>` below the working directory.
- `--mcq <data>/mcq/mcq_options_v4.jsonl` renders the multiple-choice items with the
  options of the release. `--items <file>` restricts a run to an item list.
- Runs are resumable: items already in `preds.jsonl` are skipped.
- Runners that decode with `decord` cannot read AV1. Build H.264 proxies with
  `python scripts/prepare_data.py h264` and pass `--video-dir <data>/videos_h264`.

## Runs under an earlier protocol

The paper runs of VideoChat3, MOSS-Video, LiveCC and Dispider cover the 218-item subset
(`benchmark/splits/frozen218.txt`) and were scored with an earlier multiple-choice file
and an earlier metric configuration. Their stored scores cannot be recomputed with the
files of this release. The READMEs of these systems give the details. Running these
systems on the full set with the current files gives scores that are comparable with
the other systems and differ from the stored ones.

## Add a system

Copy the runner that is closest to your system, or start from
[examples/write_predictions.py](../examples/write_predictions.py). The requirements for
a comparable result are listed in [docs/ADD_A_MODEL.md](../docs/ADD_A_MODEL.md),
section 5.
