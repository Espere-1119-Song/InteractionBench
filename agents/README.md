# Agent harness evaluation

This directory evaluates tool-using agents on InteractionBench. An agent is a command
line program that receives one text prompt, may call tools to inspect a video file, and
prints one text answer. Four agents are covered:

| Harness | Agent | Tools |
|---|---|---|
| `claude` | Claude Code CLI (`claude -p`) | seven tools of the Qwen-MM-Plugins `core` MCP server |
| `claude --no-mcp` | Claude Code CLI (`claude -p`) | built-in tools `Bash, Read, Glob, Grep, Write, Edit` only |
| `gemini` | Gemini CLI (`gemini -p`) | Qwen-MM-Plugins `core` MCP server |
| `openai` | `agents/openai_agent.py`, a tool-calling loop on an OpenAI-compatible endpoint | tools of the Qwen-MM-Plugins `core` MCP server that are in its allow-list |

The toolbox is [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins). Its `core`
capability reads images and video with ffmpeg on the local machine and needs no API key.

## Files

| File | Content |
|---|---|
| `run_polling.py` | setting 1, online polling |
| `run_grounding.py` | setting 2, offline temporal grounding |
| `run_mcq.py` | setting 3, offline multiple-choice accuracy |
| `openai_agent.py` | the agent of the `openai` harness |
| `cost_proxy.py` | local metering proxy with a spending limit, for the `openai` harness |
| `common.py` | job order and setup check, shared by the three runners |
| `config/mcp.json` | MCP server configuration of the Claude Code CLI |
| `config/gemini_settings.json` | MCP server configuration and excluded tools of the Gemini CLI |
| `config/env.example` | list of the environment variables, without values |

## Settings

| Setting | Script | Items | What the agent receives | What is measured |
|---|---|---|---|---|
| 1. Online polling | `run_polling.py` | `benchmark/splits/subset103.txt` (103) | every `--interval` seconds of stream time: the video trimmed to the current time, its own last three utterances, the question; it answers SPEAK or WAIT | timing accuracy, silence compliance and answer accuracy under the polling protocol |
| 2. Offline grounding | `run_grounding.py` | `benchmark/splits/all1060.txt` (1060) | the full video and the question; it lists timed responses `[t=12.3] <utterance>` | the same three scores, computed from the claimed times; not real time |
| 3. Offline multiple choice | `run_mcq.py` | `benchmark/splits/mcq688.txt` (688) | the video trimmed to the answer cutoff, the stem and the options; it answers with one letter | answer accuracy only |

Setting 1 starts one complete agent session per tick. With `--interval 1` the 103 items
have 11,696 ticks. For this reason setting 1 runs on the 103-item subset.

Setting 2 lets the agent see video content after each answer time. Its scores are not
comparable with the scores of setting 1 and are reported separately.

Multiple-choice items are asked as multiple choice in all three settings. Settings 1 and
2 ask the remaining items as free-form text.

## Prerequisites

1. The package and the data (see the repository README and `docs/DATA.md`):

   ```bash
   pip install -e .
   python scripts/prepare_data.py --data data/interactionbench download
   python scripts/prepare_data.py --data data/interactionbench h264
   ```

   The runners read `<data>/results/*/*/annotation.json`,
   `<data>/mcq/mcq_options_v4.jsonl`, `<data>/mcq/mcq_key_v4.jsonl` (setting 3 only)
   and the H.264 videos `<data>/videos_h264/<video_id>.mp4`.

2. `ffmpeg` with libx264, and [`uv`](https://docs.astral.sh/uv/) (provides `uvx`).

3. Qwen-MM-Plugins:

   ```bash
   git clone https://github.com/QwenLM/Qwen-MM-Plugins external/Qwen-MM-Plugins
   export QWEN_MM_PLUGINS_ROOT=$PWD/external/Qwen-MM-Plugins
   ```

   The MCP server is started by the agent as
   `uvx --from "$QWEN_MM_PLUGINS_ROOT[core]" qwen-mm-plugins-core`. The first start
   downloads the Python dependencies of the toolbox. The checkout used with these
   scripts was version 1.0.8 (commit `ab339d2`).

4. The command line tool of the harness:

   | Harness | Tool | Key |
   |---|---|---|
   | `claude` | Claude Code CLI, command `claude` | `ANTHROPIC_API_KEY` |
   | `gemini` | Gemini CLI, command `gemini` | `GEMINI_API_KEY` |
   | `openai` | none (Python standard library) | `UPSTREAM_API_KEY`, held by `cost_proxy.py` |

   Use an API key for the Claude Code CLI when `--jobs` is larger than 1. Parallel
   sessions that share one interactive login can invalidate each other when the login
   token is refreshed.

5. An agent home directory. The agent sessions run in this directory. Create it outside
   the repository and outside the data directory:

   ```bash
   export IBENCH_AGENT_HOME=/path/to/agent_home
   mkdir -p $IBENCH_AGENT_HOME/.gemini
   cp agents/config/mcp.json             $IBENCH_AGENT_HOME/.mcp.json
   cp agents/config/gemini_settings.json $IBENCH_AGENT_HOME/.gemini/settings.json
   ```

   Use one agent home per run. The runners write clips to
   `<agent-home>/videos_tick/` (setting 1) and `<agent-home>/videos_trim/` (setting 3).

6. Environment variables. `config/env.example` lists all of them.

   | Variable | Used by | Meaning |
   |---|---|---|
   | `QWEN_MM_PLUGINS_ROOT` | all harnesses with the toolbox | path of the Qwen-MM-Plugins checkout |
   | `IBENCH_AGENT_HOME` | runners | default of `--agent-home` |
   | `IBENCH_DATA` | runners | default of `--data` |
   | `IBENCH_VIDEOS_H264` | runners | default of `--video-dir` |
   | `ANTHROPIC_API_KEY` | Claude Code CLI | API key |
   | `GEMINI_API_KEY` | Gemini CLI | API key |
   | `OAI_MODEL` | `openai_agent.py` | model name on the endpoint (required) |
   | `OAI_BASE_URL` | `openai_agent.py` | endpoint, default `http://127.0.0.1:8199/v1` (the cost proxy) |
   | `OAI_API_KEY` | `openai_agent.py` | default `proxy`; set it only to call an endpoint without the proxy |
   | `OAI_MAX_TURNS` | `openai_agent.py` | maximum number of model calls per session, default 12 |
   | `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY` | `cost_proxy.py` | real endpoint and real key |
   | `GPT_PRICE_IN`, `GPT_PRICE_CACHED`, `GPT_PRICE_OUT` | `cost_proxy.py` | list prices in USD per 1M tokens |
   | `GPT_BUDGET_USD` | `cost_proxy.py` | spending limit |

   The runners stop before the first session when `.mcp.json`,
   `.gemini/settings.json`, `QWEN_MM_PLUGINS_ROOT` or `OAI_MODEL` is missing for the
   chosen harness.

## Commands

Run all commands from the repository root. The defaults are `--data
data/interactionbench`, `--video-dir <data>/videos_h264`, `--agent-home
$IBENCH_AGENT_HOME`, and the multiple-choice files under `<data>/mcq/`.

### Setting 1: online polling

```bash
# claude_qwenmm_polling_iv1 and claude_qwenmm_polling_iv1_v2 (same settings, two runs)
python agents/run_polling.py --harness claude --jobs 6 --timeout 240 --model sonnet \
    --items benchmark/splits/subset103.txt --out results/runs/claude_qwenmm_polling_iv1_v2

# claude_builtin_polling_iv1
python agents/run_polling.py --harness claude --jobs 6 --timeout 240 --model sonnet \
    --no-mcp --item-cwd \
    --items benchmark/splits/subset103.txt --out results/runs/claude_builtin_polling_iv1

# gemini_qwenmm_polling_iv1
python agents/run_polling.py --harness gemini --jobs 12

# openai_qwenmm_polling_iv1
python agents/cost_proxy.py --port 8199 --line agent-poll &
python agents/run_polling.py --harness openai --jobs 6
```

Values that are not on the command line are the defaults: `--interval 1.0`,
`--a-window 10.0`, `--timeout 240`, `--items benchmark/splits/subset103.txt`.

### Setting 2: offline temporal grounding

```bash
# claude_qwenmm_grounding
python agents/run_grounding.py --harness claude --jobs 4 --timeout 1800

# openai_qwenmm_grounding
python agents/cost_proxy.py --port 8199 --line agent-ground &
python agents/run_grounding.py --harness openai --jobs 12 --timeout 900
```

An item whose output has no parsable line and no `NO_RESPONSE` is not written to
`preds.jsonl`. Repeat the command until `preds.jsonl` has one line per item. The
claude harness uses the model alias `sonnet`; the script has no `--model` argument.

### Setting 3: offline multiple choice

```bash
# openai_qwenmm_mcqv4
python agents/cost_proxy.py --port 8199 --line agent-mcq &
python agents/run_mcq.py --harness openai --jobs 12 --timeout 420

# claude_qwenmm_mcqv4, gemini_qwenmm_mcqv4: the same command with another harness
python agents/run_mcq.py --harness claude --jobs 4
python agents/run_mcq.py --harness gemini --jobs 4
```

The claude harness uses the model alias `sonnet`; the script has no `--model` argument.

### Run directories of the paper

`--jobs` sets the number of items that run in parallel. It changes the run time, not the
prompts.

| Run directory | Setting | Harness | Arguments | Item list |
|---|---|---|---|---|
| `claude_qwenmm_polling_iv1` | 1 | `claude` | `--jobs 6 --timeout 240`, model `sonnet` | `subset103.txt` |
| `claude_qwenmm_polling_iv1_v2` | 1 | `claude` | `--jobs 6 --timeout 240 --model sonnet` | `subset103.txt` |
| `claude_builtin_polling_iv1` | 1 | `claude` | `--jobs 6 --timeout 240 --model sonnet --no-mcp --item-cwd` | `subset103.txt` |
| `gemini_qwenmm_polling_iv1` | 1 | `gemini` | `--jobs 12` (timeout 240) | `subset103.txt` |
| `openai_qwenmm_polling_iv1` | 1 | `openai` | `--jobs 6` (timeout 240) | `subset103.txt` |
| `claude_qwenmm_grounding` | 2 | `claude` | `--jobs 4 --timeout 1800` | `all1060.txt` |
| `openai_qwenmm_grounding` | 2 | `openai` | `--jobs 12 --timeout 900`; 17 items come from a later pass, see "Known limits" | `all1060.txt` |
| `openai_qwenmm_mcqv4` | 3 | `openai` | `--jobs 12 --timeout 420` | `mcq688.txt` |
| `claude_qwenmm_mcqv4`, `gemini_qwenmm_mcqv4` | 3 | `claude`, `gemini` | default output names of `run_mcq.py` | `mcq688.txt` |

The arguments in this table are the values recorded in the `config.json` of each run.
`config.json` is written again at every invocation, so it records the last invocation
of a run that was resumed.

## Output

Settings 1 and 2 write, below `results/runs/<run>/`:

- `preds.jsonl`: one line per item,
  `{"video_id", "item_index", "model", "run", "emissions": [{"t", "content", "latency_s"}], "n_polls", "poll_latencies"}`.
  Setting 1 adds `poll_latency_mean`. Setting 2 adds `claimed_t` to the emission of an
  item of time type A. A rerun skips the items that are already in the file.
- `raw/<item>.json`: setting 1, every tick with time, decision, latency and the last
  150 characters of the agent output; setting 2, the last 600 characters of the agent
  output and the wall time.
- `config.json`: the configuration of the last invocation.

Setting 3 writes `results.jsonl` with one line per item:
`{"item_id", "gt", "cutoff_s", "letter", "raw_tail", "err", "wall_s", "correct"}`.

## Scoring

Settings 1 and 2 are scored with the evaluator of the package:

```bash
ibench eval results/runs/claude_qwenmm_polling_iv1_v2/preds.jsonl \
    --data data/interactionbench --mcq-key \
    --items benchmark/splits/subset103.txt \
    --judge hf:Qwen/Qwen3-14B \
    --out results/runs/claude_qwenmm_polling_iv1_v2/eval

ibench eval results/runs/claude_qwenmm_grounding/preds.jsonl \
    --data data/interactionbench --mcq-key \
    --items benchmark/splits/all1060.txt \
    --judge hf:Qwen/Qwen3-14B \
    --out results/runs/claude_qwenmm_grounding/eval
```

Pass `--items` for every run on a subset. Without it the evaluator scores all 1060
items and counts each item without a prediction as silent. Without `--judge` the
free-form answers are scored lexically. The reference scores of the paper runs are in
`configs/paper_runs.json`.

Setting 3 is scored by `run_mcq.py`. It prints the accuracy over the items that have an
answer letter. For the accuracy over all items, divide the number of rows with
`"correct": true` by the number of items (688).

## Behaviour that affects the results

| Item | Value |
|---|---|
| Tick schedule, time type A | from `question_time_s` to `min(duration, question_time_s + a_window)`, step `interval` |
| Tick schedule, other items | from `interval` to `duration`, step `interval` |
| Missing duration | 60 s |
| Clip of a tick | first `max(t, 1.0)` seconds; libx264, preset veryfast, crf 27, no audio |
| Clip of setting 3 | first `cutoff` seconds; libx264, preset veryfast, crf 26, no audio |
| Own utterances in the prompt | the last three emissions of the item |
| Decision parsing, setting 1 | first match of `DECISION:\s*(SPEAK\|WAIT)`, case-insensitive; the response is the first line after `RESPONSE:` |
| Dropped responses, setting 1 | empty response; response that contains the text `if SPEAK, one short line` (copy of the format template) |
| Timeout, setting 1 | the session and its child processes are killed; the tick is recorded as WAIT |
| Line parsing, setting 2 | `[t=<number>] <text>` per line; lines that contain `the exact thing you would say` or `<your answer>` are dropped |
| Time type A, setting 2 | the first parsed line is kept; the emission time is `question_time_s` |
| Negative items, setting 2 | output with `NO_RESPONSE` gives an empty emission list |
| Letter parsing, setting 3 | last stand-alone option letter in the final 200 characters of the output |
| Retries, setting 3 | at most 2 further passes over the items without a letter |
| Retries, `openai_agent.py` | HTTP 429, 500, 502, 503: up to 5 retries, waits of 8, 16, 32, 64, 128 s |
| Turn limit, `openai_agent.py` | 12 model calls per session, 1500 completion tokens per call |
| Question text, settings 1 and 2 | the annotation question followed by the options; the `stem` field of the options file is not used |
| Question text, setting 3 | the `stem` field of the options file followed by the options |
| Job order | items sorted by id, then taken in turn from each capability |

## Answer-leak guards

This section states what the code does. It separates the guards that the scripts apply
from the guards that depend on the setup of the agent home.

### Guards applied by the scripts

1. **Video content after the current time (setting 1).** The agent receives the path of
   a clip that holds the first `max(t, 1.0)` seconds of the video. The clip is written
   to `<agent-home>/videos_tick/<item>_<t>.mp4` before the session and deleted after
   the session. The prompt does not contain the path of the full video.
2. **Video content after the answer (setting 3).** The agent receives a clip that ends
   at the answer cutoff. The clips stay in `<agent-home>/videos_trim/` after the run.
3. **Web and sub-agent tools, harness `claude --no-mcp`.** The command passes
   `--tools Bash,Read,Glob,Grep,Write,Edit`. This limits the built-in tools of the
   session to these six, so the web fetch tool, the web search tool and the sub-agent
   tool are not available. The command passes `--strict-mcp-config` without
   `--mcp-config`, so no MCP server is loaded. The `Bash` tool is available. The scripts
   do not limit what a shell command can read or which network address it can reach.
4. **Tools of the harness `claude` with the toolbox.** The command passes
   `--allowedTools` with seven tool names: `read_video`, `read_image`, `media_info`,
   `crop`, `draw_bbox`, `visualize`, `save_view` of the server `qwen-mm-plugins-core`.
   `--allowedTools` is a list of tools that run without a permission request. The
   command does not pass `--tools` or `--disallowedTools`, so it does not remove the
   other tools from the session. In a non-interactive session (`-p`) nobody can grant a
   permission request, so a tool that requires permission and is not on the list is
   refused. Built-in tools that require no permission stay usable.
5. **Tools of the harness `gemini`.** The command is `gemini -p PROMPT --yolo`. `--yolo`
   approves every tool call. The tool restriction is the `excludeTools` list in
   `<agent-home>/.gemini/settings.json`: `run_shell_command`, `web_fetch`,
   `google_web_search`, `write_file`, `replace`, `save_memory`. The runner stops when
   this file is missing. It does not check the content of the file.
6. **Tools of the harness `openai`.** The model is offered the tools of the MCP server
   whose name is in the allow-list of `openai_agent.py`: `read_video`, `read_image`,
   `media_info`, `crop_image`, `save_view`. The loop has no shell tool and no web tool.
   Version 1.0.8 of the toolbox names its crop tool `crop`, so `crop_image` matches no
   tool and the model is offered four tools.
7. **Working directory per item (`--item-cwd`, setting 1 only).** Without this
   argument every session runs in the agent home, and a file that an agent writes there
   stays visible to all later sessions. With this argument the sessions of an item run
   in `<agent-home>/work/<item>/`. The directory is created empty before the first tick
   of the item and deleted with its content after the last tick. The clip of a tick is
   not inside this directory; it is in `<agent-home>/videos_tick/`. That directory holds
   at any time the current clips of all items in progress (at most `--jobs` clips).
   Files that an agent writes to its working directory at one tick stay visible at the
   later ticks of the same item. Of the paper runs, only `claude_builtin_polling_iv1`
   used `--item-cwd`.
8. **Answer key.** `run_polling.py` and `run_grounding.py` do not open the answer key.
   `run_mcq.py` reads the correct letter from `--mcq-key` and uses it only to fill the
   fields `gt` and `correct`. No prompt contains an answer.

### Guards that depend on the agent home

- The agent home of the paper runs was a directory outside the data directory. It held
  the configuration files and the clips written by the runners.
- The agent home of `claude_builtin_polling_iv1` also held the file
  `.claude/settings.json` with a deny list of this form. Replace the paths with the
  paths of your installation:

  ```json
  {
    "permissions": {
      "deny": [
        "WebFetch", "WebSearch", "Agent", "Task",
        "Read(/path/to/data/**)",
        "Read(/path/to/results/**)",
        "Bash(*annotation.json*)", "Bash(*mcq_key*)"
      ]
    }
  }
  ```

  The sessions of that run had the working directory `<agent-home>/work/<item>/`. The
  scripts do not check that the CLI applied this file.

### Not guarded

- The tools `read_video`, `read_image` and `visualize` accept any readable path. The
  guards above do not prevent an agent from opening a file outside the agent home when
  it has the path.
- Without `--item-cwd`, files that an agent writes to the agent home (for example
  frames saved with `save_view`) are not deleted. A later session can read them. This
  includes sessions of another item of the same video at an earlier stream time.
- Setting 2 gives the full video to the agent by design.

## Known limits

- **Number of sessions and run time.** Setting 1 with `--interval 1` on the 103 items
  needs 11,696 sessions. The mean session time was 40.3 s in
  `claude_qwenmm_polling_iv1_v2` and 48.0 s in `claude_builtin_polling_iv1`, which gives
  131 and 156 hours of session time. The ticks of one item run in sequence, so the wall
  time of one item is the number of its ticks times the session time. Setting 2 needed
  175.6 s per item on average in `claude_qwenmm_grounding` (maximum 1674 s) and 175.7 s
  in `openai_qwenmm_grounding`. Setting 3 needed 41.9 s per item on average in
  `openai_qwenmm_mcqv4`.
- **Cost.** The cost of a run is proportional to the number of sessions. Each session
  contains several model calls with images. The scripts record no cost for the
  harnesses `claude` and `gemini`. For the harness `openai`, `cost_proxy.py` records the
  token counts of every response and a price computed from the prices in the
  environment. Set `GPT_BUDGET_USD` before a run.
- **Latency and token spend were not recorded for every run.** The files `preds.jsonl`
  of `claude_qwenmm_polling_iv1`, `gemini_qwenmm_polling_iv1` and
  `openai_qwenmm_polling_iv1` hold no per-poll latency. `claude_qwenmm_polling_iv1_v2`
  and `claude_builtin_polling_iv1` hold the latency of every poll. The runs of setting
  2 hold one wall time per item. Token counts exist only for the harness `openai`, in
  the ledger of the cost proxy. The scripts in this directory record the latency of
  every poll for all harnesses.
- **Latency is not part of any score.** The recorded latency is the wall time of a
  complete agent session. The emission time is the tick time.
- **Model versions.** `--model sonnet` is an alias that the Claude Code CLI resolves at
  run time. The harness `gemini` uses the default model of the Gemini CLI. The harness
  `openai` uses `OAI_MODEL`. The versions of the two CLI tools were not recorded. The
  field `agent_model` in the `config.json` of settings 2 and 3 is `gemini CLI default`
  for every harness other than `claude`, including `openai`. The system names of the
  paper runs are in `configs/paper_runs.json`.
- **Run-to-run variation.** Agent sessions are not deterministic.
  `claude_qwenmm_polling_iv1` and `claude_qwenmm_polling_iv1_v2` have the same
  arguments and were run at different dates; their reference total scores are 39.6 and
  56.0.
- **`openai_qwenmm_grounding`.** 1043 of the 1060 items come from `run_grounding.py`;
  `--jobs 12 --timeout 900` are the arguments of its last recorded invocation. The
  other 17 items come from a later pass with a limit of one hour per item, 4 parallel
  items, a request rate limit and a limit on the number of images. The code of that
  pass is not part of this directory.
- **Timeouts and failed sessions.** A session that exceeds `--timeout`, or that ends
  without output (for example after an authentication error), gives WAIT in setting 1,
  no prediction line in setting 2, and no letter in setting 3. In setting 1 the item
  continues with the next tick, and the item is written to `preds.jsonl` also when
  every session failed. Check `raw/<item>.json` for ticks with an empty `raw` field.
- **Spending limit.** The proxy reads the total before it forwards a request. Requests
  in progress when the limit is reached are completed and charged. After the limit,
  setting 1 records every remaining tick as WAIT and still writes the item to
  `preds.jsonl`. Remove these items from `preds.jsonl` before the run is resumed with a
  higher limit.
