# GAIA Agent (LangGraph + Gemini)

A tool-using AI agent built with [LangGraph](https://langchain-ai.github.io/langgraph/) and Google Gemini. It is my final project for the [Hugging Face AI Agents course](https://huggingface.co/learn/agents-course), and it scored **17/20 (85%)** on the GAIA level-1 validation questions used by the course leaderboard.

## What it does

The agent receives a question, decides which tools to use, and loops (ReAct style) until it can return a short, exact answer.

| Tool | Purpose |
|------|---------|
| `web_search` | Web search via DuckDuckGo (no API key needed) |
| `fetch_webpage` | Reads a page as text, keeps tables, and can jump to a keyword with `find=` |
| `wikipedia_search` | Finds the right English Wikipedia article |
| `read_file` | Reads csv, xlsx, pdf, docx, pptx, text, images and audio (images/audio go through Gemini) |
| `run_python` | Runs Python code in a subprocess for calculations and data analysis |
| `youtube_transcript` | Gets the captions of a YouTube video |

## How it works

```
START -> assistant -> (tools -> assistant)* -> END
```

The assistant node calls Gemini with the tools bound. If it asks for tools, the tools node runs them and loops back. The final line of the answer must be `FINAL ANSWER: ...`, which is parsed into a short, exact-match answer (the course leaderboard grades by exact match).

## Engineering details

Most of the work was making it run reliably on free-tier quotas and a corporate network:

- **Model pool with fallback.** `GEMINI_MODELS` takes a list of models. When a model's daily free-tier quota runs out, or it is overloaded (503), the agent switches to the next one instead of waiting.
- **Smart retries.** Rate limits and timeouts are retried with a short backoff; real errors (like a 400) fail immediately.
- **Parallel runs.** `--workers N` answers several questions at once, with log lines tagged per question.
- **Cache and resume.** Every answer is saved to `answers_cache.json`. Re-running only does what is missing, and `--redo 4,10` forces specific questions again.
- **Attachments.** The course API does not serve the attached files (it returns 404), so the runner falls back to the gated GAIA dataset on the Hugging Face Hub (needs `HF_TOKEN`).
- **Corporate TLS.** `truststore` makes Python use the operating system certificates, which fixes `CERTIFICATE_VERIFY_FAILED` behind a TLS-inspecting proxy.
- **Verbose tracing.** `-v` prints every tool call, its result, and how long each Gemini call took.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/jhonnybasso/gaia-agent-langgraph.git
cd gaia-agent-langgraph
pip install -r requirements.txt
```

Create a `.env` file (it is git-ignored):

```
GOOGLE_API_KEY=your_gemini_api_key
HF_TOKEN=your_hugging_face_token        # only needed to download GAIA attachments
GEMINI_MODELS=model-a,model-b,model-c   # optional, tried in order
```

## Usage

```bash
# Ask a single question
python agent.py "What is the capital of Australia?"

# Run one course question and watch every step
python run_local.py --limit 1 -v

# Run all 20 questions, 2 at a time (nothing is submitted)
python run_local.py --workers 2

# Redo specific questions, ignoring the cache
python run_local.py --redo 4,10

# Submit the answers to the course scoring API
python run_local.py --submit --username JhonnyBasso --agent-code https://github.com/jhonnybasso/gaia-agent-langgraph
```

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `GOOGLE_API_KEY` | required | Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Single model to use |
| `GEMINI_MODELS` | unset | Comma-separated list; overrides `GEMINI_MODEL` and enables fallback |
| `GEMINI_THINKING` | unset | `minimal`, `low`, `medium`, `high` or a token budget, to trade accuracy for speed |
| `MAX_TOOL_CALLS` | `10` | Tool calls per question before the agent must answer |
| `HF_TOKEN` | unset | Used to fetch attachments from the gated GAIA dataset |

Model names change often. You can list the models your key can use with the Gemini API (`GET /v1beta/models`).

## Notes

- The free Gemini tier has small per-model daily limits and models are sometimes overloaded. That is why the model pool exists. Results can vary between runs.
- This repository does **not** include GAIA data, answers or attachments. To download attachments you must accept the terms of the [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA), which ask you not to republish it. Do not commit `results.json` or `answers_cache.json`.
- `run_python` executes model-written code on your machine. That is fine for a course project, but do not use it as-is in production.
- Never commit your `.env`.

## Project structure

```
agent.py           LangGraph agent, tools, model pool and retry logic
run_local.py       Runs the questions locally, caches answers, optionally submits
app.py             (optional) Gradio runner, based on the course's Space template
requirements.txt   Dependencies
```

## Acknowledgements

- [Hugging Face AI Agents course](https://huggingface.co/learn/agents-course)
- [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA)
