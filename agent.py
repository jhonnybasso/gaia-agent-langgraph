"""Agente LangGraph (Gemini) para o projeto final do curso de agentes da Hugging Face.

Fluxo ReAct: assistant -> (tools -> assistant)* -> resposta "FINAL ANSWER: ...".
Todas as ferramentas funcionam SEM chaves extras: só a GOOGLE_API_KEY (Gemini) é necessária.
"""
import base64
import contextvars
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from pathlib import Path

# Redes corporativas costumam interceptar o HTTPS com um certificado próprio. O Windows confia nele,
# mas o Python por padrão não. O truststore faz o Python usar o repositório de certificados do sistema.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # sem o pacote, segue com os certificados padrão do Python
    pass

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
# Os modelos Gemini 3.x ignoram "temperature" e a biblioteca avisa a cada chamada: silencia o aviso repetido.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

# Troque o modelo sem mexer no código: GEMINI_MODEL=gemini-2.5-flash, por exemplo.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
# Depois desse número de chamadas de ferramenta, o agente é forçado a responder.
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "10"))
HEADERS = {"User-Agent": "gaia-course-agent/1.0 (Hugging Face agents course project)"}
# Mostra cada passo do agente (ferramentas chamadas e resultados). run_local.py liga com --verbose.
VERBOSE = os.getenv("AGENT_VERBOSE") == "1"
# Etiqueta da pergunta atual (ex.: "q3"); permite distinguir os logs quando várias rodam em paralelo.
LOG_TAG = contextvars.ContextVar("log_tag", default="")


def _log(msg: str) -> None:
    tag = LOG_TAG.get()
    print(f"    {'[' + tag + '] ' if tag else ''}{msg}", flush=True)


def _short(value, limit: int = 160) -> str:
    s = " ".join(str(value).split())
    return s if len(s) <= limit else s[:limit] + "..."

SYSTEM_PROMPT = """You are a careful research assistant answering questions from the GAIA benchmark.
Use the tools to find or compute the answer. Do not guess when a tool can verify the fact.

Guidelines:
- Be efficient: aim for at most 5-6 tool calls. You may issue several independent tool calls in the SAME step.
- If the question mentions an attached file, its local path is appended to the question: use read_file on it.
- For a YouTube link, try youtube_transcript.
- For facts on the web: web_search, then fetch_webpage on the best result.
  On long pages use fetch_webpage(url, find="keyword") to jump straight to the relevant passages instead of paging.
- For Wikipedia: wikipedia_search gives the intro and the page URL. For tables or lists (discographies,
  filmographies, rosters, statistics) call fetch_webpage(url, find="<table or section title>").
  Do NOT download Wikipedia through run_python. A request for the version of a page from a given year can
  usually be answered from the current page.
- For arithmetic, counting, sorting, data analysis or running code: use run_python (always print the result).
  run_python is for computation, not for downloading web pages.
- Respect every constraint in the question (units, rounding, ordering, list format, date range, etc.).
- If a search returns nothing useful, rephrase the query or try another source before giving up.

Finish with ONE last line, exactly in this format:
FINAL ANSWER: <answer>

The answer is graded by EXACT MATCH, so:
- Number: digits only, no thousands separators, no units unless the question asks for them.
- String: as few words as possible, no articles, no abbreviations (e.g. for cities), no trailing period.
- List: comma-separated, applying the rules above to each element.
- Never write anything after the FINAL ANSWER line.
"""


# ---------------------------------------------------------------- utilidades
def _clip(text: str, limit: int = 8000) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated: {len(text) - limit} more characters]"
    return text


def _text(content) -> str:
    """Gemini pode devolver str ou lista de blocos; normaliza para str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content)


def _make_llm(model: str | None = None) -> ChatGoogleGenerativeAI:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Defina o secret GOOGLE_API_KEY (Settings > Variables and secrets do Space, "
            "ou um arquivo .env local)."
        )
    os.environ["GOOGLE_API_KEY"] = key
    # timeout + poucas retentativas internas: assim os atrasos de cota aparecem no nosso log ([espera]).
    kwargs = {"model": model or MODEL_NAME, "temperature": 0, "timeout": 120, "max_retries": 2}
    # Opcional, para ganhar velocidade: GEMINI_THINKING=low (ou minimal/medium/high, ou um número de tokens).
    thinking = os.getenv("GEMINI_THINKING", "").strip().lower()
    if thinking in ("minimal", "low", "medium", "high"):
        kwargs["thinking_level"] = thinking
    elif thinking.isdigit():
        kwargs["thinking_budget"] = int(thinking)
    return ChatGoogleGenerativeAI(**kwargs)


class DailyQuotaExceeded(Exception):
    """A cota diária gratuita do modelo acabou: esperar não adianta, é preciso trocar de modelo (ou voltar amanhã)."""


class ModelOverloaded(Exception):
    """O modelo está sobrecarregado (503) ou não respondeu a tempo: vale mais tentar outro modelo do que esperar."""


def _invoke(llm, messages, attempts: int = 5):
    """Chama o LLM com retry para limites de taxa (429) e indisponibilidade (503)."""
    for i in range(attempts):
        try:
            started = time.time()
            result = llm.invoke(messages)
            if VERBOSE:
                _log(f"[gemini] resposta em {time.time() - started:.0f}s")
            return result
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            # Ex.: quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier" ou "Please retry in 8h45m".
            if "perday" in msg or "per_day" in msg or re.search(r"retry in \d+h", msg):
                raise DailyQuotaExceeded(str(e)) from e
            retryable = any(k in msg for k in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit", "499", "500", "503", "504",
                                          "cancelled", "internal", "unavailable", "overloaded",
                                          "timeout", "timed out", "deadline", "connection"))
            if not retryable or i == attempts - 1:
                raise
            overloaded = any(k in msg for k in ("499", "500", "503", "504", "cancelled", "internal", "unavailable", "overloaded", "high demand",
                                                "timeout", "timed out", "deadline", "connection"))
            if overloaded and i >= 1:  # sobrecarga persistente: o pool tenta outro modelo
                raise ModelOverloaded(str(e)) from e
            wait = 10 if overloaded else 15 * (i + 1)
            _log(f"[espera] limite/erro temporário do Gemini ({_short(e, 90)}); "
                 f"nova tentativa em {wait}s ({i + 1}/{attempts - 1})")
            time.sleep(wait)


class _ModelPool:
    """Usa os modelos de GEMINI_MODELS em ordem; quando a cota diária de um acaba, passa para o próximo."""

    def __init__(self):
        names = [m.strip() for m in os.getenv("GEMINI_MODELS", "").split(",") if m.strip()]
        self.names = names or [MODEL_NAME]
        self.dead: set[str] = set()  # cota diária esgotada
        self.cooldown: dict[str, float] = {}  # modelo -> instante até o qual está "de castigo" por sobrecarga
        self._plain: dict = {}
        self._with_tools: dict = {}
        self._lock = threading.Lock()
        self._get(self.names[0], with_tools=False)  # falha cedo se a chave não existir

    def _get(self, name: str, with_tools: bool):
        with self._lock:
            if name not in self._plain:
                self._plain[name] = _make_llm(name)
            if with_tools and name not in self._with_tools:
                self._with_tools[name] = self._plain[name].bind_tools(TOOLS)
            return self._with_tools[name] if with_tools else self._plain[name]

    def invoke(self, messages, with_tools: bool = True):
        for _ in range(8):
            alive = [n for n in self.names if n not in self.dead]
            if not alive:
                break
            ready = [n for n in alive if self.cooldown.get(n, 0) <= time.time()]
            if not ready:  # todos sobrecarregados: espera o primeiro "castigo" terminar
                wait = max(1.0, min(60.0, min(self.cooldown[n] for n in alive) - time.time()))
                _log(f"[espera] todos os modelos estão sobrecarregados; aguardando {wait:.0f}s")
                time.sleep(wait)
                continue
            for name in ready:
                try:
                    return _invoke(self._get(name, with_tools), messages)
                except DailyQuotaExceeded:
                    with self._lock:
                        first_report = name not in self.dead
                        self.dead.add(name)
                    if first_report:
                        left = [n for n in self.names if n not in self.dead]
                        _log(f"[cota] cota diária de {name} esgotada; "
                             + (f"trocando para {left[0]}" if left else "não há mais modelos na lista"))
                except ModelOverloaded:
                    self.cooldown[name] = time.time() + 90
                    _log(f"[sobrecarga] {name} está sobrecarregado; tentando outro modelo")
        if all(n in self.dead for n in self.names):
            raise DailyQuotaExceeded(
                "A cota diária gratuita acabou em todos os modelos configurados: " + ", ".join(self.names)
                + ". Tente de novo mais tarde (o cache guarda o que já foi respondido) ou adicione outros "
                  "modelos em GEMINI_MODELS."
            )
        raise ModelOverloaded("Os modelos continuam sobrecarregados. Tente de novo em alguns minutos "
                              "(o cache guarda o que já foi respondido).")


_POOL: _ModelPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> _ModelPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = _ModelPool()
        return _POOL


def extract_final_answer(text: str) -> str:
    """Extrai o que vem depois do último 'FINAL ANSWER:' e limpa a formatação."""
    matches = list(re.finditer(r"final answer\s*:\s*", text, flags=re.I))
    answer = text[matches[-1].end():] if matches else text
    answer = answer.strip()
    if not answer:
        return ""
    answer = answer.splitlines()[0].strip().strip("*` ").strip()
    answer = answer.rstrip(".").strip()
    return answer


# ---------------------------------------------------------------- ferramentas
@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo). Returns titles, URLs and snippets. Use fetch_webpage to read a result in full."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=6)
    except Exception as e:  # noqa: BLE001
        return f"Search failed: {e}"
    if not results:
        return "No results."
    return "\n\n".join(f"{r.get('title')}\n{r.get('href')}\n{r.get('body')}" for r in results)


def _pdf_text(source) -> str:
    from pypdf import PdfReader

    reader = PdfReader(source)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


@tool
def fetch_webpage(url: str, start: int = 0, find: str = "") -> str:
    """Download a web page and return its text. HTML tables are kept as rows with cells separated by ' | '.
    Also works for PDFs and plain text. Without `find`, returns 12000 characters from position `start`.
    With `find` (a word or phrase), returns only the passages around its occurrences (case-insensitive),
    which is much faster than paging through a long page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Could not fetch {url}: {e}"
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        import io

        text = _pdf_text(io.BytesIO(r.content))
    elif "html" in ctype:
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "noscript"]):
            t.decompose()
        for tr in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            tr.replace_with(soup.new_string(" | ".join(cells) + "\n"))
        text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n"))
    else:
        text = r.text
    if find:
        low, needle = text.lower(), find.lower()
        spans, pos, hits = [], 0, 0
        while len(spans) < 4 and hits < 8:
            idx = low.find(needle, pos)
            if idx == -1:
                break
            hits += 1
            lo, hi = max(0, idx - 300), min(len(text), idx + 3000)
            if spans and lo <= spans[-1][1]:
                spans[-1] = (spans[-1][0], hi)
            else:
                spans.append((lo, hi))
            pos = hi
        if not spans:
            return f"'{find}' not found in the page ({len(text)} characters). Try another keyword, or page with start=."
        return "\n\n[...]\n\n".join(text[lo:hi] for lo, hi in spans)
    window = 12000
    chunk = text[start:start + window]
    if start + window < len(text):
        chunk += f"\n...[more content: call again with start={start + window}]"
    return chunk or "(empty page)"


@tool
def wikipedia_search(query: str) -> str:
    """Search English Wikipedia. Returns the first part of the best-matching article, its URL, and other candidates.
    For tables or lists, call fetch_webpage on the URL with find=."""
    api = "https://en.wikipedia.org/w/api.php"
    try:
        hits = requests.get(
            api,
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json"},
            headers=HEADERS,
            timeout=20,
        ).json()["query"]["search"]
        if not hits:
            return "No Wikipedia results."
        title = hits[0]["title"]
        pages = requests.get(
            api,
            params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
                    "redirects": 1, "format": "json"},
            headers=HEADERS,
            timeout=20,
        ).json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
    except Exception as e:  # noqa: BLE001
        return f"Wikipedia lookup failed: {e}"
    others = "\n".join(f"- https://en.wikipedia.org/wiki/{h['title'].replace(' ', '_')}" for h in hits[1:])
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return (f"# {title}\nURL: {url}\n{_clip(extract, 5000)}\n\n"
            f"(Only the start of the article. For tables/lists use fetch_webpage on the URL with find=.)\n\n"
            f"Other candidates:\n{others}")


def _office_text(path: str, prefix: str) -> str:
    """Texto bruto de .docx (word/) ou .pptx (ppt/slides/)."""
    out = []
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".xml"))
        for n in names:
            xml = z.read(n).decode("utf-8", errors="ignore")
            out.append(re.sub(r"<[^>]+>", " ", re.sub(r"</(w:p|a:p)>", "\n", xml)))
    return re.sub(r"[ \t]+", " ", "\n".join(out))


def _ask_gemini_about_media(path: str, instruction: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    if mime.startswith("image/"):
        block = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
    else:  # áudio / vídeo
        block = {"type": "media", "mime_type": mime, "data": data}
    msg = HumanMessage(content=[{"type": "text", "text": instruction}, block])
    return _text(get_pool().invoke([msg], with_tools=False).content)


@tool
def read_file(path: str, instruction: str = "") -> str:
    """Read a local file given its path. Supports csv, xlsx/xls, pdf, docx, pptx, txt/json/py/md/xml,
    images (png/jpg/gif/webp) and audio (mp3/wav/m4a/ogg/flac). For images and audio, `instruction` says what to
    extract (default: describe the image / transcribe the audio verbatim).
    For big spreadsheets, prefer run_python with pandas to analyse them."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    ext = p.suffix.lower()
    try:
        if ext == ".csv":
            return _clip(pd.read_csv(p).to_string(), 12000)
        if ext in (".xlsx", ".xlsm", ".xls"):
            sheets = pd.read_excel(p, sheet_name=None)
            return _clip("\n\n".join(f"## Sheet: {n}\n{df.to_string()}" for n, df in sheets.items()), 12000)
        if ext == ".pdf":
            return _clip(_pdf_text(str(p)), 12000)
        if ext == ".docx":
            return _clip(_office_text(str(p), "word/document"), 12000)
        if ext == ".pptx":
            return _clip(_office_text(str(p), "ppt/slides/slide"), 12000)
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            return _ask_gemini_about_media(
                str(p), instruction or "Describe this image in detail and transcribe any text in it."
            )
        if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"):
            return _ask_gemini_about_media(str(p), instruction or "Transcribe this audio verbatim.")
        return _clip(p.read_text(encoding="utf-8", errors="ignore"), 12000)
    except Exception as e:  # noqa: BLE001
        return f"Could not read {path}: {e}"


@tool
def run_python(code: str) -> str:
    """Execute Python 3 code in a fresh subprocess (30 s timeout) and return what it prints.
    pandas is available. Always use print() for the values you want to see."""
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "Timed out after 30 seconds."
    out = p.stdout or ""
    if p.stderr:
        out += "\n[stderr]\n" + p.stderr
    return _clip(out) or "(no output - did you forget print()?)"


@tool
def youtube_transcript(url: str) -> str:
    """Get the transcript (captions) of a YouTube video from its URL.
    Fails if the video has no captions or if YouTube blocks the request."""
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    if not m:
        return "Could not find a video id in that URL."
    vid = m.group(1)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:
            text = " ".join(s.text for s in YouTubeTranscriptApi().fetch(vid))
        except AttributeError:  # versões antigas da biblioteca
            text = " ".join(i["text"] for i in YouTubeTranscriptApi.get_transcript(vid))
    except Exception as e:  # noqa: BLE001
        return f"No transcript available: {e}"
    return _clip(text, 15000)


TOOLS = [web_search, fetch_webpage, wikipedia_search, read_file, run_python, youtube_transcript]


# ---------------------------------------------------------------- grafo
def build_graph():
    pool = get_pool()

    def assistant(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        tool_calls_so_far = sum(isinstance(m, ToolMessage) for m in state["messages"])
        if tool_calls_so_far >= MAX_TOOL_CALLS:
            messages.append(HumanMessage(
                content="Tool budget exhausted. With what you know so far, reply now with your best 'FINAL ANSWER: ...'."
            ))
            return {"messages": [pool.invoke(messages, with_tools=False)]}
        return {"messages": [pool.invoke(messages, with_tools=True)]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


def _log_message(message) -> None:
    """Resume uma mensagem do grafo em uma linha (só usa ASCII nos marcadores, por causa do console do Windows)."""
    if isinstance(message, ToolMessage):
        _log(f"[resultado] {message.name}: {_short(message.content)}")
        return
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            _log(f"[ferramenta] {call['name']}({_short(call['args'])})")
    else:
        _log(f"[modelo] {_short(_text(message.content), 200)}")


class LangGraphAgent:
    def __init__(self):
        self.pool = get_pool()
        self.graph = build_graph()

    def __call__(self, question: str, file_path: str | None = None) -> str:
        prompt = question
        if file_path:
            prompt += f"\n\n[Attached file available locally at: {file_path}]"
        messages = [HumanMessage(content=prompt)]
        # stream_mode="updates" devolve as mensagens novas de cada nó, na ordem em que acontecem.
        for update in self.graph.stream({"messages": list(messages)}, config={"recursion_limit": 60},
                                        stream_mode="updates"):
            for node_output in update.values():
                new_messages = node_output.get("messages", [])
                messages.extend(new_messages)
                if VERBOSE:
                    for m in new_messages:
                        _log_message(m)
        text = _text(messages[-1].content)
        if "final answer" not in text.lower():
            # O modelo terminou sem o formato esperado (ex.: resposta vazia): pede só a resposta final.
            messages = messages + [HumanMessage(content="Reply now with only your best 'FINAL ANSWER: ...' line.")]
            text = _text(self.pool.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages, with_tools=False).content)
        return extract_final_answer(text)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the capital of Australia?"
    print(LangGraphAgent()(q))
