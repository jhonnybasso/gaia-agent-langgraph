"""Roda o agente localmente nas perguntas do projeto final e, opcionalmente, envia as respostas.

Uso:
  python run_local.py --limit 1 -v             # testa em 1 pergunta mostrando cada passo, não envia nada
  python run_local.py --workers 3              # roda as 20 perguntas, 3 ao mesmo tempo (padrão), não envia
  python run_local.py --submit --username SEU_USUARIO_HF --agent-code https://github.com/voce/repo

Só envia ao leaderboard se você passar --submit. Requer GOOGLE_API_KEY no ambiente ou num arquivo .env.
Cada resposta pronta é salva em answers_cache.json; se interromper (Ctrl+C), rodar de novo continua de onde parou.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import agent as agent_module
from agent import LangGraphAgent

API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE_PATH = "answers_cache.json"
RESULTS_PATH = "results.json"


def load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


GAIA_FILES_URL = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/"


def download_task_file(task_id: str, file_name: str | None, dest_dir: str) -> str | None:
    """Baixa o anexo de uma pergunta (None se ela não tem anexo).

    Tenta a API do curso; se ela não tiver o arquivo (404), usa o dataset GAIA, que exige HF_TOKEN e ter aceitado
    os termos do dataset. Se não conseguir, levanta erro: responder sem o arquivo que a pergunta cita só gera
    resposta errada (e ela ficaria no cache)."""
    if not file_name:
        return None
    content, problems = None, []
    try:
        r = requests.get(f"{API_URL}/files/{task_id}", timeout=30)
        r.raise_for_status()
        content = r.content
    except Exception as e:  # noqa: BLE001
        problems.append(f"API do curso: {e}")
    if content is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            problems.append("HF_TOKEN não definido (necessário para baixar o anexo do dataset GAIA)")
        else:
            try:
                r = requests.get(GAIA_FILES_URL + file_name, headers={"Authorization": f"Bearer {token}"}, timeout=60)
                r.raise_for_status()
                content = r.content
            except Exception as e:  # noqa: BLE001
                problems.append(f"dataset GAIA: {e}")
    if content is None:
        raise RuntimeError(f"anexo '{file_name}' indisponível ({'; '.join(problems)})")
    # uma pasta por tarefa evita que arquivos de perguntas diferentes com o mesmo nome se sobrescrevam
    task_dir = os.path.join(dest_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    path = os.path.join(task_dir, file_name)
    with open(path, "wb") as f:
        f.write(content)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="rodar só as N primeiras perguntas (0 = todas)")
    ap.add_argument("--workers", type=int, default=3, help="quantas perguntas rodar ao mesmo tempo (padrão 3)")
    ap.add_argument("--no-cache", action="store_true", help="ignorar respostas já salvas em answers_cache.json")
    ap.add_argument("--redo", help="números das perguntas a refazer, como aparecem no log (ex.: 4,5,10), ignorando o cache")
    ap.add_argument("-v", "--verbose", action="store_true", help="mostrar cada passo do agente (ferramentas e resultados)")
    ap.add_argument("--submit", action="store_true", help="enviar as respostas para a API de avaliação")
    ap.add_argument("--username", help="seu usuário do Hugging Face (obrigatório com --submit)")
    ap.add_argument("--agent-code", help="URL pública do seu código (obrigatório com --submit)")
    args = ap.parse_args()
    agent_module.VERBOSE = args.verbose

    if args.submit and (not args.username or not args.agent_code):
        ap.error("--submit exige --username e --agent-code")

    questions = requests.get(f"{API_URL}/questions", timeout=20).json()
    if args.limit:
        questions = questions[: args.limit]
    total = len(questions)
    print(f"{total} pergunta(s) para rodar, {max(1, args.workers)} ao mesmo tempo.")

    agent = LangGraphAgent()
    cache = {} if args.no_cache else load_cache()
    if args.redo:
        for n in {int(x) for x in args.redo.split(",") if x.strip().isdigit()}:
            if 1 <= n <= total:
                cache.pop(questions[n - 1].get("task_id"), None)
        save_json(CACHE_PATH, cache)
    lock = threading.Lock()
    tmp_dir = tempfile.mkdtemp()

    def work(i: int, item: dict):
        task_id, question = item.get("task_id"), item.get("question")
        if not task_id or question is None:
            return None
        tag = f"q{i}"
        agent_module.LOG_TAG.set(tag)  # etiqueta os logs desta pergunta
        head = f"[{tag}/{total}]"
        print(f"\n{head} {question[:110]}{'...' if len(question) > 110 else ''}", flush=True)
        answer = cache.get(task_id)
        if answer:
            print(f"{head} (cache) -> {answer}", flush=True)
            return {"task_id": task_id, "question": question, "answer": answer}
        started = time.time()
        try:
            file_path = download_task_file(task_id, item.get("file_name"), tmp_dir)
            answer = agent(question, file_path)
        except agent_module.DailyQuotaExceeded as e:
            print(f"{head} ! {e}", flush=True)
            return {"task_id": task_id, "question": question, "answer": None, "error": "cota diária esgotada"}
        except Exception as e:  # noqa: BLE001
            print(f"{head} ! erro: {e}", flush=True)
            return {"task_id": task_id, "question": question, "answer": None, "error": str(e)}
        print(f"{head} -> {answer}  ({time.time() - started:.0f}s)", flush=True)
        if answer:
            with lock:
                cache[task_id] = answer
                save_json(CACHE_PATH, cache)
        return {"task_id": task_id, "question": question, "answer": answer}

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    started_all = time.time()
    try:
        results = [r for r in pool.map(lambda p: work(*p), enumerate(questions, start=1)) if r]
    except KeyboardInterrupt:
        print("\nInterrompido. As respostas já concluídas estão salvas em answers_cache.json.")
        pool.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    pool.shutdown()

    payload = [{"task_id": r["task_id"], "submitted_answer": r["answer"]} for r in results if r.get("answer") is not None]
    save_json(RESULTS_PATH, results)
    print(f"\n{len(payload)}/{total} respondidas em {time.time() - started_all:.0f}s. Detalhes em {RESULTS_PATH}.")
    failed = [r for r in results if r.get("error")]
    if failed:
        print(f"{len(failed)} com erro; rode de novo para tentar só essas (as outras estão no cache).")

    if not args.submit:
        print("Nada foi enviado (use --submit para enviar).")
        return 0
    if not payload:
        print("Sem respostas para enviar.")
        return 1

    body = {"username": args.username.strip(), "agent_code": args.agent_code, "answers": payload}
    try:
        r = requests.post(f"{API_URL}/submit", json=body, timeout=60)
        r.raise_for_status()
        d = r.json()
        print(f"Enviado! Nota: {d.get('score', 'N/A')}% ({d.get('correct_count', '?')}/{d.get('total_attempted', '?')})")
        print(d.get("message", ""))
        return 0
    except requests.exceptions.HTTPError as e:
        print(f"Falha no envio: HTTP {e.response.status_code} - {e.response.text[:500]}")
    except Exception as e:  # noqa: BLE001
        print(f"Falha no envio: {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
