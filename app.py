import json
import os
import tempfile
import time

import gradio as gr
import pandas as pd
import requests

from agent import LangGraphAgent

# --- Constantes ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE_PATH = "answers_cache.json"  # sobrevive a erros durante a execução (some se o Space reiniciar)
QUESTION_DELAY = float(os.getenv("QUESTION_DELAY", "3"))  # pausa entre perguntas p/ aliviar o limite de taxa


def load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def download_task_file(api_url: str, task_id: str, file_name: str | None, dest_dir: str) -> str | None:
    """Baixa o arquivo anexo de uma tarefa (se houver) e devolve o caminho local."""
    if not file_name:
        return None
    try:
        r = requests.get(f"{api_url}/files/{task_id}", timeout=30)
        r.raise_for_status()
        path = os.path.join(dest_dir, file_name)
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:  # noqa: BLE001
        print(f"Could not download file for task {task_id}: {e}")
        return None


def run_and_submit_all(use_cache: bool, profile: gr.OAuthProfile | None):
    """Busca as perguntas, roda o agente em todas, envia as respostas e mostra o resultado."""
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instancia o agente
    try:
        agent = LangGraphAgent()
    except Exception as e:  # noqa: BLE001
        return f"Error initializing agent: {e}", None
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    # 2. Busca as perguntas
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
    except Exception as e:  # noqa: BLE001
        return f"Error fetching questions: {e}", None

    # 3. Roda o agente
    cache = load_cache() if use_cache else {}
    tmp_dir = tempfile.mkdtemp()
    results_log, answers_payload = [], []
    for i, item in enumerate(questions_data, start=1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        cached = cache.get(task_id)
        if cached:
            submitted_answer = cached
            print(f"[{i}/{len(questions_data)}] {task_id}: using cached answer")
        else:
            try:
                file_path = download_task_file(api_url, task_id, item.get("file_name"), tmp_dir)
                print(f"[{i}/{len(questions_data)}] {task_id}: running agent (file: {file_path})")
                submitted_answer = agent(question_text, file_path)
                if submitted_answer:
                    cache[task_id] = submitted_answer
                    save_cache(cache)
            except Exception as e:  # noqa: BLE001
                print(f"Error running agent on task {task_id}: {e}")
                results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})
                continue
            time.sleep(QUESTION_DELAY)
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Envia
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    results_df = pd.DataFrame(results_log)
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        detail = f"Server responded with status {e.response.status_code}."
        try:
            detail += f" Detail: {e.response.json().get('detail', e.response.text)}"
        except Exception:  # noqa: BLE001
            detail += f" Response: {e.response.text[:500]}"
        return f"Submission Failed: {detail}", results_df
    except requests.exceptions.Timeout:
        return "Submission Failed: The request timed out.", results_df
    except Exception as e:  # noqa: BLE001
        return f"Submission Failed: {e}", results_df


def test_random_question():
    """Roda o agente em UMA pergunta aleatória, sem enviar nada. Bom para depurar."""
    try:
        item = requests.get(f"{DEFAULT_API_URL}/random-question", timeout=15).json()
        file_path = download_task_file(
            DEFAULT_API_URL, item["task_id"], item.get("file_name"), tempfile.mkdtemp()
        )
        answer = LangGraphAgent()(item["question"], file_path)
        return item["question"], answer
    except Exception as e:  # noqa: BLE001
        return "Error", str(e)


# --- Interface Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner (LangGraph + Gemini)")
    gr.Markdown(
        """
        1. Faça login com sua conta Hugging Face (botão abaixo).
        2. (Opcional) Teste o agente em uma pergunta aleatória.
        3. Clique em **Run Evaluation & Submit All Answers**. Pode levar vários minutos.

        As respostas ficam em cache enquanto o Space estiver rodando: se algo falhar no meio,
        é só rodar de novo com o cache ligado e ele continua de onde parou.
        """
    )
    gr.LoginButton()

    with gr.Accordion("Testar uma pergunta aleatória (não envia nada)", open=False):
        test_button = gr.Button("Testar")
        test_question = gr.Textbox(label="Pergunta", lines=4, interactive=False)
        test_answer = gr.Textbox(label="Resposta do agente", interactive=False)
        test_button.click(fn=test_random_question, outputs=[test_question, test_answer])

    use_cache = gr.Checkbox(value=True, label="Reusar respostas em cache")
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, inputs=[use_cache], outputs=[status_output, results_table])

if __name__ == "__main__":
    print("Launching Gradio Interface...")
    demo.launch(debug=True, share=False)
