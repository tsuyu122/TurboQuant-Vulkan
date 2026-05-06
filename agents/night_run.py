#!/usr/bin/env python3
"""
Helper para preparar e iniciar a execução noturna do sistema de agentes.

Uso:
    py agents/night_run.py          # Testa, faz backup e prepara task.md
    py agents/night_run.py --run    # Também inicia autonomous_v3.py --watch
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"
TASK_FILE = AGENTS_DIR / "task.md"
NIGHT_PROMPT_FILE = AGENTS_DIR / "night_prompt.md"
BACKUP_SCRIPT = BASE_DIR / "backup_workspace.py"
AUTONOMOUS_SCRIPT = AGENTS_DIR / "autonomous_v3.py"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def run_command(cmd, timeout=600):
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def prepare_task():
    prompt = read_text(NIGHT_PROMPT_FILE).strip()
    if not prompt:
        raise RuntimeError(f"Arquivo de prompt noturno não encontrado: {NIGHT_PROMPT_FILE}")
    write_text(TASK_FILE, prompt)
    print(f"[OK] task.md atualizado com o prompt noturno ({TASK_FILE})")


def run_tests():
    print("[INFO] Executando testes do sistema...")
    code, out, err = run_command([sys.executable, str(AUTONOMOUS_SCRIPT), "--test"], timeout=300)
    print(out)
    if err:
        print(err)
    if code != 0:
        raise RuntimeError("Teste do sistema falhou")
    print("[OK] Testes concluídos com sucesso")


def create_backup():
    print("[INFO] Criando backup do workspace...")
    code, out, err = run_command([sys.executable, str(BACKUP_SCRIPT)], timeout=900)
    print(out)
    if err:
        print(err)
    if code != 0:
        raise RuntimeError("Backup falhou")
    print("[OK] Backup concluído")


def start_watch(interval: int = 120):
    print("[INFO] Iniciando execução em modo watch...")
    subprocess.run([sys.executable, str(AUTONOMOUS_SCRIPT), "--watch", "--interval", str(interval)], cwd=BASE_DIR)


def main():
    parser = argparse.ArgumentParser(description="Night run helper para TurboQuant-Vulkan")
    parser.add_argument("--run", action="store_true", help="Inicia autonomous_v3.py em modo watch após testes e backup")
    parser.add_argument("--interval", type=int, default=120, help="Intervalo de watch em segundos")
    args = parser.parse_args()

    prepare_task()
    run_tests()
    create_backup()

    print("\n[PRONTO] Preparação noturna concluída.")
    print("Use o comando abaixo para rodar o sistema durante a noite:")
    print(f"    py agents/autonomous_v3.py --watch --interval {args.interval}")

    if args.run:
        start_watch(args.interval)


if __name__ == "__main__":
    main()
