#!/usr/bin/env python3
"""
🤖 Autonomous Agent Daemon

Loop 100% automático:
1. Gemma lê task.md + brain.md + memory.md
2. Gemma visualiza workspace
3. Gemma cria plan.md
4. Daemon executa via Copilot CLI ou localmente
5. Resultado salvo em result.md
6. Gemma avalia e atualiza memory.md
7. Loop até tarefa completa

Usage:
    py agents/daemon.py                    # Inicia o daemon
    py agents/daemon.py --once             # Roda apenas um ciclo
    py agents/daemon.py --watch            # Monitora task.md continuamente
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── Configuração ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"

BRAIN_FILE = AGENTS_DIR / "brain.md"
MEMORY_FILE = AGENTS_DIR / "memory.md"
TASK_FILE = AGENTS_DIR / "task.md"
PLAN_FILE = AGENTS_DIR / "plan.md"
RESULT_FILE = AGENTS_DIR / "result.md"
WORKSPACE_VIEW = AGENTS_DIR / "workspace_view.txt"

SERVER_URL = "http://127.0.0.1:8090"
MAX_TOKENS = 4096


# ─── Utilidades ───────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    """Log com timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "THINK": "🧠", "EXEC": "⚡"}
    icon = icons.get(level, "•")
    print(f"[{timestamp}] {icon} {msg}")


def read_file_safe(path: Path) -> str:
    """Lê arquivo com fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except:
        return ""


def write_file(path: Path, content: str):
    """Escreve arquivo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def file_hash(path: Path) -> str:
    """Hash do arquivo para detectar mudanças."""
    content = read_file_safe(path)
    return hashlib.md5(content.encode()).hexdigest()


def check_server() -> bool:
    """Verifica se llama-server está rodando."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except:
        return False


# ─── Visualização do Workspace ────────────────────────────────────────────────

def generate_workspace_view() -> str:
    """Gera visão hierárquica do workspace."""
    
    ignore_patterns = {
        ".git", "__pycache__", ".venv", "node_modules", 
        "build", "build_vulkan", ".vscode", "*.pyc",
        "*.log", "*.egg-info"
    }
    
    def should_ignore(name: str) -> bool:
        for pattern in ignore_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True
        return False
    
    lines = ["# Workspace Structure", ""]
    
    def walk_dir(path: Path, prefix: str = ""):
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        
        for i, item in enumerate(items):
            if should_ignore(item.name):
                continue
            
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                walk_dir(item, new_prefix)
            else:
                size = item.stat().st_size
                size_str = f"{size:,}b" if size < 1024 else f"{size//1024}KB"
                lines.append(f"{prefix}{connector}{item.name} ({size_str})")
    
    walk_dir(BASE_DIR)
    return "\n".join(lines[:200])  # Limita para não estourar contexto


def read_file_preview(rel_path: str, max_lines: int = 50) -> str:
    """Lê preview de um arquivo."""
    path = BASE_DIR / rel_path
    if not path.exists():
        return f"[Arquivo não encontrado: {rel_path}]"
    
    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} linhas omitidas]"
        return content
    except:
        return f"[Erro ao ler: {rel_path}]"


# ─── Chamadas ao Gemma ────────────────────────────────────────────────────────

def call_gemma(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    """Chama Gemma local."""
    
    body = json.dumps({
        "model": "gemma",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    
    return (reasoning + "\n" + content).strip()


def gemma_analyze_task(task: str, memory: str, workspace: str) -> dict:
    """Gemma analisa a tarefa e decide o que fazer."""
    
    system = """Você é um agente de análise. Avalie a tarefa e decida a próxima ação.

RESPONDA EM JSON:
{
    "status": "needs_plan" | "needs_info" | "complete" | "error",
    "analysis": "sua análise em 1-2 frases",
    "files_to_read": ["lista de arquivos para ler se precisar mais contexto"],
    "next_action": "descrição da próxima ação"
}"""
    
    prompt = f"""## Tarefa
{task}

## Memória/Histórico
{memory}

## Estrutura do Workspace
{workspace[:3000]}

Analise e decida o próximo passo."""
    
    response = call_gemma(system, prompt, 1024)
    
    # Tenta extrair JSON
    try:
        # Procura JSON na resposta
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group())
    except:
        pass
    
    return {
        "status": "needs_plan",
        "analysis": response[:200],
        "files_to_read": [],
        "next_action": "Criar plano de execução"
    }


def gemma_create_plan(task: str, memory: str, context: str) -> str:
    """Gemma cria plano de execução."""
    
    brain = read_file_safe(BRAIN_FILE)
    
    system = f"""Você é um planejador de tarefas. Crie um plano detalhado.

{brain}

IMPORTANTE: O plano será executado LITERALMENTE por outro agente.
Seja EXTREMAMENTE específico com caminhos, código, comandos."""
    
    prompt = f"""## Tarefa
{task}

## Memória/Histórico
{memory}

## Contexto Adicional
{context}

Crie o plano de execução no formato especificado."""
    
    return call_gemma(system, prompt, MAX_TOKENS)


def gemma_evaluate_result(task: str, plan: str, result: str) -> dict:
    """Gemma avalia o resultado da execução."""
    
    system = """Avalie se a execução foi bem-sucedida.

RESPONDA EM JSON:
{
    "success": true | false,
    "complete": true | false,
    "analysis": "o que foi feito e o que falta",
    "next_steps": "próximos passos se não completo",
    "memory_update": "resumo comprimido para salvar na memória"
}"""
    
    prompt = f"""## Tarefa Original
{task}

## Plano Executado
{plan}

## Resultado da Execução
{result}

Avalie o resultado."""
    
    response = call_gemma(system, prompt, 1024)
    
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group())
    except:
        pass
    
    return {
        "success": True,
        "complete": True,
        "analysis": "Não foi possível avaliar automaticamente",
        "memory_update": f"Tarefa executada: {task[:100]}"
    }


# ─── Execução de Plano ────────────────────────────────────────────────────────

def parse_plan(plan_text: str) -> list:
    """Extrai passos do plano."""
    steps = []
    
    # Procura padrões de checkbox
    pattern = r'- \[ \] \*\*(?:Passo \d+|[^*]+)\*\*:?\s*([^\n]+)'
    matches = re.finditer(pattern, plan_text)
    
    for match in matches:
        step_text = match.group(0)
        steps.append({
            "text": step_text,
            "start": match.start(),
            "end": match.end()
        })
    
    return steps


def execute_step_local(step_text: str) -> dict:
    """Tenta executar um passo localmente."""
    
    result = {"success": False, "output": "", "method": "local"}
    
    # Detecta tipo de ação
    if "Comando:" in step_text or "rodar" in step_text.lower():
        # Extrai comando
        cmd_match = re.search(r'`([^`]+)`', step_text)
        if cmd_match:
            cmd = cmd_match.group(1)
            try:
                proc = subprocess.run(
                    cmd, shell=True, cwd=BASE_DIR,
                    capture_output=True, text=True, timeout=60
                )
                result["success"] = proc.returncode == 0
                result["output"] = proc.stdout + proc.stderr
            except Exception as e:
                result["output"] = str(e)
    
    elif "criar" in step_text.lower() and "Arquivo:" in step_text:
        # Extrai caminho e conteúdo
        path_match = re.search(r'Arquivo:\s*`([^`]+)`', step_text)
        content_match = re.search(r'```\w*\n([\s\S]*?)```', step_text)
        
        if path_match and content_match:
            path = BASE_DIR / path_match.group(1)
            content = content_match.group(1)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                result["success"] = True
                result["output"] = f"Arquivo criado: {path_match.group(1)}"
            except Exception as e:
                result["output"] = str(e)
    
    return result


def execute_with_copilot_cli(plan: str) -> str:
    """Tenta executar via GitHub Copilot CLI."""
    
    # Verifica se gh copilot está disponível
    try:
        proc = subprocess.run(
            ["gh", "copilot", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            return None
    except:
        return None
    
    # Para comandos, usa gh copilot suggest
    results = []
    steps = parse_plan(plan)
    
    for step in steps:
        if "comando" in step["text"].lower():
            cmd_match = re.search(r'`([^`]+)`', step["text"])
            if cmd_match:
                cmd = cmd_match.group(1)
                try:
                    # Usa gh copilot suggest para validar
                    proc = subprocess.run(
                        ["gh", "copilot", "suggest", "-t", "shell", cmd],
                        capture_output=True, text=True, timeout=30
                    )
                    results.append(f"Comando sugerido: {proc.stdout}")
                except:
                    pass
    
    return "\n".join(results) if results else None


def wait_for_manual_execution(timeout: int = 300) -> Optional[str]:
    """Aguarda execução manual e retorno em result.md."""
    
    log("Aguardando execução manual...", "WARN")
    log(f"  → Execute os passos em agents/plan.md", "INFO")
    log(f"  → Escreva o resultado em agents/result.md", "INFO")
    log(f"  → Ou digite 'skip' para pular", "INFO")
    
    initial_hash = file_hash(RESULT_FILE)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Verifica se result.md mudou
        if file_hash(RESULT_FILE) != initial_hash:
            return read_file_safe(RESULT_FILE)
        
        time.sleep(2)
    
    return None


# ─── Loop Principal ───────────────────────────────────────────────────────────

def update_memory(memory_update: str):
    """Atualiza memória com compressão."""
    
    current = read_file_safe(MEMORY_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Adiciona nova entrada
    new_entry = f"\n### {timestamp}\n{memory_update}\n"
    
    # Se memória muito grande, comprime
    lines = current.split("\n")
    if len(lines) > 100:
        # Mantém header e últimas 50 linhas
        header = "\n".join(lines[:10])
        recent = "\n".join(lines[-50:])
        current = header + "\n\n[Histórico antigo comprimido]\n\n" + recent
    
    write_file(MEMORY_FILE, current + new_entry)


def run_cycle() -> bool:
    """Executa um ciclo do loop. Retorna True se tarefa completa."""
    
    # 1. Carrega contexto
    log("Carregando contexto...", "INFO")
    task = read_file_safe(TASK_FILE)
    memory = read_file_safe(MEMORY_FILE)
    
    # Verifica se há tarefa
    if "[Descreva a tarefa" in task or not task.strip():
        log("Nenhuma tarefa definida em agents/task.md", "WARN")
        return True  # Nada a fazer
    
    # 2. Gera visão do workspace
    log("Analisando workspace...", "INFO")
    workspace = generate_workspace_view()
    write_file(WORKSPACE_VIEW, workspace)
    
    # 3. Gemma analisa
    log("Gemma analisando tarefa...", "THINK")
    analysis = gemma_analyze_task(task, memory, workspace)
    log(f"  Status: {analysis.get('status', 'unknown')}", "INFO")
    log(f"  Análise: {analysis.get('analysis', '')[:100]}", "INFO")
    
    # 4. Se precisa ler arquivos
    files_to_read = analysis.get("files_to_read", [])
    context = ""
    if files_to_read:
        log(f"Lendo {len(files_to_read)} arquivos...", "INFO")
        for f in files_to_read[:5]:  # Máximo 5
            content = read_file_preview(f)
            context += f"\n### {f}\n```\n{content}\n```\n"
    
    # 5. Gemma cria plano
    log("Gemma criando plano...", "THINK")
    plan = gemma_create_plan(task, memory, context)
    write_file(PLAN_FILE, plan)
    log(f"  Plano salvo em agents/plan.md", "OK")
    
    # 6. Tenta executar localmente
    log("Tentando execução local...", "EXEC")
    steps = parse_plan(plan)
    results = []
    
    for i, step in enumerate(steps):
        log(f"  Passo {i+1}/{len(steps)}...", "INFO")
        result = execute_step_local(step["text"])
        
        if result["success"]:
            log(f"    ✅ {result['output'][:50]}", "OK")
            results.append(f"Passo {i+1}: OK - {result['output']}")
        else:
            # Tenta Copilot CLI
            cli_result = execute_with_copilot_cli(step["text"])
            if cli_result:
                results.append(f"Passo {i+1}: Via CLI - {cli_result}")
            else:
                results.append(f"Passo {i+1}: Precisa execução manual")
    
    # 7. Salva resultado
    result_text = f"""# Resultado da Execução

**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Execução Automática

{chr(10).join(results)}

## Status

{"✅ Completo" if all("OK" in r for r in results) else "⚠️ Parcial - alguns passos precisam de atenção"}
"""
    write_file(RESULT_FILE, result_text)
    
    # 8. Gemma avalia
    log("Gemma avaliando resultado...", "THINK")
    evaluation = gemma_evaluate_result(task, plan, result_text)
    
    log(f"  Sucesso: {evaluation.get('success', False)}", "INFO")
    log(f"  Completo: {evaluation.get('complete', False)}", "INFO")
    
    # 9. Atualiza memória
    memory_update = evaluation.get("memory_update", "Ciclo executado")
    update_memory(memory_update)
    
    return evaluation.get("complete", False)


def daemon_loop(watch: bool = False, interval: int = 5):
    """Loop principal do daemon."""
    
    print("=" * 60)
    print("🤖 AUTONOMOUS AGENT DAEMON")
    print("=" * 60)
    print(f"Workspace: {BASE_DIR}")
    print(f"Modo: {'Watch' if watch else 'Single'}")
    print("=" * 60)
    print()
    
    # Verifica servidor
    if not check_server():
        log("llama-server não está rodando!", "ERROR")
        log(f"Inicie em: {SERVER_URL}", "INFO")
        return
    
    log("Servidor OK", "OK")
    
    last_task_hash = ""
    
    while True:
        # Em modo watch, só roda quando task.md muda
        if watch:
            current_hash = file_hash(TASK_FILE)
            if current_hash == last_task_hash:
                time.sleep(interval)
                continue
            last_task_hash = current_hash
        
        log("Iniciando ciclo...", "INFO")
        
        try:
            complete = run_cycle()
            
            if complete:
                log("Tarefa completa!", "OK")
                if not watch:
                    break
            else:
                log("Tarefa incompleta, continuando...", "WARN")
                
        except KeyboardInterrupt:
            log("Interrompido pelo usuário", "WARN")
            break
        except Exception as e:
            log(f"Erro: {e}", "ERROR")
            if not watch:
                break
        
        if not watch:
            break
        
        time.sleep(interval)
    
    print()
    print("=" * 60)
    print("Daemon finalizado")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daemon de Agentes Autônomos")
    parser.add_argument("--once", action="store_true", help="Roda apenas um ciclo")
    parser.add_argument("--watch", action="store_true", help="Monitora task.md continuamente")
    parser.add_argument("--interval", type=int, default=5, help="Intervalo entre checks (segundos)")
    parser.add_argument("--port", type=int, default=8090, help="Porta do llama-server")
    args = parser.parse_args()
    
    global SERVER_URL
    SERVER_URL = f"http://127.0.0.1:{args.port}"
    
    daemon_loop(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()
