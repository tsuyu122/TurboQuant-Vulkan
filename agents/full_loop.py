#!/usr/bin/env python3
"""
🤖 TurboQuant Full Autonomous Loop

Loop 100% autônomo que:
1. Gemma analisa tarefa e cria plano
2. Tenta executar localmente
3. Se precisar Copilot: usa auto_typer para enviar ao VS Code
4. Monitora result.md para resposta
5. Gemma avalia e continua

Uso:
    py agents/full_loop.py                    # Loop único
    py agents/full_loop.py --watch            # Monitora continuamente
    py agents/full_loop.py --task "tarefa"    # Define tarefa e roda
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
from typing import Optional, Tuple

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"

BRAIN_FILE = AGENTS_DIR / "brain.md"
MEMORY_FILE = AGENTS_DIR / "memory.md"
TASK_FILE = AGENTS_DIR / "task.md"
PLAN_FILE = AGENTS_DIR / "plan.md"
RESULT_FILE = AGENTS_DIR / "result.md"
CONTEXT_FILE = AGENTS_DIR / "turboquant_context.md"

SERVER_URL = "http://127.0.0.1:8090"
MAX_TOKENS = 2048

# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    now = datetime.now().strftime("%H:%M:%S")
    symbols = {
        "OK": "✅", "FAIL": "❌", "INFO": "ℹ️", "WARN": "⚠️",
        "THINK": "🧠", "EXEC": "⚡", "WAIT": "⏳", "TYPE": "⌨️"
    }
    print(f"[{now}] {symbols.get(level, '•')} {msg}")


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


# ─── Gemma API ────────────────────────────────────────────────────────────────

def check_server() -> bool:
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.getcode() == 200
    except:
        return False


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
    
    return content.strip()


# ─── Análise e Planejamento ───────────────────────────────────────────────────

def gemma_create_plan(task: str, context: str, memory: str) -> str:
    """Gemma cria plano de execução."""
    
    brain = read_file(BRAIN_FILE)
    tq_context = read_file(CONTEXT_FILE)
    
    system = f"""Você é um planejador de tarefas técnicas.
{brain}

CONTEXTO TURBOQUANT:
{tq_context[:3000]}

FORMATO OBRIGATÓRIO:
```markdown
# Tarefa: [título]

## Passos

- [ ] **Passo 1**: [descrição]
  - Arquivo: `caminho/arquivo.ext`
  - Ação: criar | editar | rodar
  - Conteúdo/Comando: [código ou comando]

## Verificação
[como verificar]
```"""
    
    prompt = f"""## Tarefa
{task}

## Contexto
{context[:2000]}

## Memória
{memory[:1000]}

Crie o plano de execução."""
    
    return call_gemma(system, prompt, MAX_TOKENS)


def gemma_evaluate(task: str, result: str) -> dict:
    """Gemma avalia resultado."""
    
    system = """Avalie se a tarefa foi concluída com sucesso.

RESPONDA EM JSON:
{
    "success": true | false,
    "complete": true | false,
    "analysis": "análise breve",
    "next_action": "próximo passo se não completo"
}"""
    
    prompt = f"""## Tarefa
{task}

## Resultado
{result}

Avalie."""
    
    response = call_gemma(system, prompt, 512)
    
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group())
    except:
        pass
    
    return {"success": True, "complete": True, "analysis": response[:200]}


# ─── Execução ─────────────────────────────────────────────────────────────────

def try_local_execution(plan: str) -> Tuple[bool, str]:
    """Tenta executar passos localmente."""
    
    results = []
    executed = 0
    
    # Extrai passos
    pattern = r'- \[ \] \*\*([^*]+)\*\*[:\s]*([^\n]*)'
    
    for match in re.finditer(pattern, plan):
        step_name = match.group(1)
        step_desc = match.group(2)
        
        # Verifica se é criação de arquivo
        file_match = re.search(r'Arquivo:\s*`([^`]+)`', plan[match.end():match.end()+500])
        code_match = re.search(r'```\w*\n([\s\S]*?)```', plan[match.end():match.end()+2000])
        
        if file_match and code_match and "criar" in step_desc.lower():
            # Cria arquivo
            filepath = BASE_DIR / file_match.group(1)
            content = code_match.group(1)
            
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content, encoding="utf-8")
                results.append(f"✅ Criado: {file_match.group(1)}")
                executed += 1
            except Exception as e:
                results.append(f"❌ Erro criando {file_match.group(1)}: {e}")
        
        # Verifica se é comando
        cmd_match = re.search(r'Comando:\s*`([^`]+)`', plan[match.end():match.end()+200])
        if cmd_match:
            cmd = cmd_match.group(1)
            try:
                proc = subprocess.run(
                    cmd, shell=True, cwd=BASE_DIR,
                    capture_output=True, text=True, timeout=60
                )
                if proc.returncode == 0:
                    results.append(f"✅ Comando OK: {cmd[:30]}...")
                    executed += 1
                else:
                    results.append(f"⚠️ Comando falhou: {proc.stderr[:100]}")
            except Exception as e:
                results.append(f"❌ Erro no comando: {e}")
    
    return executed > 0, "\n".join(results)


def check_copilot_cli_capability() -> Tuple[bool, str]:
    """
    Verifica se o GitHub Copilot CLI está disponível E consegue executar código.
    
    Retorna: (disponível, motivo)
    
    NOTA: Atualmente o gh copilot CLI só faz 'suggest' e 'explain'.
    Ele NÃO consegue executar código ou criar arquivos diretamente.
    Se uma versão futura adicionar essa capacidade, esta função detectará.
    """
    try:
        # Verifica se gh copilot existe
        result = subprocess.run(
            ["gh", "copilot", "--version"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            return False, "gh copilot não instalado"
        
        # Verifica se tem capacidade de execução (não tem atualmente)
        # O CLI atual só tem 'suggest' e 'explain', não 'execute' ou 'run'
        help_result = subprocess.run(
            ["gh", "copilot", "--", "--help"],
            capture_output=True, text=True, timeout=10
        )
        
        help_text = help_result.stdout + help_result.stderr
        
        # Procura por comandos de execução (run, execute, do, etc)
        execution_commands = ["execute", "run ", "do ", "apply", "create-file"]
        has_execution = any(cmd in help_text.lower() for cmd in execution_commands)
        
        if has_execution:
            return True, "CLI com capacidade de execução"
        else:
            return False, "CLI não tem capacidade de executar código (só suggest/explain)"
            
    except FileNotFoundError:
        return False, "gh não encontrado no PATH"
    except subprocess.TimeoutExpired:
        return False, "Timeout verificando CLI"
    except Exception as e:
        return False, f"Erro: {e}"


def execute_via_copilot_cli(plan: str) -> Tuple[bool, str]:
    """
    Tenta executar plano via GitHub Copilot CLI.
    
    NOTA: Esta função está preparada para quando/se o CLI ganhar 
    capacidade de execução. Atualmente retorna False.
    """
    available, reason = check_copilot_cli_capability()
    
    if not available:
        log(f"CLI indisponível: {reason}", "INFO")
        return False, reason
    
    # Se CLI tiver capacidade de execução, implementar aqui
    # Por enquanto, o CLI só faz suggest/explain
    
    results = []
    
    # Exemplo de como seria se o CLI suportasse execução:
    # try:
    #     result = subprocess.run(
    #         ["gh", "copilot", "execute", "--plan", plan_file],
    #         capture_output=True, text=True, timeout=120
    #     )
    #     return result.returncode == 0, result.stdout
    # except Exception as e:
    #     return False, str(e)
    
    return False, "CLI não suporta execução direta (ainda)"


def send_to_copilot_via_typing(message: str, delay: int = 3) -> bool:
    """Envia para Copilot usando automação física."""
    
    log("Usando automação física (auto_typer)...", "TYPE")
    
    # Verifica se já foi importado
    auto_typer_path = AGENTS_DIR / "auto_typer.py"
    if not auto_typer_path.exists():
        log("auto_typer.py não encontrado!", "FAIL")
        return False
    
    try:
        # Roda o auto_typer
        result = subprocess.run(
            [sys.executable, str(auto_typer_path), 
             "--message", message,
             "--delay", str(delay)],
            cwd=BASE_DIR,
            timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        log(f"Erro no auto_typer: {e}", "FAIL")
        return False


def wait_for_result(timeout: int = 300) -> Optional[str]:
    """Aguarda alteração em result.md."""
    
    initial_hash = file_hash(RESULT_FILE)
    initial_content = read_file(RESULT_FILE)
    start_time = time.time()
    
    log(f"Aguardando resultado (max {timeout}s)...", "WAIT")
    
    while time.time() - start_time < timeout:
        current_hash = file_hash(RESULT_FILE)
        
        if current_hash != initial_hash:
            new_content = read_file(RESULT_FILE)
            if new_content != initial_content:
                log("Resultado recebido!", "OK")
                return new_content
        
        time.sleep(2)
    
    log("Timeout aguardando resultado", "WARN")
    return None


# ─── Loop Principal ───────────────────────────────────────────────────────────

def update_memory(summary: str):
    """Atualiza memory.md com novo resumo."""
    
    current = read_file(MEMORY_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    new_entry = f"\n### {timestamp}\n{summary}\n"
    
    # Comprime se muito grande
    lines = current.split("\n")
    if len(lines) > 80:
        header = "\n".join(lines[:5])
        recent = "\n".join(lines[-40:])
        current = header + "\n\n[...comprimido...]\n\n" + recent
    
    write_file(MEMORY_FILE, current + new_entry)


def run_cycle(use_auto_typer: bool = True) -> bool:
    """
    Executa um ciclo completo. Retorna True se tarefa completa.
    
    Ordem de prioridade:
    1. Execução local (criar arquivos, rodar comandos simples)
    2. GitHub Copilot CLI (SE tiver capacidade de execução)
    3. Automação física (fallback - auto_typer)
    """
    
    # 1. Carrega tarefa
    task = read_file(TASK_FILE)
    if not task.strip() or "[Descreva" in task:
        log("Nenhuma tarefa em task.md", "WARN")
        return True
    
    memory = read_file(MEMORY_FILE)
    context = ""  # Pode adicionar mais contexto aqui
    
    # 2. Gemma cria plano
    log("Gemma criando plano...", "THINK")
    plan = gemma_create_plan(task, context, memory)
    write_file(PLAN_FILE, plan)
    log("Plano salvo em plan.md", "OK")
    
    # 3. Tenta execução local primeiro
    log("Tentando execução local...", "EXEC")
    local_success, local_result = try_local_execution(plan)
    
    if local_success:
        log(f"✅ Execução local bem-sucedida!", "OK")
        log(f"   {local_result[:100]}", "INFO")
        
        # Registra resultado
        result_text = f"""# Resultado da Execução Local

**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Método**: Execução local pelo daemon

## Resultados

{local_result}
"""
        write_file(RESULT_FILE, result_text)
        update_memory(f"Executado localmente: {local_result[:100]}")
        return True
    
    log("Execução local não conseguiu (passos complexos)", "INFO")
    
    # 4. Tenta GitHub Copilot CLI (se tiver capacidade)
    log("Verificando GitHub Copilot CLI...", "INFO")
    cli_available, cli_reason = check_copilot_cli_capability()
    
    if cli_available:
        log(f"CLI disponível: {cli_reason}", "OK")
        cli_success, cli_result = execute_via_copilot_cli(plan)
        
        if cli_success:
            log("✅ Executado via CLI!", "OK")
            
            # Gemma avalia
            log("Gemma avaliando resultado...", "THINK")
            evaluation = gemma_evaluate(task, cli_result)
            update_memory(f"Executado via CLI: {evaluation.get('analysis', '')[:100]}")
            return evaluation.get("complete", True)
        else:
            log(f"CLI falhou: {cli_result}", "WARN")
    else:
        log(f"CLI não disponível: {cli_reason}", "INFO")
    
    # 5. Fallback: Automação física (auto_typer)
    if use_auto_typer:
        log("Usando fallback: automação física (auto_typer)...", "TYPE")
        
        # Prepara mensagem
        message = "executa o plano"
        
        sent = send_to_copilot_via_typing(message, delay=5)
        
        if sent:
            # Aguarda resultado
            result = wait_for_result(timeout=180)
            
            if result:
                # Gemma avalia
                log("Gemma avaliando resultado...", "THINK")
                evaluation = gemma_evaluate(task, result)
                
                log(f"Avaliação: {evaluation.get('analysis', '')[:80]}", "INFO")
                
                # Atualiza memória
                update_memory(f"Tarefa: {task[:50]}...\nResultado: {evaluation.get('analysis', '')}")
                
                return evaluation.get("complete", True)
        else:
            log("Falha na automação física", "FAIL")
    else:
        log("Automação física desabilitada (--no-auto)", "WARN")
    
    return False


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Full Autonomous Loop")
    parser.add_argument("--watch", "-w", action="store_true", help="Monitora continuamente")
    parser.add_argument("--task", "-t", type=str, help="Define tarefa e roda")
    parser.add_argument("--no-auto", action="store_true", help="Não usa automação física")
    parser.add_argument("--interval", type=int, default=30, help="Intervalo em watch mode (s)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤖 TURBOQUANT FULL AUTONOMOUS LOOP")
    print("="*60)
    
    # Verifica servidor
    if not check_server():
        log("Servidor Gemma não está rodando!", "FAIL")
        log("Inicie com: llama-server -m modelo.gguf --port 8090", "INFO")
        return 1
    
    log("Servidor Gemma OK", "OK")
    
    # Se passou tarefa, define
    if args.task:
        task_content = f"""# 🎯 Tarefa Atual

## Tarefa
{args.task}

## Prioridade
- [x] Alta (fazer agora)
"""
        write_file(TASK_FILE, task_content)
        log(f"Tarefa definida: {args.task[:50]}...", "OK")
    
    # Executa
    if args.watch:
        log("Modo watch - monitorando continuamente", "INFO")
        last_task_hash = ""
        
        while True:
            current_hash = file_hash(TASK_FILE)
            
            if current_hash != last_task_hash:
                log("Nova tarefa detectada!", "INFO")
                last_task_hash = current_hash
                
                complete = run_cycle(use_auto_typer=not args.no_auto)
                
                if complete:
                    log("Tarefa completa! Aguardando próxima...", "OK")
            
            time.sleep(args.interval)
    else:
        # Ciclo único
        complete = run_cycle(use_auto_typer=not args.no_auto)
        
        if complete:
            log("Ciclo completo!", "OK")
            return 0
        else:
            log("Ciclo incompleto - pode precisar intervenção", "WARN")
            return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
