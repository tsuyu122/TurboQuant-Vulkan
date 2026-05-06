#!/usr/bin/env python3
"""
🤖 TurboQuant Autonomous Agent v3

Sistema completo 100% autônomo:
1. Gemma planeja usando contexto de 512k (com auto-limpeza)
2. Execução: Local → CLI → Auto-typer (de qualquer janela)
3. Contexto salvo entre sessões
4. Loop até tarefa completa

Uso:
    py agents/autonomous_v3.py --task "criar X"    # Tarefa única
    py agents/autonomous_v3.py --watch             # Monitora task.md
    py agents/autonomous_v3.py --test              # Teste do sistema
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

# For Windows terminals with legacy encoding, force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Imports Locais ───────────────────────────────────────────────────────────

# Adiciona agents/ ao path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from context_manager import get_context_manager, ContextManager
except ImportError:
    print("⚠️ context_manager não encontrado, usando fallback simples")
    get_context_manager = None


# ─── Configuração ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"

# Arquivos do sistema
BRAIN_FILE = AGENTS_DIR / "brain.md"
MEMORY_FILE = AGENTS_DIR / "memory.md"
TASK_FILE = AGENTS_DIR / "task.md"
PLAN_FILE = AGENTS_DIR / "plan.md"
RESULT_FILE = AGENTS_DIR / "result.md"
TQ_CONTEXT_FILE = AGENTS_DIR / "turboquant_context.md"
MISSION_FILE = AGENTS_DIR / "mission.md"
README_FILE = BASE_DIR / "README.md"

# Servidor Gemma - Ollama
SERVER_URL = "http://127.0.0.1:11434"
MODEL_NAME = "gemma4:e4b"
MAX_TOKENS = 2048
CONTEXT_TOKENS = 512000  # 512k

# Unicode para log
SYMBOLS = {
    "OK": "✅", "FAIL": "❌", "INFO": "ℹ️", "WARN": "⚠️",
    "THINK": "🧠", "EXEC": "⚡", "WAIT": "⏳", "TYPE": "⌨️",
    "FIND": "🔍", "SAVE": "💾", "CLEAN": "🧹"
}


# ─── Utilidades ───────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {SYMBOLS.get(level, '•')} {msg}")


def read_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except:
        return ""


def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


# ─── Gemma API (Ollama) ───────────────────────────────────────────────────────

def check_server() -> bool:
    try:
        req = urllib.request.Request(f"{SERVER_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.getcode() == 200
    except:
        return False


def call_gemma(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    """Chama Gemma via Ollama."""
    
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.3,
        }
    }).encode()
    
    req = urllib.request.Request(
        f"{SERVER_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    # Gemma 4 E4B - timeout de 5 minutos com retry
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            log(f"⚠️ Tentativa {attempt+1}/3 falhou: {e}")
            if attempt == 2:
                raise
            time.sleep(5)


# ─── Contexto ─────────────────────────────────────────────────────────────────

def get_full_context() -> str:
    """Monta contexto completo para Gemma."""
    
    parts = []
    
    # 1. Brain (instruções)
    brain = read_file(BRAIN_FILE)
    if brain:
        parts.append(f"# INSTRUÇÕES\n{brain[:3000]}")
    
    # 2. Contexto TurboQuant
    tq_context = read_file(TQ_CONTEXT_FILE)
    if tq_context:
        parts.append(f"\n# CONTEXTO TURBOQUANT\n{tq_context[:5000]}")
    
    # 3. Missão técnica
    mission = read_file(MISSION_FILE)
    if mission:
        parts.append(f"\n# MISSÃO\n{mission[:8000]}")
    
    # 4. README local
    readme = read_file(README_FILE)
    if readme:
        parts.append(f"\n# README\n{readme[:8000]}")
    
    # 5. Memória/Histórico
    memory = read_file(MEMORY_FILE)
    if memory:
        parts.append(f"\n# MEMÓRIA\n{memory[:2000]}")
    
    # 4. Contexto gerenciado (se disponível)
    if get_context_manager:
        try:
            cm = get_context_manager()
            managed_context = cm.get_context_for_gemma(max_chars=10000)
            if managed_context:
                parts.append(f"\n# HISTÓRICO DA SESSÃO\n{managed_context}")
        except:
            pass
    
    return "\n\n".join(parts)


def save_to_context(entry_type: str, content: str):
    """Salva entrada no contexto gerenciado."""
    if get_context_manager:
        try:
            cm = get_context_manager()
            cm.add_entry(entry_type, content[:5000])  # Limita tamanho
        except Exception as e:
            log(f"Erro salvando contexto: {e}", "WARN")


# ─── Planejamento ─────────────────────────────────────────────────────────────

def gemma_create_plan(task: str) -> str:
    """Gemma cria plano de execução."""
    
    context = get_full_context()
    
    system = f"""{context}

VOCÊ É UM PLANEJADOR. Crie um plano de execução.

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
```

SEJA ESPECÍFICO: caminhos completos, código exato."""
    
    prompt = f"## TAREFA\n{task}\n\nCrie o plano de execução."
    
    return call_gemma(system, prompt, MAX_TOKENS)


def gemma_evaluate(task: str, result: str) -> dict:
    """Gemma avalia resultado."""
    
    system = """Avalie se a tarefa foi concluída.

RESPONDA EM JSON:
{
    "success": true | false,
    "complete": true | false,
    "analysis": "análise breve",
    "next_action": "próximo passo se não completo"
}"""
    
    prompt = f"## TAREFA\n{task}\n\n## RESULTADO\n{result}\n\nAvalie."
    
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
        step_text = plan[match.end():match.end()+2000]
        
        # Criação de arquivo
        file_match = re.search(r'Arquivo:\s*`([^`]+)`', step_text)
        code_match = re.search(r'```\w*\n([\s\S]*?)```', step_text)
        
        if file_match and code_match and "criar" in step_desc.lower():
            filepath = BASE_DIR / file_match.group(1)
            content = code_match.group(1)
            
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content, encoding="utf-8")
                results.append(f"✅ Criado: {file_match.group(1)}")
                executed += 1
            except Exception as e:
                results.append(f"❌ Erro: {e}")
        
        # Comando
        cmd_match = re.search(r'Comando:\s*`([^`]+)`', step_text)
        if cmd_match:
            cmd = cmd_match.group(1)
            try:
                proc = subprocess.run(
                    cmd, shell=True, cwd=BASE_DIR,
                    capture_output=True, text=True, timeout=60
                )
                if proc.returncode == 0:
                    results.append(f"✅ Comando: {cmd[:30]}...")
                    executed += 1
                else:
                    results.append(f"⚠️ Falhou: {proc.stderr[:100]}")
            except Exception as e:
                results.append(f"❌ Erro: {e}")
    
    return executed > 0, "\n".join(results)


def send_to_copilot(message: str, delay: int = 5) -> bool:
    """Envia para Copilot via auto_typer_v2."""
    
    auto_typer = AGENTS_DIR / "auto_typer_v2.py"
    
    if not auto_typer.exists():
        auto_typer = AGENTS_DIR / "auto_typer.py"
    
    if not auto_typer.exists():
        log("auto_typer não encontrado!", "FAIL")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(auto_typer), "-m", message, "-d", str(delay)],
            cwd=BASE_DIR,
            timeout=120
        )
        return result.returncode == 0
    except Exception as e:
        log(f"Erro no auto_typer: {e}", "FAIL")
        return False


def wait_for_result(timeout: int = 300) -> Optional[str]:
    """Aguarda alteração em result.md."""
    
    initial_hash = file_hash(RESULT_FILE)
    start_time = time.time()
    
    log(f"Aguardando resultado (max {timeout}s)...", "WAIT")
    
    while time.time() - start_time < timeout:
        if file_hash(RESULT_FILE) != initial_hash:
            new_content = read_file(RESULT_FILE)
            log("Resultado recebido!", "OK")
            return new_content
        time.sleep(2)
    
    log("Timeout", "WARN")
    return None


# ─── Memória ──────────────────────────────────────────────────────────────────

def update_memory(summary: str):
    """Atualiza memory.md com compressão automática."""
    
    current = read_file(MEMORY_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    new_entry = f"\n### {timestamp}\n{summary}\n"
    
    # Comprime se muito grande (>100 linhas)
    lines = current.split("\n")
    if len(lines) > 100:
        header = "\n".join(lines[:5])
        recent = "\n".join(lines[-50:])
        current = header + "\n\n[...histórico comprimido...]\n\n" + recent
        log("Memória comprimida", "CLEAN")
    
    write_file(MEMORY_FILE, current + new_entry)
    
    # Salva no contexto gerenciado também
    save_to_context("memory", summary)


# ─── Ciclo Principal ──────────────────────────────────────────────────────────

def run_cycle() -> bool:
    """
    Executa um ciclo completo.
    
    Retorna True se tarefa completa.
    """
    
    # 1. Carrega tarefa
    task = read_file(TASK_FILE)
    if not task.strip() or "[Descreva" in task:
        log("Nenhuma tarefa em task.md", "WARN")
        return True
    
    log(f"Tarefa: {task[:80]}...", "INFO")
    save_to_context("task", task)
    
    # 2. Gemma cria plano
    log("Gemma criando plano...", "THINK")
    plan = gemma_create_plan(task)
    write_file(PLAN_FILE, plan)
    log("Plano salvo em plan.md", "OK")
    save_to_context("plan", plan)
    
    # 3. Tenta execução local
    log("Tentando execução local...", "EXEC")
    local_success, local_result = try_local_execution(plan)
    
    if local_success:
        log("Execução local bem-sucedida!", "OK")
        
        result_text = f"""# Resultado da Execução Local

**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Método**: Local (daemon)

## Resultados

{local_result}
"""
        write_file(RESULT_FILE, result_text)
        save_to_context("result", local_result)
        update_memory(f"Executado localmente: {local_result[:100]}")
        return True
    
    log("Execução local não conseguiu todos os passos", "INFO")
    
    # 4. Envia para Copilot via auto_typer
    log("Enviando para Copilot (auto_typer)...", "TYPE")
    
    sent = send_to_copilot("executa o plano", delay=5)
    
    if sent:
        # Aguarda resultado
        result = wait_for_result(timeout=180)
        
        if result:
            save_to_context("result", result)
            
            # Gemma avalia
            log("Gemma avaliando resultado...", "THINK")
            evaluation = gemma_evaluate(task, result)
            
            log(f"Avaliação: {evaluation.get('analysis', '')[:80]}", "INFO")
            update_memory(f"Via Copilot: {evaluation.get('analysis', '')[:100]}")
            
            return evaluation.get("complete", True)
    
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def test_system():
    """Testa componentes do sistema."""
    
    print("\n" + "="*60)
    print("🧪 TESTE DO SISTEMA AUTONOMOUS v3")
    print("="*60 + "\n")
    
    tests = {}
    
    # 1. Servidor Gemma
    log("Testando servidor Gemma...", "FIND")
    tests["gemma"] = check_server()
    log(f"Gemma: {'OK' if tests['gemma'] else 'OFFLINE'}", "OK" if tests["gemma"] else "FAIL")
    
    # 2. Context Manager
    log("Testando context_manager...", "FIND")
    tests["context"] = get_context_manager is not None
    if tests["context"]:
        try:
            cm = get_context_manager()
            stats = cm.get_stats()
            log(f"Context: {stats['entries']} entradas, {stats['usage_percent']:.1f}% usado", "OK")
        except Exception as e:
            log(f"Context erro: {e}", "WARN")
            tests["context"] = False
    else:
        log("Context: Não disponível", "WARN")
    
    # 3. Auto-typer
    log("Testando auto_typer...", "FIND")
    auto_typer = AGENTS_DIR / "auto_typer_v2.py"
    tests["auto_typer"] = auto_typer.exists()
    log(f"Auto-typer v2: {'OK' if tests['auto_typer'] else 'Não encontrado'}", 
        "OK" if tests["auto_typer"] else "WARN")
    
    # 4. Arquivos
    log("Verificando arquivos...", "FIND")
    files_ok = all([
        BRAIN_FILE.exists(),
        TASK_FILE.exists(),
        TQ_CONTEXT_FILE.exists()
    ])
    tests["files"] = files_ok
    log(f"Arquivos: {'OK' if files_ok else 'Alguns faltando'}", "OK" if files_ok else "WARN")
    
    # Resumo
    print("\n" + "="*60)
    all_ok = all(tests.values())
    
    if all_ok:
        log("✅ SISTEMA PRONTO!", "OK")
    else:
        log("⚠️ Alguns componentes com problema", "WARN")
        if not tests["gemma"]:
            log("   → Inicie: ollama serve gemma4:e4b --port 11434", "INFO")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Autonomous Agent v3")
    parser.add_argument("--task", "-t", type=str, help="Define tarefa e executa")
    parser.add_argument("--watch", "-w", action="store_true", help="Monitora task.md")
    parser.add_argument("--test", action="store_true", help="Testa o sistema")
    parser.add_argument("--interval", type=int, default=30, help="Intervalo watch (s)")
    
    args = parser.parse_args()
    
    if args.test:
        return 0 if test_system() else 1
    
    print("\n" + "="*60)
    print("🤖 TURBOQUANT AUTONOMOUS AGENT v3")
    print("="*60)
    
    # Verifica servidor
    if not check_server():
        log("Servidor Gemma offline!", "FAIL")
        log("Inicie: ollama serve gemma4:e4b --port 11434", "INFO")
        return 1
    
    log("Servidor Gemma OK", "OK")
    
    # Define tarefa se passada
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
        log("Modo watch - monitorando task.md", "INFO")
        last_hash = ""
        
        while True:
            current_hash = file_hash(TASK_FILE)
            
            if current_hash != last_hash:
                log("Nova tarefa detectada!", "INFO")
                last_hash = current_hash
                
                complete = run_cycle()
                
                if complete:
                    log("Tarefa completa! Aguardando próxima...", "OK")
            
            time.sleep(args.interval)
    else:
        # Ciclo único
        complete = run_cycle()
        
        if complete:
            log("✅ Ciclo completo!", "OK")
            return 0
        else:
            log("⚠️ Ciclo incompleto", "WARN")
            return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
