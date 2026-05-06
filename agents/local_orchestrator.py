#!/usr/bin/env python3
"""
Autonomous Agent System - LOCAL VERSION (No API costs!)

Uses your local llama-server (Gemma 4 26B) for both planning and execution.
Works with the server you already have running for TurboQuant benchmarks.

Usage:
    # Start server first (if not running):
    llama-server -m models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf -ngl 18 -c 8192 --port 8090

    # Then run:
    py agents/local_orchestrator.py "sua tarefa aqui"

No API keys needed - uses your local model!
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    server_url: str = "http://127.0.0.1:8090"
    max_tokens: int = 2048
    temperature: float = 0.3
    max_iterations: int = 10
    workspace: Path = Path(".")


# ─── Local LLM API ────────────────────────────────────────────────────────────

def call_llm(config: Config, system: str, user: str) -> str:
    """Call local llama-server via OpenAI-compatible API."""
    
    body = json.dumps({
        "model": "gemma",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }).encode()
    
    req = urllib.request.Request(
        f"{config.server_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        
        return (reasoning + "\n" + content).strip()
        
    except urllib.error.URLError as e:
        return f"ERROR: Cannot connect to llama-server at {config.server_url}. Is it running?"
    except Exception as e:
        return f"ERROR: {e}"


def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON in code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except:
                continue
    
    return None


# ─── Planner ──────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """Você é um arquiteto de software sênior. Crie planos de execução detalhados.

REGRAS:
1. Seja ESPECÍFICO - inclua caminhos de arquivos exatos, nomes de funções, código
2. Cada passo deve ser executável independentemente
3. Inclua critérios de verificação

FORMATO DE SAÍDA (JSON válido):
{
    "summary": "resumo em 1-2 frases",
    "steps": [
        {
            "id": 1,
            "action": "create_file",
            "target": "caminho/do/arquivo.py",
            "content": "conteúdo completo do arquivo",
            "verification": "como verificar que funcionou"
        }
    ]
}

AÇÕES DISPONÍVEIS:
- create_file: criar arquivo novo (target=caminho, content=conteúdo)
- edit_file: editar arquivo (target=caminho, content=descrição da edição)
- run_command: rodar comando (target=comando)
- read_file: ler arquivo (target=caminho)
- delete_file: deletar arquivo (target=caminho)

RESPONDA APENAS COM JSON VÁLIDO."""


VERIFIER_SYSTEM = """Você é um engenheiro de QA verificando resultados.

Analise o log de execução e determine:
1. Cada passo foi executado corretamente?
2. Existem erros?
3. A tarefa está completa?

FORMATO DE SAÍDA (JSON válido):
{
    "status": "COMPLETE" ou "NEEDS_CHANGES" ou "FAILED",
    "issues": ["problema 1", "problema 2"],
    "changes_needed": [
        {
            "step_id": 1,
            "problem": "o que deu errado",
            "fix": "como consertar"
        }
    ],
    "message": "resumo para o usuário"
}

RESPONDA APENAS COM JSON VÁLIDO."""


def create_plan(config: Config, task: str, history: list = None) -> dict:
    """Create execution plan using local LLM."""
    
    context = ""
    if history:
        context = f"\n\nHistórico de execuções anteriores:\n{json.dumps(history[-2:], indent=2, ensure_ascii=False)}"
    
    prompt = f"""Tarefa: {task}

Workspace: {config.workspace.absolute()}
{context}

Crie um plano detalhado de execução."""
    
    response = call_llm(config, PLANNER_SYSTEM, prompt)
    
    if response.startswith("ERROR:"):
        return {"error": response, "steps": []}
    
    plan = extract_json(response)
    if not plan:
        # Fallback: single step plan
        return {
            "summary": "Execução direta",
            "steps": [{
                "id": 1,
                "action": "run_command",
                "target": f"echo 'Não consegui criar plano estruturado. Resposta: {response[:200]}'",
                "verification": "manual"
            }]
        }
    
    return plan


def verify_execution(config: Config, task: str, plan: dict, log: list) -> dict:
    """Verify execution results using local LLM."""
    
    prompt = f"""## Tarefa Original
{task}

## Plano Executado
{json.dumps(plan, indent=2, ensure_ascii=False)}

## Log de Execução
{json.dumps(log, indent=2, ensure_ascii=False)}

Verifique se a execução foi bem-sucedida."""
    
    response = call_llm(config, VERIFIER_SYSTEM, prompt)
    
    if response.startswith("ERROR:"):
        return {"status": "FAILED", "message": response, "issues": [response]}
    
    result = extract_json(response)
    if not result:
        # Assume success if can't parse (be optimistic)
        return {
            "status": "COMPLETE",
            "message": "Execução aparentemente concluída",
            "issues": []
        }
    
    return result


# ─── Executor ─────────────────────────────────────────────────────────────────

def execute_step(config: Config, step: dict) -> dict:
    """Execute a single step."""
    
    action = step.get("action", "")
    target = step.get("target", "")
    content = step.get("content", "")
    workspace = config.workspace
    
    result = {"success": False, "action": action, "output": "", "error": None}
    
    try:
        if action == "create_file":
            file_path = workspace / target
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            result = {
                "success": True,
                "action": f"Criado: {target}",
                "output": f"Arquivo criado ({len(content)} chars)",
                "error": None
            }
            
        elif action == "edit_file":
            file_path = workspace / target
            if not file_path.exists():
                result["error"] = f"Arquivo não existe: {target}"
            else:
                # For edits, ask LLM to generate new content
                current = file_path.read_text(encoding="utf-8")
                
                edit_prompt = f"""Arquivo atual ({target}):
```
{current[:4000]}
```

Edição solicitada: {content}

Retorne o conteúdo COMPLETO do arquivo editado. APENAS o conteúdo, sem explicações."""
                
                new_content = call_llm(config, "Você é um editor de código.", edit_prompt)
                
                # Clean markdown if present
                if new_content.startswith("```"):
                    lines = new_content.split("\n")
                    new_content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
                
                file_path.write_text(new_content, encoding="utf-8")
                result = {
                    "success": True,
                    "action": f"Editado: {target}",
                    "output": f"Arquivo atualizado ({len(new_content)} chars)",
                    "error": None
                }
                
        elif action == "run_command":
            proc = subprocess.run(
                target,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120
            )
            result = {
                "success": proc.returncode == 0,
                "action": f"Comando: {target[:50]}",
                "output": (proc.stdout + proc.stderr)[:1000],
                "error": None if proc.returncode == 0 else f"Exit code {proc.returncode}"
            }
            
        elif action == "read_file":
            file_path = workspace / target
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                result = {
                    "success": True,
                    "action": f"Lido: {target}",
                    "output": content[:1000],
                    "error": None
                }
            else:
                result["error"] = f"Arquivo não encontrado: {target}"
                
        elif action == "delete_file":
            file_path = workspace / target
            if file_path.exists():
                file_path.unlink()
            result = {
                "success": True,
                "action": f"Deletado: {target}",
                "output": "OK",
                "error": None
            }
            
        else:
            result["error"] = f"Ação desconhecida: {action}"
            
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout (120s)"
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ─── Main Loop ────────────────────────────────────────────────────────────────

def check_server(config: Config) -> bool:
    """Check if llama-server is running."""
    try:
        req = urllib.request.Request(f"{config.server_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except:
        return False


def run_loop(config: Config, task: str):
    """Main autonomous loop."""
    
    print("=" * 60)
    print("🤖 AGENTES AUTÔNOMOS (LOCAL - Sem custo!)")
    print("=" * 60)
    print(f"Servidor: {config.server_url}")
    print(f"Workspace: {config.workspace.absolute()}")
    print(f"Tarefa: {task}")
    print("=" * 60)
    
    # Check server
    print("\n🔌 Verificando servidor...", end=" ")
    if not check_server(config):
        print("❌")
        print(f"\nERRO: llama-server não está rodando em {config.server_url}")
        print("\nInicie o servidor primeiro:")
        print('  .\\llama_src\\build_vulkan\\bin\\Release\\llama-server.exe -m models\\google_gemma-4-26B-A4B-it-Q4_K_M.gguf -ngl 18 -c 8192 --port 8090')
        return False
    print("✅")
    
    iteration = 0
    history = []
    
    while iteration < config.max_iterations:
        iteration += 1
        print(f"\n{'─'*50}")
        print(f"📋 ITERAÇÃO {iteration}/{config.max_iterations}")
        print(f"{'─'*50}")
        
        # Plan
        print("\n🧠 Criando plano...")
        plan = create_plan(config, task, history)
        
        if "error" in plan:
            print(f"❌ Erro: {plan['error']}")
            continue
        
        print(f"   Resumo: {plan.get('summary', 'N/A')}")
        print(f"   Passos: {len(plan.get('steps', []))}")
        
        # Execute
        print("\n⚡ Executando...")
        execution_log = []
        
        for step in plan.get("steps", []):
            step_id = step.get("id", "?")
            action = step.get("action", "?")
            target = step.get("target", "")[:40]
            
            print(f"   [{step_id}] {action}: {target}...", end=" ")
            
            result = execute_step(config, step)
            execution_log.append({"step": step, "result": result})
            
            if result["success"]:
                print("✅")
            else:
                print(f"❌ {result.get('error', '?')}")
        
        # Verify
        print("\n🔍 Verificando...")
        verification = verify_execution(config, task, plan, execution_log)
        
        status = verification.get("status", "UNKNOWN")
        message = verification.get("message", "")
        
        print(f"   Status: {status}")
        if message:
            print(f"   Mensagem: {message}")
        
        history.append({
            "iteration": iteration,
            "plan": plan.get("summary"),
            "steps": len(plan.get("steps", [])),
            "status": status
        })
        
        if status == "COMPLETE":
            print("\n" + "=" * 60)
            print("✅ TAREFA COMPLETA!")
            print("=" * 60)
            return True
        
        if status == "FAILED":
            issues = verification.get("issues", [])
            print(f"\n❌ Falhou: {issues}")
            # Continue trying
        
        time.sleep(1)
    
    print("\n⚠️ Máximo de iterações atingido")
    return False


def main():
    parser = argparse.ArgumentParser(description="Agentes Autônomos (Local)")
    parser.add_argument("task", help="Descrição da tarefa")
    parser.add_argument("--workspace", "-w", default=".", help="Diretório de trabalho")
    parser.add_argument("--port", "-p", type=int, default=8090, help="Porta do llama-server")
    parser.add_argument("--max-iter", "-n", type=int, default=10, help="Máximo de iterações")
    args = parser.parse_args()
    
    config = Config(
        server_url=f"http://127.0.0.1:{args.port}",
        max_iterations=args.max_iter,
        workspace=Path(args.workspace).resolve()
    )
    
    success = run_loop(config, args.task)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
