#!/usr/bin/env python3
"""
planner.py - Gemma 4 cria um plano para o Copilot executar

Fluxo:
1. Você roda: py agents/planner.py "sua tarefa"
2. Gemma 4 cria o plano em agents/plan.md
3. No VS Code, você digita: /plan
4. O Copilot lê e executa o plano

Sem API keys - usa seu modelo local!
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PLAN_FILE = BASE_DIR / "agents" / "plan.md"
SERVER_URL = "http://127.0.0.1:8090"


PLANNER_PROMPT = """Você é um arquiteto de software sênior. Sua tarefa é criar um PLANO DETALHADO que outro agente (Copilot) vai executar.

REGRAS CRÍTICAS:
1. Seja EXTREMAMENTE específico - caminhos de arquivos, nomes de funções, código exato
2. Cada passo deve ser executável independentemente
3. Use checkbox "- [ ]" para cada passo (o executor vai marcar como feito)
4. Inclua código completo quando necessário (use blocos ```)
5. O executor NÃO pode te responder - o plano deve ser completo

FORMATO OBRIGATÓRIO:
```markdown
# Tarefa: [descrição clara]

## Contexto
[informações relevantes sobre o projeto, arquivos existentes, etc.]

## Passos

- [ ] **Passo 1**: [descrição]
  - Arquivo: `caminho/do/arquivo.py`
  - Ação: criar/editar/deletar
  - Conteúdo:
  ```python
  # código completo aqui
  ```

- [ ] **Passo 2**: [descrição]
  - Comando: `comando a rodar`
  - Resultado esperado: [o que deve acontecer]

## Verificação
[como saber se funcionou]

## Notas
[avisos, dependências, etc.]
```

IMPORTANTE: O plano SERÁ EXECUTADO LITERALMENTE. Seja preciso.
"""


def call_gemma(task: str, context: str = "") -> str:
    """Call local Gemma to create plan."""
    
    user_prompt = f"""Tarefa solicitada: {task}

{f"Contexto adicional:{chr(10)}{context}" if context else ""}

Crie o plano de execução seguindo o formato especificado."""
    
    body = json.dumps({
        "model": "gemma",
        "messages": [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.3,
    }).encode()
    
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    print("🧠 Consultando Gemma 4...")
    
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    
    return (reasoning + "\n" + content).strip()


def extract_plan(response: str) -> str:
    """Extract markdown plan from response."""
    # Try to find markdown block first
    if "# Tarefa:" in response:
        # Find start of plan
        start = response.find("# Tarefa:")
        return response[start:]
    
    # Clean up if wrapped in code block
    if response.startswith("```markdown"):
        lines = response.split("\n")
        return "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    
    return response


def check_server() -> bool:
    """Check if llama-server is running."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 Planner - Cria planos para o Copilot executar",
        epilog="Depois de criar o plano, digite /plan no chat do Copilot"
    )
    parser.add_argument("task", help="Descrição da tarefa")
    parser.add_argument("--context", "-c", help="Contexto adicional (texto ou @arquivo)")
    parser.add_argument("--port", "-p", type=int, default=8090, help="Porta do servidor")
    args = parser.parse_args()
    
    global SERVER_URL
    SERVER_URL = f"http://127.0.0.1:{args.port}"
    
    print("=" * 60)
    print("📋 GEMMA 4 PLANNER")
    print("=" * 60)
    print(f"Tarefa: {args.task}")
    print()
    
    # Check server
    if not check_server():
        print("❌ llama-server não está rodando!")
        print(f"   Inicie em: {SERVER_URL}")
        sys.exit(1)
    
    # Load context if file reference
    context = ""
    if args.context:
        if args.context.startswith("@"):
            ctx_file = Path(args.context[1:])
            if ctx_file.exists():
                context = ctx_file.read_text(encoding="utf-8")
            else:
                print(f"⚠️ Arquivo de contexto não encontrado: {ctx_file}")
        else:
            context = args.context
    
    # Generate plan
    try:
        response = call_gemma(args.task, context)
        plan = extract_plan(response)
    except Exception as e:
        print(f"❌ Erro ao gerar plano: {e}")
        sys.exit(1)
    
    # Add metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""<!-- 
Plano gerado por Gemma 4
Data: {timestamp}
Tarefa original: {args.task}
-->

"""
    
    full_plan = header + plan
    
    # Save plan
    PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLAN_FILE.write_text(full_plan, encoding="utf-8")
    
    print()
    print("=" * 60)
    print("✅ PLANO CRIADO!")
    print("=" * 60)
    print(f"📄 Arquivo: {PLAN_FILE}")
    print()
    print("Próximo passo:")
    print("  1. Abra o chat do Copilot no VS Code")
    print("  2. Digite: /plan")
    print("  3. O Copilot vai ler e executar o plano")
    print()
    print("─" * 60)
    print("Preview do plano:")
    print("─" * 60)
    print(plan[:1000])
    if len(plan) > 1000:
        print("... [truncado]")


if __name__ == "__main__":
    main()
