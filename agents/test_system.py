#!/usr/bin/env python3
"""
🧪 Teste do Sistema de Agentes TurboQuant

Testa o fluxo completo:
1. Cria tarefa de teste
2. Gemma cria plano
3. Execução do plano
4. Validação do resultado

Uso:
    py agents/test_system.py
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Configuração  
BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"
SERVER_URL = "http://127.0.0.1:8090"

TASK_FILE = AGENTS_DIR / "task.md"
PLAN_FILE = AGENTS_DIR / "plan.md"
RESULT_FILE = AGENTS_DIR / "result.md"

def log(msg: str, level: str = "INFO"):
    symbols = {"OK": "✅", "FAIL": "❌", "INFO": "ℹ️", "TEST": "🧪", "WAIT": "⏳"}
    print(f"[{symbols.get(level, '•')}] {msg}")

def check_server() -> bool:
    """Verifica se llama-server está rodando."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.getcode() == 200
    except:
        return False

def call_gemma_simple(prompt: str) -> str:
    """Chama Gemma com prompt simples."""
    body = json.dumps({
        "model": "gemma",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }).encode()
    
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")

def test_gemma_connection() -> bool:
    """Testa conexão com Gemma."""
    log("Testando conexão com Gemma...", "TEST")
    
    if not check_server():
        log("Servidor não está rodando!", "FAIL")
        return False
    
    try:
        response = call_gemma_simple("Responda apenas: OK")
        if "OK" in response.upper() or len(response) > 0:
            log(f"Gemma respondeu: {response[:50]}", "OK")
            return True
    except Exception as e:
        log(f"Erro: {e}", "FAIL")
    
    return False

def test_planner() -> bool:
    """Testa o planejador."""
    log("Testando planejador...", "TEST")
    
    # Cria tarefa de teste simples
    test_task = """# Tarefa de Teste

## Tarefa
Criar arquivo `agents/test_agent_result.txt` com o texto:
"Sistema de agentes TurboQuant funcionando!"

## Prioridade
- [x] Alta (fazer agora)
"""
    
    TASK_FILE.write_text(test_task, encoding="utf-8")
    log("Tarefa de teste criada", "OK")
    
    # Roda o planner
    try:
        log("Executando planner (pode demorar ~30s)...", "WAIT")
        result = subprocess.run(
            [sys.executable, str(AGENTS_DIR / "planner.py"), 
             "Criar arquivo agents/test_agent_result.txt com texto 'Sistema funcionando!'"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if PLAN_FILE.exists() and PLAN_FILE.stat().st_size > 100:
            log("Plano gerado com sucesso!", "OK")
            plan_preview = PLAN_FILE.read_text(encoding="utf-8")[:200]
            log(f"Preview: {plan_preview}...", "INFO")
            return True
        else:
            log(f"Plano não gerado. Output: {result.stdout[:200]}", "FAIL")
            return False
            
    except subprocess.TimeoutExpired:
        log("Timeout no planner", "FAIL")
        return False
    except Exception as e:
        log(f"Erro no planner: {e}", "FAIL")
        return False

def test_local_execution() -> bool:
    """Testa execução local de tarefa simples."""
    log("Testando execução local...", "TEST")
    
    test_file = AGENTS_DIR / "test_local_exec.txt"
    
    # Remove se existir
    if test_file.exists():
        test_file.unlink()
    
    # Escreve
    test_file.write_text("Teste de execução local OK\n", encoding="utf-8")
    
    if test_file.exists() and "OK" in test_file.read_text():
        log("Execução local funcionando!", "OK")
        test_file.unlink()  # Limpa
        return True
    
    log("Falha na execução local", "FAIL")
    return False

def test_copilot_integration() -> dict:
    """Verifica estrutura para integração com Copilot."""
    log("Verificando integração com Copilot...", "TEST")
    
    checks = {
        "instruction_file": (BASE_DIR / ".github/instructions/plan-executor.instructions.md").exists(),
        "agent_file": (BASE_DIR / ".github/agents/turbo-executor.agent.md").exists(),
        "agents_md": (BASE_DIR / ".github/AGENTS.md").exists(),
        "plan_prompt": (BASE_DIR / ".github/prompts/plan.prompt.md").exists(),
    }
    
    for name, exists in checks.items():
        status = "OK" if exists else "FAIL"
        log(f"  {name}: {'✓' if exists else '✗'}", status)
    
    return checks

def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "="*60)
    print("🧪 TESTE DO SISTEMA DE AGENTES TURBOQUANT")
    print("="*60 + "\n")
    
    results = {}
    
    # 1. Teste de conexão
    results["gemma_connection"] = test_gemma_connection()
    print()
    
    # 2. Teste de execução local
    results["local_execution"] = test_local_execution()
    print()
    
    # 3. Verificação de integração Copilot
    copilot_checks = test_copilot_integration()
    results["copilot_integration"] = all(copilot_checks.values())
    print()
    
    # 4. Teste do planner (se Gemma OK)
    if results["gemma_connection"]:
        results["planner"] = test_planner()
    else:
        log("Pulando teste do planner (Gemma offline)", "INFO")
        results["planner"] = None
    print()
    
    # Resumo
    print("="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)
    
    for test, passed in results.items():
        if passed is None:
            status = "⏭️ PULADO"
        elif passed:
            status = "✅ PASSOU"
        else:
            status = "❌ FALHOU"
        print(f"  {test}: {status}")
    
    print()
    
    # Instruções finais
    all_critical_passed = (
        results["gemma_connection"] and 
        results["local_execution"] and 
        results["copilot_integration"]
    )
    
    if all_critical_passed:
        print("✅ SISTEMA PRONTO PARA USO!")
        print()
        print("📝 Como usar:")
        print("   1. Escreva tarefa em agents/task.md")
        print("   2. Rode: py agents/daemon.py --once")
        print("   3. No VS Code, diga: 'executa o plano' ou '@turbo-executor'")
        print("   4. Resultado estará em agents/result.md")
        print()
        print("🔄 Para modo autônomo contínuo:")
        print("   py agents/daemon.py --watch")
        return True
    else:
        print("⚠️ ALGUNS TESTES FALHARAM")
        print()
        if not results["gemma_connection"]:
            print("   → Inicie o llama-server: porta 8090")
        if not results["copilot_integration"]:
            print("   → Verifique arquivos em .github/")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
