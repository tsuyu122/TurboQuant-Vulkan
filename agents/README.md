# 🤖 Sistema de Agentes TurboQuant

Sistema 100% autônomo e **GRATUITO** que usa Gemma local para planejar e Copilot para executar.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL LOOP (full_loop.py)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Gemma lê task.md + brain.md + memory.md                 │
│                    ↓                                        │
│  2. Gemma cria plan.md                                      │
│                    ↓                                        │
│  3. Execução (ordem de prioridade):                         │
│     ┌─────────────────────────────────────────────┐         │
│     │ a) LOCAL - criar arquivos, comandos simples │         │
│     │            ↓ (se não conseguir)             │         │
│     │ b) CLI - gh copilot (se tiver capacidade)   │         │
│     │            ↓ (se não disponível)            │         │
│     │ c) AUTO_TYPER - automação física no VS Code │         │
│     └─────────────────────────────────────────────┘         │
│                    ↓                                        │
│  4. Resultado salvo em result.md                            │
│                    ↓                                        │
│  5. Gemma avalia e atualiza memory.md                       │
│                    ↓                                        │
│  6. Loop até tarefa completa                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Métodos de Execução

### 1. Execução Local (Preferida)
- Cria/edita arquivos diretamente
- Roda comandos shell simples
- Mais rápido, sem dependências externas

### 2. GitHub Copilot CLI (Se disponível)
- Usa `gh copilot` se disponível E tiver capacidade de execução
- **Nota**: Atualmente o CLI só faz `suggest` e `explain`
- Quando/se ganhar capacidade de execução, será usado automaticamente

### 3. Automação Física (Fallback)
- Usa `pyautogui` para simular digitação no VS Code
- Abre chat do Copilot (Ctrl+Alt+I)
- Cola mensagem e envia
- ⚠️ Não mexa no mouse/teclado durante execução!

## Arquivos do Sistema

| Arquivo | Descrição |
|---------|-----------|
| `brain.md` | Instruções persistentes do Gemma |
| `memory.md` | Histórico comprimido |
| `task.md` | Tarefa atual |
| `plan.md` | Plano gerado pelo Gemma |
| `result.md` | Resultado da execução |
| `turboquant_context.md` | Contexto técnico do projeto |

## Scripts Principais

| Script | Uso |
|--------|-----|
| `full_loop.py` | **Loop completo autônomo** (usar este!) |
| `daemon.py` | Daemon original |
| `planner.py` | Só gera plano |
| `auto_typer.py` | Automação física |
| `test_system.py` | Testes do sistema |

## Uso Rápido

```bash
# 1. Defina tarefa
echo "Criar arquivo test.py que imprime hello" > agents/task.md

# 2. Rode o loop
py agents/full_loop.py

# 3. Ou modo watch (monitora continuamente)
py agents/full_loop.py --watch

# 4. Ou defina tarefa direto
py agents/full_loop.py --task "criar um script que faz X"
```

## Opções

```bash
py agents/full_loop.py --help

  --watch, -w     Monitora task.md continuamente
  --task, -t      Define tarefa direto na linha de comando
  --no-auto       Não usa automação física (só local + CLI)
  --interval N    Intervalo em watch mode (segundos)
```

## Requisitos

- Python 3.8+
- llama-server rodando na porta 8090 (Gemma)
- pyautogui + pyperclip (para automação física)
- VS Code aberto (para automação física)

## Instalação

```bash
py -m pip install pyautogui pyperclip
```

## Integração com VS Code

Quando a automação física é usada, ou manualmente:

1. No chat do Copilot, diga: `executa o plano` ou `@turbo-executor`
2. Ou use o prompt: `/plan`
3. O resultado será salvo em `agents/result.md`

---

## Fluxo de Decisão

```
Tarefa recebida
      │
      ▼
┌─────────────────┐
│ Execução Local? │──Sim──▶ Executa localmente ──▶ ✓ Feito
└─────────────────┘
      │ Não
      ▼
┌─────────────────┐
│ CLI disponível  │──Sim──▶ Executa via CLI ──▶ ✓ Feito
│ com execução?   │
└─────────────────┘
      │ Não
      ▼
┌─────────────────┐
│ Auto-typer?     │──Sim──▶ Automação física ──▶ Aguarda result.md
└─────────────────┘
      │ Não (--no-auto)
      ▼
   ⚠️ Não executado
```
