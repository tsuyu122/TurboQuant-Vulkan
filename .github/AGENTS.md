# Available Agents for TurboQuant-Vulkan

## TurboQuant Executor

**Name:** `turbo-executor`

**Description:** Agente especializado em executar planos do sistema TurboQuant. Lê `agents/plan.md`, executa os passos, e salva resultados em `agents/result.md`.

**Invocation:** Use `@turbo-executor` no chat ou diga "executa o plano"

**Capabilities:**
- Executa passos de `agents/plan.md` sequencialmente
- Cria/edita arquivos de código
- Roda comandos no terminal
- Marca passos como completos `[x]`
- Gera relatório em `agents/result.md`

**Workflow:**
1. Lê `agents/plan.md`
2. Para cada `- [ ]`, executa a ação
3. Marca como `- [x]` após sucesso
4. Salva resultado em `agents/result.md`
5. Se encontrar erros, tenta corrigir (máx 2 tentativas)

---

## TurboQuant Planner

**Name:** `turbo-planner`

**Description:** Alias para rodar o planejador Gemma local.

**Invocation:** Use `@turbo-planner "tarefa"` ou rode `py agents/planner.py "tarefa"`

---

## Explore (Default)

**Name:** `Explore`

**Description:** Agente padrão de exploração de código. Útil para pesquisar no workspace.
