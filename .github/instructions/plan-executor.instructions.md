---
description: "Detecta quando o usuário quer executar planos do Gemma. Use quando: executa plano, roda plano, plan, aplica plano"
applyTo: "**"
---

# Instruções para Execução de Planos do Gemma

Quando o usuário disser algo como:
- "executa o plano"
- "roda o plano"
- "aplica o plano"
- "executa"
- "plan"

**SEMPRE** verifique primeiro se existe o arquivo `agents/plan.md`.

Se existir:
1. Leia o conteúdo completo de `agents/plan.md`
2. Execute TODOS os passos marcados como `- [ ]`
3. Marque cada passo como `- [x]` após completar
4. Crie `agents/result.md` com o resultado da execução

Se não existir:
- Informe que não há plano pendente
- Sugira rodar: `py agents/planner.py "descrição da tarefa"`
