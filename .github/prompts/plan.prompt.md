---
description: "Executa o plano gerado pelo Gemma 4 em agents/plan.md. Use quando quiser que eu execute instruções escritas pelo modelo local."
name: "Executar Plano Local"
argument-hint: "executa o plano do Gemma"
agent: "agent"
---

# Executor de Plano Local

Leia o arquivo `agents/plan.md` no workspace e execute TODAS as instruções contidas nele.

## Regras de Execução

1. **Leia o arquivo completo** `agents/plan.md`
2. Para cada passo marcado como `[ ]` (não feito):
   - Execute a ação indicada
   - Marque como `[x]` quando completar
3. Se encontrar erros:
   - Anote em `agents/result.md`
   - Continue com os próximos passos se possível
4. Ao final:
   - Crie/atualize `agents/result.md` com o resumo da execução
   - Informe o status: ✅ Completo, ⚠️ Parcial, ❌ Falhou

## Formato esperado do plan.md

```markdown
# Tarefa: [descrição]

## Passos

- [ ] Passo 1: criar arquivo X com conteúdo Y
- [ ] Passo 2: editar arquivo Z na linha N
- [ ] Passo 3: rodar comando `abc`

## Contexto

[informações adicionais se houver]
```

## Após executar

Crie um arquivo `agents/result.md` com:
```markdown
# Resultado da Execução

**Status**: ✅ Completo / ⚠️ Parcial / ❌ Falhou
**Timestamp**: [data/hora]

## Passos Executados
- [x] Passo 1: OK
- [x] Passo 2: OK  
- [ ] Passo 3: ERRO - [motivo]

## Observações
[qualquer nota relevante]
```
