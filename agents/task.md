# Prompt Final para Execução Noturna

Use os documentos locais disponíveis para entender o projeto e executar o trabalho:
- `agents/mission.md`
- `README.md`
- `agents/brain.md`
- `agents/turboquant_context.md`
- `agents/memory.md`

## Objetivo geral

1. Entender o projeto TurboQuant-Vulkan e o foco em KV cache, Vulkan e GPUs AMD.
2. Priorizar diagnóstico e validação de TQ3_2.
3. Testar o sistema de agentes end-to-end e garantir que o fluxo funcione.
4. Criar benchmarks, medidas e relatório técnico de qualidade/memória/performance.
5. Atualizar `agents/memory.md` com um histórico técnico relevante.
6. Produzir um resultado consistente e documentado ao amanhecer.

## Instruções para o agente

- Leia o README local e o arquivo de missão antes de criar o plano.
- Use o prompt como base para gerar `agents/plan.md`.
- Mantenha `agents/plan.md` em formato estrito de plano com passos claros.
- Se necessário, crie arquivos de teste e geração de resultados.
- Não destrua dados existentes e não altere arquivos fora da pasta `agents/` sem justificativa.

## Critérios de sucesso

- `py agents/autonomous_v3.py --test` deve passar.
- Um backup completo da pasta deve ser gerado.
- O fluxo de criação do plano, execução e avaliação deve estar funcionando.
- Deve ficar pronto para rodar durante a noite com o prompt noturno.