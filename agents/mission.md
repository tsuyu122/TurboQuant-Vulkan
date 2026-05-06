# TurboQuant-Vulkan — Documento Técnico Completo

## Repositório

GitHub:
https://github.com/tsuyu122/TurboQuant-Vulkan

O modelo deve ler o README diretamente do repositório ou da pasta local sempre que possível.
Este documento serve como guia conceitual e técnico complementar.

---

# 1. VISÃO GERAL DO PROJETO

TurboQuant-Vulkan é um projeto focado em desenvolver técnicas avançadas de compressão e quantização de KV cache para modelos de linguagem, com ênfase em:

* Execução eficiente em Vulkan
* Otimização para GPUs AMD
* Uso local em hardware limitado
* Suporte a contextos longos
* Melhor equilíbrio entre qualidade, memória e desempenho

O projeto não busca apenas replicar quantizações existentes (Q4, Q5, Q8), mas criar uma nova família de técnicas (TQ) com melhorias reais.

---

# 2. FOCO TÉCNICO

## 2.1 Plataforma

* GPU alvo: AMD
* Backend: Vulkan
* Stack: semelhante a ggml / llama.cpp
* Ambiente: inferência local

## 2.2 Problema central

O KV cache cresce proporcionalmente ao tamanho do contexto e se torna o principal consumidor de VRAM.

Objetivo:

* Reduzir drasticamente o tamanho do KV cache
* Minimizar impacto na qualidade
* Manter ou melhorar desempenho

## 2.3 Meta ideal

Um sistema que consiga:

* Usar significativamente menos VRAM que FP16 / Q4 / Q8
* Manter qualidade próxima ao baseline
* Manter ou aumentar tokens/s
* Escalar bem para contextos longos
* Funcionar bem em GPUs AMD via Vulkan

---

# 3. ESTRUTURA DOS MODELOS TQ

## 3.1 TQ2_0

### Definição

Modo de compressão agressiva.

### Objetivo

* Minimizar uso de memória
* Explorar limite inferior de qualidade aceitável

### Características

* VRAM muito baixa
* Qualidade reduzida
* Uso recomendado como componente em sistemas híbridos

---

## 3.2 TQ3_0

### Definição

Modo base de alta qualidade da família TQ.

### Objetivo

* Servir como baseline de qualidade
* Melhorar eficiência comparado a Q4/Q8

### Características

* Alta retenção semântica
* Boa performance
* Base para comparação

---

## 3.3 TQ3_1

### Definição

Modo híbrido estático.

### Estrutura

* K em TQ3_0
* V em TQ2_0

### Objetivo

* Reduzir memória sem perder muita qualidade

### Características

* Excelente trade-off
* VRAM reduzida
* Qualidade ainda alta
* Uso prático eficiente

---

## 3.4 TQ3_2

### Definição correta

Evolução conceitual do TQ3_1.

### Regras fundamentais

* NÃO alterar estrutura de armazenamento
* K permanece em TQ3_0
* V permanece em TQ2_0
* Não criar novos formatos de KV
* Não aumentar VRAM significativamente

### Objetivo real

Melhorar qualidade usando COMPUTE, não armazenamento.

### O que deve fazer

* Aplicar correções durante decode
* Reduzir erro de quantização
* Melhorar coerência sem aumentar custo

### O que não pode ser

* Reempacotamento com mais bits
* Multiplicador arbitrário sem base
* Repetição do TQ3_1
* Mudança que aumente memória

### Resultado esperado

* Qualidade maior que TQ3_1
* VRAM aproximadamente igual ao TQ3_1
* Performance próxima do TQ3_1
* Melhor comportamento em contexto longo

---

# 4. MÉTRICAS DE AVALIAÇÃO

## 4.1 Performance

* Tokens por segundo
* Latência

## 4.2 Memória

* Uso de VRAM
* KV cache size
* Buffers adicionais

## 4.3 Qualidade

* Coerência
* Raciocínio
* Retenção de contexto
* Estabilidade

## 4.4 Contexto longo

Testes em:

* 32k
* 64k
* 128k+
* Máximo possível

---

# 5. RESULTADOS ESPERADOS

| Modelo | Qualidade        | VRAM           | Performance |
| ------ | ---------------- | -------------- | ----------- |
| TQ2_0  | Baixa            | Muito baixa    | Alta        |
| TQ3_0  | Alta             | Média          | Boa         |
| TQ3_1  | Boa              | Baixa          | Boa         |
| TQ3_2  | Melhor que TQ3_1 | Igual ao TQ3_1 | Igual       |

---

# 6. FLUXO DE TRABALHO

1. Ler este documento
2. Ler README do projeto
3. Entender diferenças entre TQ
4. Testar TQ3_2
5. Diagnosticar problemas
6. Corrigir mantendo conceito
7. Validar métricas
8. Projetar TQ3_3

---

# 7. TQ3_3 — REQUISITOS PARA SER REVOLUCIONÁRIO

## Objetivo

Criar algo que seja uma contribuição real ao campo.

## Deve cumprir pelo menos um dos seguintes:

### 1. Compute-aware quantization

O sistema otimiza o resultado final do cálculo, não apenas o valor armazenado.

### 2. Estrutura adaptativa

Tratamento diferente baseado na importância estrutural dos dados.

### 3. Redução de erro acumulado

Minimizar degradação ao longo do contexto.

### 4. Eficiência de bandwidth

Reduzir acesso à memória e melhorar locality.

### 5. Integração com attention

Otimizar KV baseado no uso real na attention.

## Não permitido

* Pequenas variações de TQ3_2
* Ajustes de constante
* Mudanças superficiais

## Critérios de sucesso

* Qualidade >= TQ3_1
* VRAM <= TQ3_1
* Performance >= TQ3_1
* Melhoria perceptível

---

# 8. TQ3_4 — MODELO LIVRE

## Condição

Só pode ser iniciado após TQ3_3 estar completo e validado.

## Regra principal

Pode fazer qualquer coisa, desde que:

* NÃO copie fundamentos dos modelos anteriores

## Liberdade total

Pode:

* mudar representação
* abandonar quantização tradicional
* criar sistemas híbridos
* explorar novas ideias

## Restrições

* Não repetir TQ2, TQ3, TQ3_1 ou TQ3_2
* Não fazer variações pequenas

## Objetivo

Explorar espaço completamente novo.

---

# 9. REGRAS GERAIS

* Não criar inovação falsa
* Não esconder custo de memória
* Não degradar performance sem justificativa
* Não validar sem benchmark real
* Manter foco em AMD + Vulkan + contexto longo

---

# 10. MEMÓRIA PERMANENTE

Esta seção deve ser atualizada continuamente.
Ela é lida a cada execução e serve como estado persistente do sistema.

## Projeto

* Caminho:
* Repositório: https://github.com/tsuyu122/TurboQuant-Vulkan
* Estado atual:
* Branch atual:
* Commit atual:

## Setup

* GPU:
* Driver:
* Backend (Vulkan, etc):
* Modelo usado:
* Sistema operacional:

## Estado dos modelos

* TQ2_0:
* TQ3_0:
* TQ3_1:
* TQ3_2:
* TQ3_3:

## Problemas atuais

*
*
*

## Últimos resultados de benchmark

*
*
*

## Próximas ações

*
*
*

## Ideias futuras

*
*
*

## Instruções para o modelo executor

## (O que deve ser passado para o modelo que implementa código)

*
*
*

## Observações importantes

*
*
*

---

# 11. ANOTAÇÕES LIVRES (LOG DE PESQUISA)

Este campo é para anotações contínuas e raciocínio acumulado.
Pode conter hipóteses, testes, erros, ideias, decisões e descobertas.

Use como um diário técnico.

Formato recomendado:

* Data:
* Contexto:
* O que foi testado:
* Resultado:
* Interpretação:
* Próximo passo:

Regras:

* NÃO apagar histórico antigo
* Sempre adicionar novas entradas
* Ser claro e técnico
* Evitar anotações vagas
* Priorizar observações úteis para evolução do projeto
