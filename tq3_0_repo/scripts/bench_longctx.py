#!/usr/bin/env python3
# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
# -*- coding: utf-8 -*-
"""
bench_longctx.py
Teste de contexto longo: f16 / tq3_0 / tq2_0 x 2 perguntas longas x 1 sessao.
Gera grafico de linhas: Q1->Q2->Q3->Q4->Q5->QL1->QL2 para os 3 modos.
"""
import json, time, os, subprocess, sys, datetime
import urllib.request, urllib.error

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE   = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(BASE, "llama_src", "build_vulkan", "bin", "Release", "llama-server.exe")
MODEL  = os.path.join(BASE, "models", "google_gemma-4-26B-A4B-it-Q4_K_M.gguf")
STDE   = os.path.join(os.environ.get("TEMP", BASE), "bench_longctx_stderr.txt")
OUT    = os.path.join(BASE, "bench_longctx_responses.json")
SCORES_JSON = os.path.join(BASE, "bench_scores.json")
CHART  = os.path.join(BASE, "bench_linechart.png")

HOST, PORT = "127.0.0.1", 8099
CHAT_URL   = f"http://{HOST}:{PORT}/v1/chat/completions"
HEALTH_URL = f"http://{HOST}:{PORT}/health"

NGL     = 30
CTX     = 16384
MAX_TOK = 1500
TEMP    = 0.05

KV_TYPES = ["f16", "tq3_0", "tq2_0"]

# =============================================================================
# DOCUMENTO LONGO (~2200 tokens)
# =============================================================================
DOC_LONG = """\
RELATORIO TECNICO - ESTACAO METEOROLOGICA ALFA-7341
Dra. Mariana Fontes - Instituto de Geofisica de Lisboa - 1987-03-14
VERSAO COMPLETA - CLASSIFICACAO: USO INTERNO

=========================================================================
SECCAO 1 - IDENTIFICACAO
=========================================================================
Codigo da estacao        : ALFA-7341
Localizacao              : 38.7169 N, 9.1395 W  (Tapada da Ajuda, Lisboa)
Altitude                 : 77.4 metros acima do nivel medio do mar
Data de instalacao       : 14 de Marco de 1987
Responsavel cientifica   : Dra. Mariana Fontes  (n. funcional MF-3829)
Frequencia de amostragem : 96 leituras/dia (a cada 15 minutos)
Codigo de calibracao     : CAL-1987-03-MF

=========================================================================
SECCAO 2 - TEMPERATURAS (graus C)  - semana 7 a 13 Marco 1987
=========================================================================
Dia       Min     Max     Media   Amplitude
07 Mar   12.3    19.8    16.05    7.5
08 Mar   11.7    21.4    16.55    9.7
09 Mar   10.9    23.1    17.00   12.2
10 Mar   13.2    24.7    18.95   11.5
11 Mar   14.1    26.3    20.20   12.2
12 Mar   13.8    25.9    19.85   12.1
13 Mar   12.6    27.4    20.00   14.8

Temperatura MAXIMA absoluta : 27.4 C  (13-Mar-1987)
Temperatura MINIMA absoluta : 10.9 C  (09-Mar-1987)
Media semanal das maximas   : 24.09 C
Media semanal das minimas   : 12.66 C
Amplitude media diaria      : 11.43 C
Desvio padrao da maxima     :  2.84 C

=========================================================================
SECCAO 3 - PRESSAO ATMOSFERICA (hPa)
=========================================================================
Dia       06h     12h     18h     24h
07 Mar   1018.2  1016.7  1015.3  1014.9
08 Mar   1013.1  1011.4  1010.8  1010.2
09 Mar   1009.7  1008.3  1007.1  1006.4
10 Mar   1005.9  1007.2  1009.4  1011.8
11 Mar   1013.3  1015.1  1016.0  1017.2
12 Mar   1018.6  1019.4  1019.9  1020.1
13 Mar   1020.3  1021.2  1022.0  1022.7

Pressao maxima : 1022.7 hPa  (13-Mar 24h)
Pressao minima : 1006.4 hPa  (09-Mar 24h)
Variacao total :   16.3 hPa
Gradiente medio de subida (10-13 Mar) : +4.63 hPa/dia

=========================================================================
SECCAO 4 - PRECIPITACAO E HUMIDADE
=========================================================================
Dia       Precip (mm)   Humidade Rel. Media (%)   Ponto de Orvalho (C)
07 Mar        0.0              68                         9.6
08 Mar        1.2              72                        10.2
09 Mar        8.7              83                        11.9
10 Mar        3.4              78                        11.1
11 Mar        0.0              62                         9.8
12 Mar        0.0              58                         8.7
13 Mar        0.0              55                         8.3

Precipitacao total semanal : 13.3 mm
Dias com precipitacao      : 3  (08, 09 e 10 de Marco)
Humidade relativa maxima   : 83%  (09-Mar)
Humidade relativa minima   : 55%  (13-Mar)

=========================================================================
SECCAO 5 - FORMULA DE INDICE DE CONFORTO TERMICO (ICT)
=========================================================================
Formula propria da Dra. Fontes (1987):

    ICT = (T_max  x  P_min) / (H_med  x  10)

Onde:
  T_max = temperatura maxima do dia (C)
  P_min = pressao minima registada nesse dia (valor mais baixo dos 4 horarios)
  H_med = humidade relativa media do dia (%)

Exemplo verificado para 09-Mar:
  ICT = (23.1 x 1006.4) / (83 x 10)
      = 23247.84 / 830
      = 28.01

=========================================================================
SECCAO 6 - EQUIPAMENTOS
=========================================================================
Termometro PT100    - precisao +/- 0.05 C   - serie PT-19870314-A
Barometro Fuess 200 - precisao +/- 0.1 hPa  - serie BFU-7631
Higrometro cap.     - precisao +/- 1.5%     - serie HIG-0047
Pluviometro         - resolucao 0.2 mm      - serie PLU-2281
Ultima calibracao   : 01 de Marco de 1987  (protocolo CAL-1987-03-MF)
Proxima calibracao  : 01 de Setembro de 1987

=========================================================================
SECCAO 7 - NOTAS DA DRA. FONTES
=========================================================================
"O periodo de 9-10 de Marco caracterizou-se por uma perturbacao atlantica
de intensidade moderada, responsavel pela queda de pressao ate 1006.4 hPa
e pelos 8.7 mm de precipitacao no dia 9. A inversao termica de 27.4 C no
dia 13 foi inesperada para a epoca.

O codigo de acesso a base de dados desta estacao e: BETA-5618.
O codigo de arquivo e: ARQ-1987-MF-07."

=========================================================================
SECCAO 8 - LEITURAS HORARIAS DE TEMPERATURA (graus C, 4 leituras/dia)
=========================================================================
Dia       06h     12h     18h     24h
07 Mar   12.3    19.8    17.2    13.1
08 Mar   11.7    21.4    18.9    13.0
09 Mar   10.9    23.1    20.4    12.1
10 Mar   13.2    24.7    22.1    14.8
11 Mar   14.1    26.3    23.8    15.2
12 Mar   13.8    25.9    23.1    14.9
13 Mar   12.6    27.4    24.2    14.3

Taxa de arrefecimento nocturno media : 1.8 C/hora (18h ate 06h do dia seguinte)
Taxa de aquecimento matutino media   : 2.1 C/hora (06h-12h)
Nota: leituras as 06h correspondem ao minimo diario; leituras as 12h ao maximo.

=========================================================================
SECCAO 9 - COMPARACAO COM ESTACAO VIZINHA GAMA-2891
=========================================================================
Estacao GAMA-2891 (Cascais, 12 km a oeste de ALFA-7341)
Altitude              : 15 metros acima do nivel medio do mar
Responsavel tecnico   : Dr. Rui Andrade (numero funcional RA-4421)
Codigo de acesso BD   : GAMA-ACC-2891
Diferenca de altitude : 62.4 metros (ALFA-7341 e mais elevada)

Pressao atmosferica P_min GAMA-2891 (hPa) - mesma semana:
Dia       P_min ALFA   P_min GAMA   Diferenca (ALFA - GAMA)
07 Mar    1014.9       1015.7       -0.8 hPa
08 Mar    1010.2       1011.1       -0.9 hPa
09 Mar    1006.4       1007.1       -0.7 hPa
10 Mar    1005.9       1006.8       -0.9 hPa
11 Mar    1013.3       1014.1       -0.8 hPa
12 Mar    1018.6       1019.3       -0.7 hPa
13 Mar    1020.3       1021.0       -0.7 hPa

Nota: ALFA-7341 regista consistentemente ~0.79 hPa inferior a GAMA-2891.
Gradiente barometrico observado : -0.79 hPa / 62.4 m = -0.0127 hPa/m
Gradiente teorico ICAO          : -0.0125 hPa/m a 15 graus C

Temperatura maxima T_max (graus C) - comparacao ALFA vs GAMA:
Dia       T_max ALFA   T_max GAMA   Diferenca
07 a 13   +1.4 C superior em todos os dias (media +1.4 C)
Causa provavel: efeito de abrigo da vegetacao na Tapada da Ajuda.

=========================================================================
SECCAO 10 - HISTORICO CLIMATICO - MARCO 1985 E 1986
=========================================================================
Estacao ALFA-7341 - registos de Marco dos anos anteriores.

MARCO 1985:
  T_max absoluta      : 21.4 C  (22-Mar-1985)
  T_min absoluta      :  8.3 C  (04-Mar-1985)
  T_max media mensal  : 17.8 C
  Precipitacao total  : 45.2 mm
  Dias com chuva      : 11
  Pressao media       : 1016.4 hPa
  Pressao minima      : 1002.7 hPa  (09-Mar-1985, depressao atlantica)

MARCO 1986:
  T_max absoluta      : 23.8 C  (29-Mar-1986)
  T_min absoluta      :  7.1 C  (02-Mar-1986)
  T_max media mensal  : 19.3 C
  Precipitacao total  : 28.7 mm
  Dias com chuva      : 7
  Pressao media       : 1018.2 hPa
  Pressao minima      : 1008.3 hPa  (17-Mar-1986)

COMPARACAO 1987 vs MEDIA 1985-1986:
  Media das T_max absolutas 1985-86 : (21.4 + 23.8) / 2 = 22.6 C
  T_max absoluta 1987               : 27.4 C
  Anomalia positiva                 : +4.8 C  (RECORDE DAS 3 SERIES)
  Precipitacao total 1985-86 media  : (45.2 + 28.7) / 2 = 36.95 mm/mes
  Precipitacao 1987 (7 dias)        : 13.3 mm  (periodo parcial)

=========================================================================
SECCAO 11 - LOG DE CALIBRACAO E VISITAS TECNICAS
=========================================================================
Data        Tecnico   Instrumento   Operacao realizada
----------  --------  ------------  ----------------------------------------
01-Mar-87   MF-3829   PT100         Correcao de derivacao +0.02 C
01-Mar-87   MF-3829   Fuess 200     Ajuste de zero (pressao absoluta)
01-Mar-87   MF-3829   HIG-0047      Calibracao completa vs padrao WMO
01-Mar-87   MF-3829   PLU-2281      Substituicao de boia de nivel
06-Mar-87   MF-3829   Todos         Verificacao pre-periodo de observacao
14-Mar-87   MF-3829   Todos         Instalacao oficial da estacao

Codigo do protocolo actual     : CAL-1987-03-MF
Proxima calibracao completa    : 01 de Setembro de 1987
Codigo proximo protocolo (prev): CAL-SET-1987

Historico:
  CAL-1986-09-MF  01-Set-1986  (ultima antes de Marco 1987)
  CAL-1987-03-MF  01-Mar-1987  (actual)

=========================================================================
SECCAO 12 - REGISTOS ADMINISTRATIVOS DO PROJECTO
=========================================================================
Instituto de Geofisica de Lisboa
Departamento de Meteorologia de Superficie
Projecto       : REDE-METEO-LISBOA-1987
Codigo FCT     : FCT-87-MET-034
Financiamento  : Fundacao para a Ciencia e Tecnologia
Duracao        : Janeiro 1987 a Dezembro 1989
Coordenadora   : Dra. Mariana Fontes (MF-3829)
Co-investigador: Dr. Rui Andrade (RA-4421)
Estacoes       : ALFA-7341 (Lisboa / Tapada da Ajuda)
                 GAMA-2891 (Cascais / Boca do Inferno)

Arquivo morto dos registos      : ARQ-1987-MF-07
Codigo de acesso base de dados  : BETA-5618
Contacto de manutencao          : ext. 2847 (servico 24h)
Suporte de dados                : disco magnetico 5.25 pol. (backups semanais)

=========================================================================
FIM DO RELATORIO COMPLETO - ALFA-7341 - 14-Mar-1987
=========================================================================
"""

# =============================================================================
# 2 PERGUNTAS LONGAS
# =============================================================================
QUESTIONS_LONG = [
    {
        "label": "QL1 ICT Completo + Cross-Reference",
        "prompt": (
            "Analise detalhada com base no relatorio tecnico completo que memorizaste:\n\n"

            "PARTE A — Calculo ICT para os 7 dias:\n"
            "Usando a formula ICT = (T_max x P_min) / (H_med x 10) da Dra. Fontes,\n"
            "calcula o ICT para cada dia de 07 a 13 de Marco mostrando passo a passo:\n"
            "  - T_max do dia (tabela Seccao 2)\n"
            "  - P_min do dia: valor MINIMO entre as 4 leituras de pressao da Seccao 3\n"
            "  - H_med do dia (Seccao 4)\n"
            "  - Calculo completo: numerador, denominador, resultado arredondado a 2 casas\n"
            "Apresenta os 7 resultados numa tabela resumo no final dessa parte.\n\n"

            "PARTE B — Ranking e diferencas:\n"
            "1. Lista todos os 7 dias por ordem DECRESCENTE de ICT.\n"
            "2. Qual o dia com ICT mais elevado e qual o mais baixo?\n"
            "3. Diferenca absoluta: ICT_max - ICT_min = ?\n"
            "4. Percentagem de superioridade do ICT maximo sobre o minimo:\n"
            "   Formula: ((ICT_max - ICT_min) / ICT_min) x 100 = ?\n"
            "   Arredonda a 2 casas decimais.\n\n"

            "PARTE C — Cross-reference com GAMA-2891 nos dias de chuva:\n"
            "Para os 3 dias com precipitacao (08, 09, 10 de Marco):\n"
            "1. Qual era o P_min de ALFA-7341 em cada um desses dias?\n"
            "2. Qual era o P_min de GAMA-2891 nesses mesmos dias (Seccao 9)?\n"
            "3. Em qual dos 3 dias a diferenca de pressao (ALFA - GAMA) foi mais negativa?\n\n"

            "PARTE D — Impacto da incerteza do barometro no ICT do dia 13:\n"
            "O Fuess 200 tem precisao de +/- 0.1 hPa.\n"
            "Para o dia 13 de Marco (ICT mais alto), qual seria o ICT resultante se\n"
            "a pressao medida estivesse subestimada pelo erro maximo (+0.1 hPa)?\n"
            "Ou seja: recalcula o ICT com P_min = 1020.3 + 0.1 = 1020.4 hPa.\n"
            "De quantas unidades mudaria o ICT? E se a pressao estivesse sobreavaliada\n"
            "(-0.1 hPa)? Mostra ambos os calculos."
        ),
    },
    {
        "label": "QL2 Auditoria Tecnica Total",
        "prompt": (
            "Auditoria tecnica completa do relatorio ALFA-7341.\n"
            "Responde exclusivamente da tua memoria desta conversa:\n\n"

            "PARTE A — Inventario completo de equipamentos (Seccao 6):\n"
            "Para cada um dos 4 instrumentos de medicao, indica:\n"
            "  - Nome do instrumento\n"
            "  - Numero de serie EXATO (tal como aparece no relatorio)\n"
            "  - Precisao ou resolucao\n\n"

            "PARTE B — Inventario completo de codigos:\n"
            "Lista TODOS os codigos alfanumericos mencionados no relatorio:\n"
            "  1. Codigo da estacao\n"
            "  2. Codigo de acesso a base de dados (da estacao ALFA-7341)\n"
            "  3. Codigo de arquivo dos registos\n"
            "  4. Codigo do protocolo de calibracao actual\n"
            "  5. Numero funcional da responsavel cientifica\n"
            "  6. Numero funcional do co-investigador da estacao GAMA-2891\n\n"

            "PARTE C — Reproducao da nota da Seccao 7:\n"
            "Reproduz o mais fielmente possivel o texto completo da nota da Dra. Fontes\n"
            "(comeca com 'O periodo de 9-10 de Marco...').\n"
            "Inclui obrigatoriamente os dois codigos mencionados no texto.\n\n"

            "PARTE D — Comparacao historica (Seccao 10):\n"
            "  1. T_max absoluta de Marco de 1985 e Marco de 1986?\n"
            "  2. A T_max de 27.4 C de 13-Mar-1987 e recorde das 3 series?\n"
            "  3. Precipitacao total de Marco de 1985 e Marco de 1986 (em mm)?\n"
            "  4. Qual a anomalia de temperatura entre 1987 e a media de 1985-1986?\n\n"

            "PARTE E — Sintese de gestao e codigo de projecto:\n"
            "  1. Quando foi a ultima calibracao completa antes deste periodo?\n"
            "  2. Quando esta programada a proxima calibracao?\n"
            "  3. Qual o codigo de financiamento FCT deste projecto?\n"
            "  4. Quantos dias de dados foram recolhidos ANTES da instalacao oficial?"
        ),
    },
]

# =============================================================================
# CRITERIOS DE SCORING
# =============================================================================
CRITERIOS_LONG = {
    "QL1": [
        ["29.55"],                        # ICT 07 Mar
        ["28.01"],                        # ICT 09 Mar (dado no doc, ancora)
        ["31.85"],                        # ICT 10 Mar
        ["43.00", "43,00", "43.0"],       # ICT 11 Mar
        ["45.49"],                        # ICT 12 Mar
        ["50.83"],                        # ICT 13 Mar
        ["22.82"],                        # diferenca ICTmax - ICTmin
        ["81.47", "81.46", "81,47"],      # percentagem de diferenca
    ],
    "QL2": [
        ["PT-19870314-A"],                # serie termometro PT100
        ["BFU-7631"],                     # serie barometro
        ["HIG-0047"],                     # serie higrometro
        ["PLU-2281"],                     # serie pluviometro
        ["BETA-5618"],                    # codigo acesso BD
        ["ARQ-1987-MF-07"],              # codigo arquivo
        ["CAL-1987-03-MF"],              # codigo calibracao
        ["RA-4421"],                      # funcional co-investigador GAMA-2891
        ["1 de Setembro", "setembro", "Set-1987"],  # proxima calibracao
        ["FCT-87-MET-034"],              # codigo FCT (seccao 12 - hard recall)
    ],
}

Q_IDS_LONG = ["QL1", "QL2"]


# =============================================================================
# GESTAO DO SERVIDOR
# =============================================================================
_proc = None


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True, timeout=8)
    time.sleep(2)


def start_server(kv: str) -> bool:
    global _proc
    kill_all()
    print(f"\n  >> Iniciando  KV={kv} | ngl={NGL} | ctx={CTX}")
    cmd = [SERVER, "-m", MODEL,
           "-ngl", str(NGL), "-c", str(CTX),
           "-ctk", kv, "-ctv", kv,
           "--host", HOST, "--port", str(PORT),
           "--parallel", "1", "--no-warmup",
           "--reasoning-budget", "0"]
    with open(STDE, "w", encoding="utf-8") as fh:
        _proc = subprocess.Popen(cmd, stderr=fh, stdout=subprocess.DEVNULL)
    for _ in range(180):
        time.sleep(1)
        if _proc.poll() is not None:
            print("  FALHOU - processo terminou")
            return False
        try:
            resp = urllib.request.urlopen(HEALTH_URL, timeout=2)
            if resp.status == 200:
                print("  OK - Pronto")
                return True
        except Exception:
            pass
    print("  TIMEOUT aguardando servidor")
    return False


def stop_server():
    global _proc
    if _proc:
        _proc.terminate()
        try: _proc.wait(timeout=12)
        except Exception: _proc.kill()
        _proc = None
    time.sleep(1)


def get_vram_mib() -> float:
    try:
        import re
        txt = open(STDE, encoding="utf-8", errors="ignore").read()
        vals = re.findall(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", txt, re.I)
        if vals: return sum(float(v) for v in vals)
    except Exception:
        pass
    return 0.0


# =============================================================================
# API CHAT
# =============================================================================
def chat(messages: list) -> dict:
    payload = json.dumps({
        "messages": messages,
        "temperature": TEMP,
        "max_tokens": MAX_TOK,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        CHAT_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"    API erro: {e}")
        return {"text": "", "tps": 0.0}
    elapsed = time.time() - t0
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    tim = data.get("timings", {})
    if tim.get("predicted_n") and tim.get("predicted_ms"):
        tps = tim["predicted_n"] / (tim["predicted_ms"] / 1000.0)
    else:
        n   = (data.get("usage") or {}).get("completion_tokens", 0)
        tps = n / max(elapsed, 0.001)
    return {"text": text, "tps": tps}


# =============================================================================
# SESSAO DE BENCHMARK
# =============================================================================
def run_session(kv: str) -> dict:
    messages = [{"role": "user",
                 "content": "Les e memoriza este relatorio tecnico completo:\n\n" + DOC_LONG}]

    print(f"    [intro] enviando documento... ", end="", flush=True)
    r0 = chat(messages)
    print(f"{r0['tps']:.1f} t/s  ({len(r0['text'].split())} palavras resposta)")
    if r0["text"]:
        messages.append({"role": "assistant", "content": r0["text"]})

    qa_pairs = []
    for q in QUESTIONS_LONG:
        print(f"\n    >>> {q['label']}")
        messages.append({"role": "user", "content": q["prompt"]})
        r = chat(messages)
        words = len(r["text"].split())
        print(f"    <<< {r['tps']:.1f} t/s  ({words} palavras)")
        if r["text"]:
            messages.append({"role": "assistant", "content": r["text"]})

        print()
        print("    " + "-" * 64)
        for line in r["text"].replace("\r\n", "\n").split("\n"):
            print(f"    {line}")
        print("    " + "-" * 64)
        print()

        qa_pairs.append({
            "question_label": q["label"],
            "question": q["prompt"],
            "response": r["text"],
            "tps": r["tps"],
            "words": words,
        })

    return {
        "kv": kv,
        "qa": qa_pairs,
        "avg_tps": sum(p["tps"] for p in qa_pairs) / max(len(qa_pairs), 1),
        "timestamp": datetime.datetime.now().isoformat(),
    }


# =============================================================================
# SCORING
# =============================================================================
def score_resposta(resp: str, qid: str) -> tuple:
    criterios = CRITERIOS_LONG.get(qid, [])
    hits = 0
    detalhes = []
    for grupo in criterios:
        ok = any(k.lower() in resp.lower() for k in grupo)
        hits += int(ok)
        detalhes.append((grupo[0], ok))
    return hits, len(criterios), detalhes


def score_session(session: dict) -> dict:
    per_q = {}
    for idx, qa in enumerate(session["qa"]):
        qid = Q_IDS_LONG[idx]
        hits, total, _ = score_resposta(qa["response"], qid)
        per_q[qid] = hits / total if total else 0.0
    return per_q


def imprimir_detalhes(results: dict):
    print("\n" + "=" * 62)
    print("  DETALHES POR CRITERIO")
    print("=" * 62)
    for kv, r in results.items():
        print(f"\n  --- {kv} ---")
        for idx, qa in enumerate(r["session"]["qa"]):
            qid = Q_IDS_LONG[idx]
            hits, total, dets = score_resposta(qa["response"], qid)
            print(f"  {qid}: {hits}/{total} criterios")
            for kw, ok in dets:
                print(f"    {'OK' if ok else '--'} '{kw}'")


# =============================================================================
# GRAFICO DE LINHAS COMBINADO
# =============================================================================
def gerar_linechart(scores_combined: dict):
    """
    scores_combined[kv] = {
        "Q1": float, "Q2": float, ..., "Q5": float,
        "QL1": float, "QL2": float
    }
    Todos os valores sao fraccoes (0-1).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[AVISO] matplotlib nao disponivel.")
        return

    ORDER  = ["Q1", "Q2", "Q3", "Q4", "Q5", "QL1", "QL2"]
    LABELS = [
        "Q1\nRecall\nbásico",
        "Q2\nICT\n7 dias",
        "Q3\nDeltas\ne refs",
        "Q4\nAnálise\ncomposta",
        "Q5\nSíntese\nmemória",
        "QL1\nICT\nlongo",
        "QL2\nAuditoria\ntotal",
    ]

    CORES = {
        "f16":   "#4fc3f7",
        "tq3_0": "#81c784",
        "tq2_0": "#ffb74d",
    }
    MARKERS = {
        "f16":   "o",
        "tq3_0": "s",
        "tq2_0": "^",
    }

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    # Zona de separacao entre testes standard e long-context
    ax.axvspan(4.5, 6.5, color="#3a3a3a", alpha=0.6, zorder=0)
    ax.text(5.5, 103, "contexto longo", ha="center", va="bottom",
            fontsize=8, color="#888", style="italic")
    ax.text(2.0, 103, "teste standard  (5 perguntas encadeadas)",
            ha="center", va="bottom", fontsize=8, color="#888", style="italic")
    ax.axvline(4.5, color="#555", linestyle="--", linewidth=0.8)

    x = np.arange(len(ORDER))

    for kv in KV_TYPES:
        if kv not in scores_combined:
            continue
        sc = scores_combined[kv]
        vals = [sc.get(qid, float("nan")) * 100 for qid in ORDER]
        cor  = CORES.get(kv, "#ccc")
        mk   = MARKERS.get(kv, "o")
        ax.plot(x, vals, color=cor, linewidth=2.2, marker=mk,
                markersize=8, markerfacecolor=cor, markeredgecolor="#1e1e1e",
                markeredgewidth=1.5, label=kv, zorder=3)
        for xi, v in zip(x, vals):
            if not (v != v):  # nao eh NaN
                ax.text(xi, v + 2.5, f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, color=cor, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5, color="white")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", color="white", fontsize=11)
    ax.set_title(
        "Qualidade por Posição no Contexto — f16 / tq3_0 / tq2_0",
        color="white", fontsize=13, fontweight="bold", pad=12
    )
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#444", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(100, color="#555", linestyle="--", linewidth=0.7)
    ax.legend(facecolor="#3d3d3d", edgecolor="#555", labelcolor="white",
              fontsize=10, loc="lower left")

    plt.tight_layout()
    plt.savefig(CHART, dpi=150, bbox_inches="tight", facecolor="#1e1e1e")
    print(f"\n[OK] Grafico guardado: {CHART}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 62)
    print("  BENCH LONG CONTEXT — 3 modos x 2 perguntas longas x 1 sessao")
    print("=" * 62)
    print(f"  Modelo  : {os.path.basename(MODEL)}")
    print(f"  KV tipos: {', '.join(KV_TYPES)}")
    print(f"  CTX={CTX}  NGL={NGL}  Temp={TEMP}  MaxTok={MAX_TOK}")
    print()

    for path, name in [(SERVER, "Servidor"), (MODEL, "Modelo")]:
        if not os.path.exists(path):
            print(f"ERRO: {name} nao encontrado: {path}")
            sys.exit(1)

    long_results = {}

    for kv in KV_TYPES:
        print(f"\n{'='*62}")
        print(f"  KV = {kv.upper()}")
        print(f"{'='*62}")

        ok = start_server(kv)
        vram = get_vram_mib()
        print(f"  VRAM KV cache: {vram:.0f} MiB\n")

        if ok:
            sess = run_session(kv)
        else:
            sess = {"kv": kv, "qa": [], "avg_tps": 0.0,
                    "timestamp": datetime.datetime.now().isoformat()}

        per_q = score_session(sess)

        stop_server()

        long_results[kv] = {
            "vram_mib": vram,
            "session":  sess,
            "per_q":    per_q,
            "avg_tps":  sess["avg_tps"],
            "ok":       ok,
        }

        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(long_results, f, ensure_ascii=False, indent=2)

    # ── Imprimir scores ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {'KV':^8}  {'QL1':^8}  {'QL2':^8}  {'Media':^8}  {'t/s':^7}  {'VRAM':^7}")
    print("  " + "-" * 56)
    for kv, r in long_results.items():
        pq = r["per_q"]
        ql1 = pq.get("QL1", 0) * 100
        ql2 = pq.get("QL2", 0) * 100
        med = (ql1 + ql2) / 2
        print(f"  {kv:^8}  {ql1:^7.1f}%  {ql2:^7.1f}%  {med:^7.1f}%  "
              f"{r['avg_tps']:^6.1f}  {r['vram_mib']:^6.0f}")

    imprimir_detalhes(long_results)

    # ── Combinar com Q1-Q5 de bench_scores.json ───────────────────────────────
    scores_combined = {kv: {} for kv in KV_TYPES}

    if os.path.exists(SCORES_JSON):
        try:
            with open(SCORES_JSON, encoding="utf-8") as f:
                hist = json.load(f)
            for kv in KV_TYPES:
                if kv in hist and "per_q" in hist[kv]:
                    scores_combined[kv].update(hist[kv]["per_q"])
        except Exception as e:
            print(f"[AVISO] Erro a ler {SCORES_JSON}: {e}")
    else:
        print(f"[AVISO] {SCORES_JSON} nao encontrado — Q1-Q5 serao NaN no grafico")

    for kv, r in long_results.items():
        scores_combined[kv].update(r["per_q"])

    # Normaliza: bench_scores.json guarda percentagens (0-100 ou 0-1?)
    # bench_score.py guarda per_q como fraccao (0-1) x 100 no display
    # mas grava no JSON como fraccao. Verificar:
    for kv in KV_TYPES:
        sc = scores_combined.get(kv, {})
        for qid in list(sc.keys()):
            v = sc[qid]
            if v is not None and v > 1.5:
                sc[qid] = v / 100.0  # converter de percentagem para fraccao

    gerar_linechart(scores_combined)

    print(f"\n[OK] Respostas guardadas: {OUT}")
    print("=" * 62)


if __name__ == "__main__":
    main()
