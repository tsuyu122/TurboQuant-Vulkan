#!/usr/bin/env python3
# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
"""
Benchmark KV Cache — Recolha de respostas para julgamento manual
Compara: f16 / q4_0 / tq3_0 / tq2_0

Fase 1 (este script): corre inferencia, guarda todas as respostas em JSON
Fase 2 (separado)   : le o JSON, atribui scores, gera grafico

Uso: py bench_quality.py
"""

import json, time, os, subprocess, sys, datetime
import urllib.request, urllib.error

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(BASE, "llama_src", "build_vulkan", "bin", "Release", "llama-server.exe")
MODEL  = os.path.join(BASE, "models", "google_gemma-4-26B-A4B-it-Q4_K_M.gguf")
STDE   = os.path.join(os.environ.get("TEMP", BASE), "bench_quality_stderr.txt")
OUT    = os.path.join(BASE, "bench_quality_responses.json")

HOST, PORT  = "127.0.0.1", 8099
CHAT_URL    = f"http://{HOST}:{PORT}/v1/chat/completions"
HEALTH_URL  = f"http://{HOST}:{PORT}/health"

# ── Parametros ────────────────────────────────────────────────────────────────
NGL      = 30
CTX      = 16384
SESSIONS = 2
MAX_TOK  = 900
TEMP     = 0.05

KV_TYPES = ["tq2_0"]

# ── Documento tecnico (~2500 tokens) ──────────────────────────────────────────
# Enche o KV cache com dados numericos densos.
# As perguntas seguintes exigem recall preciso destes valores.

DOC = """\
RELATORIO TECNICO - ESTACAO METEOROLOGICA ALFA-7341
Dra. Mariana Fontes - Instituto de Geofisica de Lisboa - 1987-03-14

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
Pressao minima :  1006.4 hPa  (09-Mar 24h)
Variacao total :    16.3 hPa
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
FIM DO RELATORIO - Classificacao: USO INTERNO
=========================================================================
"""

# ── 5 Perguntas encadeadas ────────────────────────────────────────────────────
QUESTIONS = [
    {
        "label": "Q1 Recall basico",
        "prompt": (
            "Com base no relatorio que leste:\n"
            "1. Qual o codigo da estacao?\n"
            "2. Nome completo e numero funcional da responsavel cientifica?\n"
            "3. Coordenadas exatas e altitude?\n"
            "4. Temperatura maxima absoluta da semana (dia e valor)?\n"
            "5. Pressao minima absoluta da semana (dia, hora e valor)?\n"
            "6. Quais os dois codigos mencionados nas notas da Dra. Fontes?"
        ),
    },
    {
        "label": "Q2 ICT para todos os 7 dias",
        "prompt": (
            "Usando a formula ICT = (T_max x P_min) / (H_med x 10) que esta no relatorio:\n\n"
            "Calcula o ICT para cada um dos 7 dias (07 a 13 de Marco).\n"
            "Para P_min usa o valor mais baixo entre as 4 leituras diarias de pressao.\n"
            "Mostra o calculo de cada dia passo a passo e arredonda a 2 casas decimais.\n"
            "No final indica qual o dia com ICT mais alto e qual com ICT mais baixo."
        ),
    },
    {
        "label": "Q3 Deltas e referencias cruzadas",
        "prompt": (
            "Usando os ICTs que calculaste nesta conversa:\n\n"
            "a) Qual a diferenca entre o ICT do dia 13 e o ICT do dia 09? (ICT_13 - ICT_09)\n"
            "b) Em que percentagem o ICT do dia 13 e superior ao do dia 09?\n"
            "   Formula: ((ICT_13 - ICT_09) / ICT_09) x 100\n"
            "c) Qual a precipitacao total da semana? Em quantos dias choveu?\n"
            "d) Numero de serie do barometro e a sua precisao."
        ),
    },
    {
        "label": "Q4 Analise composta",
        "prompt": (
            "Responde usando apenas os dados do relatorio original:\n\n"
            "1. Calcula a amplitude termica media dos 3 dias em que choveu "
            "(usa os valores de amplitude da tabela de temperaturas).\n"
            "2. Qual a variacao total de pressao ao longo da semana (P_max - P_min)?\n"
            "3. O relatorio cobre 07 a 13 de Marco. A estacao foi instalada em 14 de Marco. "
            "Quantos dias ANTES da instalacao oficial comecou o registo de dados?\n"
            "4. Repete o codigo de calibracao e o codigo da estacao."
        ),
    },
    {
        "label": "Q5 Sintese total de memoria",
        "prompt": (
            "TESTE FINAL - responde apenas da tua memoria desta conversa:\n\n"
            "1. Codigo da estacao + codigo de acesso BD + codigo de arquivo\n"
            "2. Numero funcional da cientista + numero de serie do termometro PT100\n"
            "3. ICT calculado para o dia 09 de Marco (da nossa conversa)\n"
            "4. Temperatura maxima do dia 11 e pressao as 06h desse dia\n"
            "5. Precipitacao do dia 09 de Marco em mm\n"
            "6. Gradiente medio de subida de pressao entre 10 e 13 de Marco\n"
            "7. Altitude da estacao em metros"
        ),
    },
]

# ── Gestao do servidor ────────────────────────────────────────────────────────
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
            _tail_stderr(20)
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


def _tail_stderr(n=15):
    try:
        lines = open(STDE, encoding="utf-8", errors="ignore").readlines()
        for ln in lines[-n:]: print("    STDERR:", ln.rstrip())
    except Exception:
        pass


def get_vram_mib() -> float:
    try:
        import re
        txt = open(STDE, encoding="utf-8", errors="ignore").read()
        vals = re.findall(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", txt, re.I)
        if vals: return sum(float(v) for v in vals)
    except Exception:
        pass
    return 0.0


# ── API chat ──────────────────────────────────────────────────────────────────
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
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"    API erro: {e}")
        return {"text": "", "tps": 0.0}
    elapsed = time.time() - t0
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    tim  = data.get("timings", {})
    if tim.get("predicted_n") and tim.get("predicted_ms"):
        tps = tim["predicted_n"] / (tim["predicted_ms"] / 1000.0)
    else:
        n   = (data.get("usage") or {}).get("completion_tokens", 0)
        tps = n / max(elapsed, 0.001)
    return {"text": text, "tps": tps}


# ── Sessao de benchmark ───────────────────────────────────────────────────────
def run_session(kv: str, sid: int) -> dict:
    SEP = "=" * 68

    # Injeta o documento como 1a mensagem
    messages = [{"role": "user",
                 "content": "Les e memoriza este relatorio tecnico completo:\n\n" + DOC}]

    print(f"    [intro] enviando documento... ", end="", flush=True)
    r0 = chat(messages)
    print(f"{r0['tps']:.1f} t/s")
    if r0["text"]:
        messages.append({"role": "assistant", "content": r0["text"]})

    qa_pairs = []

    for q in QUESTIONS:
        print(f"\n    >>> {q['label']}")
        messages.append({"role": "user", "content": q["prompt"]})
        r = chat(messages)
        print(f"    <<< {r['tps']:.1f} t/s  ({len(r['text'].split())} palavras)")
        if r["text"]:
            messages.append({"role": "assistant", "content": r["text"]})

        # Imprime a resposta completa, linha a linha
        print()
        print("    " + "-"*64)
        for line in r["text"].replace("\r\n", "\n").split("\n"):
            print(f"    {line}")
        print("    " + "-"*64)
        print()

        qa_pairs.append({
            "question_label": q["label"],
            "question": q["prompt"],
            "response": r["text"],
            "tps": r["tps"],
            "words": len(r["text"].split()),
        })

    return {
        "kv": kv,
        "session": sid,
        "qa": qa_pairs,
        "avg_tps": sum(p["tps"] for p in qa_pairs) / max(len(qa_pairs), 1),
        "timestamp": datetime.datetime.now().isoformat(),
    }


# ── Loop principal ────────────────────────────────────────────────────────────
def run_all() -> dict:
    all_results = {}

    for kv in KV_TYPES:
        print(f"\n{'='*62}")
        print(f"  KV = {kv.upper()}")
        print(f"{'='*62}")

        ok   = start_server(kv)
        vram = get_vram_mib()
        print(f"  VRAM KV cache: {vram:.0f} MiB")

        sessions = []
        if ok:
            for s in range(SESSIONS):
                print(f"\n  -- Sessao {s+1}/{SESSIONS} " + "-"*40)
                sess = run_session(kv, s + 1)
                sessions.append(sess)
                if s < SESSIONS - 1:
                    print("\n  (pausa 5s entre sessoes...)")
                    time.sleep(5)

        stop_server()

        all_results[kv] = {
            "vram_mib":  vram,
            "sessions":  sessions,
            "avg_tps":   (sum(s["avg_tps"] for s in sessions) / max(len(sessions), 1))
                          if sessions else 0.0,
            "ok": ok,
        }

        # Grava JSON intermedio (recovery se crashar)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n  Guardado: {OUT}")

    return all_results


# ── Sumario final ─────────────────────────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "=" * 62)
    print("  SUMARIO")
    print(f"  {'KV':>6}  {'t/s medio':>10}  {'VRAM MiB':>10}  Status")
    print("  " + "-" * 48)
    for kv, r in results.items():
        status = "OK" if r["ok"] else "FALHOU"
        print(f"  {kv:>6}  {r['avg_tps']:>10.1f}  {r['vram_mib']:>10.0f}  {status}")
    print()
    print(f"  Respostas completas em: {OUT}")
    print("=" * 62)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 62)
    print("  BENCHMARK KV CACHE - RECOLHA DE RESPOSTAS")
    print("=" * 62)
    print(f"  Modelo   : {os.path.basename(MODEL)}")
    print(f"  KV tipos : {', '.join(KV_TYPES)}")
    print(f"  Sessoes  : {SESSIONS} x 5 perguntas encadeadas")
    print(f"  CTX={CTX}  NGL={NGL}  Temp={TEMP}  MaxTok={MAX_TOK}")
    print()

    for path, name in [(SERVER, "Servidor"), (MODEL, "Modelo")]:
        if not os.path.exists(path):
            print(f"ERRO: {name} nao encontrado:\n  {path}")
            sys.exit(1)

    try:
        results = run_all()
    except KeyboardInterrupt:
        print("\n  Interrompido")
        results = {}
    finally:
        stop_server()
        kill_all()

    if results:
        print_summary(results)
