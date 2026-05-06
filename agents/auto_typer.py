#!/usr/bin/env python3
"""
🤖 TurboQuant Auto-Typer

Automação física que simula digitação no VS Code.
Usa pyautogui para controlar teclado/mouse quando CLI não está disponível.

IMPORTANTE: Não mexa no mouse/teclado enquanto roda!

Uso:
    py agents/auto_typer.py                    # Envia "executa o plano"
    py agents/auto_typer.py --paste-plan       # Cola o plan.md inteiro
    py agents/auto_typer.py --message "texto"  # Envia mensagem customizada
    py agents/auto_typer.py --delay 5          # Delay antes de começar (segundos)
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import pyautogui
    import pyperclip
except ImportError:
    print("❌ Instale: py -m pip install pyautogui pyperclip")
    sys.exit(1)

# Configuração
BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"
PLAN_FILE = AGENTS_DIR / "plan.md"

# Configurações pyautogui
pyautogui.PAUSE = 0.1  # Pausa entre ações
pyautogui.FAILSAFE = True  # Move mouse pro canto = para tudo


def log(msg: str, level: str = "INFO"):
    symbols = {"OK": "✅", "WARN": "⚠️", "INFO": "ℹ️", "TYPE": "⌨️", "WAIT": "⏳"}
    print(f"[{symbols.get(level, '•')}] {msg}")


def countdown(seconds: int):
    """Contagem regressiva visual."""
    log(f"Começando em {seconds} segundos... (mova mouse pro canto para CANCELAR)", "WAIT")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    print()


def focus_vscode():
    """Tenta focar no VS Code."""
    log("Focando no VS Code...", "INFO")
    
    # Tenta encontrar janela do VS Code
    try:
        import subprocess
        # Windows: usa PowerShell para trazer VS Code para frente
        ps_script = '''
        $vscode = Get-Process | Where-Object {$_.MainWindowTitle -like "*Visual Studio Code*"} | Select-Object -First 1
        if ($vscode) {
            [void][System.Reflection.Assembly]::LoadWithPartialName("Microsoft.VisualBasic")
            [Microsoft.VisualBasic.Interaction]::AppActivate($vscode.Id)
            Write-Host "OK"
        } else {
            Write-Host "NOT_FOUND"
        }
        '''
        result = subprocess.run(["powershell", "-Command", ps_script], 
                              capture_output=True, text=True, timeout=5)
        if "OK" in result.stdout:
            log("VS Code focado!", "OK")
            time.sleep(0.5)
            return True
        else:
            log("VS Code não encontrado, tentando Alt+Tab...", "WARN")
    except Exception as e:
        log(f"Erro ao focar: {e}", "WARN")
    
    # Fallback: Alt+Tab
    pyautogui.hotkey('alt', 'tab')
    time.sleep(0.5)
    return True


def open_copilot_chat():
    """Abre o chat do Copilot no VS Code."""
    log("Abrindo chat do Copilot (Ctrl+Alt+I)...", "INFO")
    
    # Atalho padrão do Copilot Chat
    pyautogui.hotkey('ctrl', 'alt', 'i')
    time.sleep(1.5)  # Espera abrir
    
    log("Chat deveria estar aberto", "OK")


def type_message(message: str, use_clipboard: bool = True):
    """Digita ou cola uma mensagem."""
    
    if use_clipboard:
        log(f"Colando mensagem ({len(message)} chars)...", "TYPE")
        # Copia para clipboard e cola
        pyperclip.copy(message)
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'v')
    else:
        log(f"Digitando mensagem ({len(message)} chars)...", "TYPE")
        # Digita caractere por caractere (mais lento mas mais confiável)
        pyautogui.typewrite(message, interval=0.02)
    
    time.sleep(0.3)


def send_message():
    """Envia a mensagem (Enter)."""
    log("Enviando (Enter)...", "TYPE")
    pyautogui.press('enter')
    time.sleep(0.5)
    log("Mensagem enviada!", "OK")


def read_plan() -> str:
    """Lê o conteúdo de plan.md."""
    if not PLAN_FILE.exists():
        log(f"Arquivo não encontrado: {PLAN_FILE}", "WARN")
        return ""
    
    content = PLAN_FILE.read_text(encoding="utf-8")
    log(f"Plan.md lido ({len(content)} chars)", "OK")
    return content


def run_automation(message: str = None, paste_plan: bool = False, delay: int = 3):
    """Executa a automação completa."""
    
    print("\n" + "="*60)
    print("🤖 TURBOQUANT AUTO-TYPER")
    print("="*60)
    print()
    print("⚠️  ATENÇÃO:")
    print("   • NÃO mexa no mouse/teclado durante a execução")
    print("   • Mova o mouse para o CANTO SUPERIOR ESQUERDO para CANCELAR")
    print()
    
    # Determina mensagem
    if paste_plan:
        plan_content = read_plan()
        if not plan_content:
            log("Plano vazio ou não encontrado!", "WARN")
            return False
        # Encapsula em comando
        final_message = f"executa este plano:\n\n{plan_content}"
    elif message:
        final_message = message
    else:
        final_message = "executa o plano"
    
    log(f"Mensagem preparada: {final_message[:50]}...", "INFO")
    
    # Countdown
    countdown(delay)
    
    # Execução
    try:
        # 1. Foca no VS Code
        focus_vscode()
        
        # 2. Abre chat do Copilot
        open_copilot_chat()
        
        # 3. Digita/cola mensagem
        type_message(final_message, use_clipboard=True)
        
        # 4. Envia
        send_message()
        
        print()
        log("✨ Automação concluída com sucesso!", "OK")
        return True
        
    except pyautogui.FailSafeException:
        print()
        log("🛑 CANCELADO pelo usuário (failsafe ativado)", "WARN")
        return False
    except Exception as e:
        print()
        log(f"Erro na automação: {e}", "WARN")
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto-typer para VS Code Copilot")
    parser.add_argument("--message", "-m", type=str, help="Mensagem customizada para enviar")
    parser.add_argument("--paste-plan", "-p", action="store_true", help="Cola o conteúdo de plan.md")
    parser.add_argument("--delay", "-d", type=int, default=3, help="Delay antes de começar (segundos)")
    parser.add_argument("--test", "-t", action="store_true", help="Modo teste (não envia)")
    
    args = parser.parse_args()
    
    if args.test:
        log("Modo teste - apenas verificando...", "INFO")
        log(f"pyautogui: OK", "OK")
        log(f"pyperclip: OK", "OK")
        log(f"Plan existe: {PLAN_FILE.exists()}", "OK" if PLAN_FILE.exists() else "WARN")
        return
    
    success = run_automation(
        message=args.message,
        paste_plan=args.paste_plan,
        delay=args.delay
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
