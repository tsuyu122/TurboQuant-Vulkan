#!/usr/bin/env python3
"""
🤖 TurboQuant Auto-Typer v2

Automação robusta que funciona de QUALQUER janela:
1. Encontra VS Code
2. Foca na janela
3. Abre chat do Copilot
4. Cola mensagem
5. Envia

Funciona mesmo se VS Code estiver minimizado ou em segundo plano.

Uso:
    py agents/auto_typer_v2.py                      # Envia "executa o plano"
    py agents/auto_typer_v2.py -m "mensagem"        # Mensagem customizada
    py agents/auto_typer_v2.py --paste-plan         # Cola plan.md
    py agents/auto_typer_v2.py --delay 3            # Delay antes de começar
"""

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

try:
    import pyautogui
    import pyperclip
except ImportError:
    print("❌ Instale: py -m pip install pyautogui pyperclip")
    sys.exit(1)

# Windows API para manipulação de janelas
try:
    import win32gui
    import win32con
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("⚠️ win32gui não disponível, usando fallback")

# Configuração
BASE_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = BASE_DIR / "agents"
PLAN_FILE = AGENTS_DIR / "plan.md"

# pyautogui config
pyautogui.PAUSE = 0.15
pyautogui.FAILSAFE = True


def log(msg: str, level: str = "INFO"):
    symbols = {"OK": "✅", "WARN": "⚠️", "INFO": "ℹ️", "TYPE": "⌨️", "WAIT": "⏳", "FIND": "🔍"}
    print(f"[{symbols.get(level, '•')}] {msg}")


def countdown(seconds: int):
    """Contagem regressiva."""
    log(f"Iniciando em {seconds}s... (CANTO SUPERIOR ESQUERDO = CANCELAR)", "WAIT")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    print()


# ─── Funções Windows ──────────────────────────────────────────────────────────

def find_vscode_window():
    """Encontra janela do VS Code."""
    
    if not HAS_WIN32:
        return None
    
    vscode_hwnd = None
    
    def enum_callback(hwnd, results):
        nonlocal vscode_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Visual Studio Code" in title or "- Code" in title:
                vscode_hwnd = hwnd
                return False  # Para de enumerar
        return True
    
    try:
        win32gui.EnumWindows(enum_callback, None)
    except:
        pass
    
    return vscode_hwnd


def focus_window_win32(hwnd):
    """Foca janela usando Win32 API."""
    
    if not HAS_WIN32 or not hwnd:
        return False
    
    try:
        # Se minimizada, restaura
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        
        # Traz para frente
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.3)
        return True
    except Exception as e:
        log(f"Erro ao focar: {e}", "WARN")
        return False


def focus_vscode_powershell():
    """Foca VS Code usando PowerShell (fallback)."""
    
    ps_script = '''
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")]
    public static extern bool IsIconic(IntPtr hWnd);
}
"@

$vscode = Get-Process | Where-Object {$_.MainWindowTitle -like "*Visual Studio Code*" -or $_.MainWindowTitle -like "*- Code*"} | Select-Object -First 1

if ($vscode -and $vscode.MainWindowHandle -ne 0) {
    $hwnd = $vscode.MainWindowHandle
    
    # Restaura se minimizado
    if ([Win32]::IsIconic($hwnd)) {
        [Win32]::ShowWindow($hwnd, 9)  # SW_RESTORE
    }
    
    # Traz para frente
    [Win32]::SetForegroundWindow($hwnd)
    Write-Host "OK"
} else {
    Write-Host "NOT_FOUND"
}
'''
    
    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True, timeout=10
        )
        return "OK" in result.stdout
    except:
        return False


def focus_vscode():
    """Foca no VS Code usando método mais confiável."""
    
    log("Procurando VS Code...", "FIND")
    
    # Método 1: Win32 API direto
    if HAS_WIN32:
        hwnd = find_vscode_window()
        if hwnd:
            log(f"VS Code encontrado (hwnd={hwnd})", "OK")
            if focus_window_win32(hwnd):
                time.sleep(0.5)
                return True
    
    # Método 2: PowerShell
    log("Tentando via PowerShell...", "INFO")
    if focus_vscode_powershell():
        log("VS Code focado via PowerShell!", "OK")
        time.sleep(0.5)
        return True
    
    # Método 3: Alt+Tab (último recurso)
    log("Tentando Alt+Tab...", "WARN")
    pyautogui.hotkey('alt', 'tab')
    time.sleep(0.5)
    return True


def is_chat_open() -> bool:
    """Verifica se o chat do Copilot está aberto (heurística)."""
    # Não há forma confiável de detectar, assume que precisamos abrir
    return False


def open_copilot_chat():
    """Abre o chat do Copilot no VS Code."""
    
    log("Abrindo chat do Copilot...", "INFO")
    
    # Tenta múltiplos atalhos (diferentes configs)
    shortcuts = [
        ('ctrl', 'alt', 'i'),      # Padrão
        ('ctrl', 'shift', 'i'),    # Alternativo
    ]
    
    # Usa o primeiro
    pyautogui.hotkey(*shortcuts[0])
    time.sleep(1.0)
    
    log("Chat do Copilot aberto", "OK")


def clear_input():
    """Limpa campo de input."""
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.1)
    pyautogui.press('delete')
    time.sleep(0.1)


def paste_message(message: str):
    """Cola mensagem no campo."""
    
    log(f"Colando mensagem ({len(message)} chars)...", "TYPE")
    
    # Copia para clipboard
    pyperclip.copy(message)
    time.sleep(0.2)
    
    # Limpa qualquer texto existente
    clear_input()
    
    # Cola
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.3)


def send_message():
    """Envia a mensagem."""
    log("Enviando (Enter)...", "TYPE")
    pyautogui.press('enter')
    time.sleep(0.5)
    log("Mensagem enviada!", "OK")


def read_plan() -> str:
    """Lê conteúdo de plan.md."""
    if not PLAN_FILE.exists():
        return ""
    return PLAN_FILE.read_text(encoding="utf-8")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_automation(message: str = None, paste_plan: bool = False, delay: int = 3):
    """Executa automação completa."""
    
    print("\n" + "="*60)
    print("🤖 TURBOQUANT AUTO-TYPER v2")
    print("="*60)
    print()
    print("⚠️  ATENÇÃO:")
    print("   • Funciona de qualquer janela")
    print("   • Vai encontrar e focar VS Code automaticamente")
    print("   • CANTO SUPERIOR ESQUERDO = CANCELAR")
    print()
    
    # Prepara mensagem
    if paste_plan:
        plan_content = read_plan()
        if not plan_content:
            log("Plan.md vazio ou não encontrado!", "WARN")
            return False
        final_message = f"executa este plano:\n\n{plan_content}"
    elif message:
        final_message = message
    else:
        final_message = "executa o plano"
    
    log(f"Mensagem: {final_message[:60]}{'...' if len(final_message) > 60 else ''}", "INFO")
    
    # Countdown
    countdown(delay)
    
    try:
        # 1. Encontra e foca VS Code
        if not focus_vscode():
            log("Não foi possível focar VS Code!", "WARN")
            # Continua mesmo assim, pode estar já em foco
        
        # 2. Abre chat
        open_copilot_chat()
        
        # 3. Cola mensagem
        paste_message(final_message)
        
        # 4. Envia
        send_message()
        
        print()
        log("✨ Automação concluída!", "OK")
        return True
        
    except pyautogui.FailSafeException:
        print()
        log("🛑 CANCELADO (failsafe)", "WARN")
        return False
    except Exception as e:
        print()
        log(f"Erro: {e}", "WARN")
        return False


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Auto-Typer v2")
    parser.add_argument("--message", "-m", type=str, help="Mensagem a enviar")
    parser.add_argument("--paste-plan", "-p", action="store_true", help="Cola plan.md")
    parser.add_argument("--delay", "-d", type=int, default=3, help="Delay inicial (s)")
    parser.add_argument("--test", "-t", action="store_true", help="Modo teste")
    
    args = parser.parse_args()
    
    if args.test:
        log("Modo teste", "INFO")
        log(f"pyautogui: OK", "OK")
        log(f"pyperclip: OK", "OK")
        log(f"win32gui: {'OK' if HAS_WIN32 else 'Não disponível (usando fallback)'}", "OK" if HAS_WIN32 else "WARN")
        log(f"Plan existe: {PLAN_FILE.exists()}", "OK" if PLAN_FILE.exists() else "WARN")
        
        hwnd = find_vscode_window() if HAS_WIN32 else None
        log(f"VS Code encontrado: {hwnd is not None}", "OK" if hwnd else "WARN")
        return
    
    success = run_automation(
        message=args.message,
        paste_plan=args.paste_plan,
        delay=args.delay
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
