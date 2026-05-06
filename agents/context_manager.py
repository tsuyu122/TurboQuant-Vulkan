#!/usr/bin/env python3
"""
📚 TurboQuant Context Manager

Gerencia contexto de até 512k tokens com:
- Salvamento persistente entre sessões
- Limpeza automática quando encher
- Compressão inteligente de histórico

Arquivos:
- context/session.json     → Estado da sessão atual
- context/history.jsonl    → Histórico de todas as sessões
- context/compressed.md    → Contexto comprimido para Gemma
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuração
BASE_DIR = Path(__file__).resolve().parent.parent
CONTEXT_DIR = BASE_DIR / "agents" / "context"
CONTEXT_DIR.mkdir(exist_ok=True)

SESSION_FILE = CONTEXT_DIR / "session.json"
HISTORY_FILE = CONTEXT_DIR / "history.jsonl"
COMPRESSED_FILE = CONTEXT_DIR / "compressed.md"

# Limites (aproximado: 1 token ≈ 4 chars)
MAX_TOKENS = 512_000
MAX_CHARS = MAX_TOKENS * 4  # ~2MB
COMPRESS_THRESHOLD = 0.8  # Comprime quando atingir 80%
KEEP_RECENT_ENTRIES = 50  # Mantém últimas N entradas sem comprimir


class ContextManager:
    """Gerenciador de contexto persistente."""
    
    def __init__(self):
        self.session = self._load_session()
    
    def _load_session(self) -> dict:
        """Carrega sessão do disco."""
        if SESSION_FILE.exists():
            try:
                return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            except:
                pass
        
        return {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started": datetime.now().isoformat(),
            "entries": [],
            "total_chars": 0,
            "compressed": False
        }
    
    def _save_session(self):
        """Salva sessão no disco."""
        SESSION_FILE.write_text(
            json.dumps(self.session, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    def add_entry(self, entry_type: str, content: str, metadata: dict = None):
        """
        Adiciona entrada ao contexto.
        
        Args:
            entry_type: "task", "plan", "result", "error", "note"
            content: Conteúdo da entrada
            metadata: Metadados opcionais
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,
            "content": content,
            "chars": len(content),
            "metadata": metadata or {}
        }
        
        self.session["entries"].append(entry)
        self.session["total_chars"] += len(content)
        
        # Verifica se precisa comprimir
        if self.session["total_chars"] > MAX_CHARS * COMPRESS_THRESHOLD:
            self._compress()
        
        self._save_session()
        self._append_history(entry)
    
    def _compress(self):
        """Comprime contexto antigo."""
        entries = self.session["entries"]
        
        if len(entries) <= KEEP_RECENT_ENTRIES:
            return
        
        # Separa antigas e recentes
        old_entries = entries[:-KEEP_RECENT_ENTRIES]
        recent_entries = entries[-KEEP_RECENT_ENTRIES:]
        
        # Cria resumo das antigas
        summary_parts = []
        for entry in old_entries:
            short = entry["content"][:200]
            summary_parts.append(f"[{entry['type']}] {short}...")
        
        compressed_summary = f"""# Contexto Comprimido

**Sessão**: {self.session['id']}
**Comprimido em**: {datetime.now().isoformat()}
**Entradas comprimidas**: {len(old_entries)}

## Resumo

{chr(10).join(summary_parts[:20])}

[...{len(summary_parts) - 20} entradas omitidas...]
"""
        
        # Salva comprimido
        COMPRESSED_FILE.write_text(compressed_summary, encoding="utf-8")
        
        # Atualiza sessão
        self.session["entries"] = recent_entries
        self.session["total_chars"] = sum(e["chars"] for e in recent_entries)
        self.session["compressed"] = True
        self.session["last_compression"] = datetime.now().isoformat()
        
        print(f"✅ Contexto comprimido: {len(old_entries)} entradas → resumo")
    
    def _append_history(self, entry: dict):
        """Adiciona ao histórico permanente."""
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def get_context_for_gemma(self, max_chars: int = 50000) -> str:
        """
        Retorna contexto formatado para Gemma.
        
        Args:
            max_chars: Máximo de caracteres a retornar
        """
        parts = []
        
        # Adiciona resumo comprimido se existir
        if COMPRESSED_FILE.exists():
            compressed = COMPRESSED_FILE.read_text(encoding="utf-8")
            parts.append(f"## Histórico Comprimido\n{compressed[:5000]}")
        
        # Adiciona entradas recentes
        parts.append("\n## Entradas Recentes\n")
        
        char_count = sum(len(p) for p in parts)
        
        for entry in reversed(self.session["entries"]):
            entry_text = f"\n### [{entry['type']}] {entry['timestamp'][:16]}\n{entry['content']}\n"
            
            if char_count + len(entry_text) > max_chars:
                break
            
            parts.append(entry_text)
            char_count += len(entry_text)
        
        return "\n".join(parts)
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do contexto."""
        return {
            "session_id": self.session["id"],
            "entries": len(self.session["entries"]),
            "total_chars": self.session["total_chars"],
            "total_tokens_approx": self.session["total_chars"] // 4,
            "usage_percent": (self.session["total_chars"] / MAX_CHARS) * 100,
            "compressed": self.session["compressed"]
        }
    
    def clear(self, archive: bool = True):
        """
        Limpa contexto atual.
        
        Args:
            archive: Se True, salva sessão atual no histórico antes de limpar
        """
        if archive and self.session["entries"]:
            # Salva resumo final no histórico
            final_summary = {
                "timestamp": datetime.now().isoformat(),
                "type": "session_end",
                "content": f"Sessão {self.session['id']} encerrada com {len(self.session['entries'])} entradas",
                "chars": 0,
                "metadata": {"session_id": self.session["id"]}
            }
            self._append_history(final_summary)
        
        # Reset sessão
        self.session = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started": datetime.now().isoformat(),
            "entries": [],
            "total_chars": 0,
            "compressed": False
        }
        self._save_session()
        
        print("✅ Contexto limpo")
    
    def new_session(self):
        """Inicia nova sessão (arquiva atual)."""
        self.clear(archive=True)


# Singleton global
_context_manager: Optional[ContextManager] = None

def get_context_manager() -> ContextManager:
    """Retorna instância global do gerenciador."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TurboQuant Context Manager")
    parser.add_argument("command", choices=["stats", "clear", "show", "add"], 
                       help="Comando a executar")
    parser.add_argument("--type", "-t", default="note", help="Tipo da entrada (para add)")
    parser.add_argument("--content", "-c", help="Conteúdo (para add)")
    
    args = parser.parse_args()
    
    cm = get_context_manager()
    
    if args.command == "stats":
        stats = cm.get_stats()
        print("\n📊 Estatísticas do Contexto")
        print("=" * 40)
        print(f"Sessão: {stats['session_id']}")
        print(f"Entradas: {stats['entries']}")
        print(f"Chars: {stats['total_chars']:,}")
        print(f"Tokens (aprox): {stats['total_tokens_approx']:,}")
        print(f"Uso: {stats['usage_percent']:.1f}%")
        print(f"Comprimido: {'Sim' if stats['compressed'] else 'Não'}")
        
    elif args.command == "clear":
        cm.clear()
        
    elif args.command == "show":
        context = cm.get_context_for_gemma(max_chars=5000)
        print(context)
        
    elif args.command == "add":
        if args.content:
            cm.add_entry(args.type, args.content)
            print(f"✅ Entrada adicionada: {args.type}")
        else:
            print("❌ Use --content para especificar conteúdo")


if __name__ == "__main__":
    main()
