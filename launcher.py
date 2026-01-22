#!/usr/bin/env python3
"""
CTGV System Launcher
Permite escolher entre interface gráfica e linha de comando
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='CTGV System Launcher')
    parser.add_argument('--gui', action='store_true',
                       help='Inicia interface gráfica')
    parser.add_argument('--example', action='store_true',
                       help='Executa exemplo em linha de comando')

    args = parser.parse_args()

    if args.gui:
        print("Iniciando interface gráfica...")
        try:
            from gui import main as gui_main
            gui_main()
        except ImportError as e:
            print(f"Erro ao importar GUI: {e}")
            print("Certifique-se de que tkinter está instalado")
            sys.exit(1)
    elif args.example:
        print("Executando exemplo...")
        from example import demonstrate_ctgv_system
        demonstrate_ctgv_system()
    else:
        print("CTGV System - Versatile Geometric Topology Computing")
        print("=" * 60)
        print("Uso:")
        print("  python launcher.py --gui        # Interface gráfica")
        print("  python launcher.py --example    # Exemplo em linha de comando")
        print("  python example.py               # Exemplo direto")
        print("  python gui.py                   # GUI direto")
        print("=" * 60)

if __name__ == "__main__":
    main()