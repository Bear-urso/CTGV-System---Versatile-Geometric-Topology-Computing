#!/usr/bin/env python3
"""
CTGV System - Environment Check
Verifica se todas as dependências estão instaladas
"""
import sys
import importlib

def check_dependency(name, package_name=None):
    """Verifica se uma dependência está instalada"""
    if package_name is None:
        package_name = name

    try:
        importlib.import_module(package_name)
        print(f"✓ {name} - OK")
        return True
    except ImportError:
        print(f"✗ {name} - MISSING")
        return False

def main():
    print("CTGV System - Verificação de Dependências")
    print("=" * 50)

    # Dependências obrigatórias
    required = [
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
    ]

    # Dependências opcionais
    optional = [
        ("tkinter", "tkinter"),  # Para GUI
    ]

    print("Dependências Obrigatórias:")
    all_required = True
    for name, package in required:
        if not check_dependency(name, package):
            all_required = False

    print("\nDependências Opcionais:")
    for name, package in optional:
        check_dependency(name, package)

    print("\nVerificação dos Módulos CTGV:")
    ctgv_modules = [
        "ctgv.shapes",
        "ctgv.vector_field",
        "ctgv.gebit",
        "ctgv.engine",
        "ctgv.modeler",
        "ctgv.arbiter",
        "ctgv.clarification",
        "ctgv.utils"
    ]

    all_ctgv = True
    for module in ctgv_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module} - OK")
        except ImportError as e:
            print(f"✗ {module} - ERROR: {e}")
            all_ctgv = False

    print("\n" + "=" * 50)
    if all_required and all_ctgv:
        print("🎉 Ambiente configurado corretamente!")
        print("\nPara usar o sistema:")
        print("  python launcher.py --gui      # Interface gráfica")
        print("  python launcher.py --example  # Exemplo em linha de comando")
    else:
        print("❌ Problemas encontrados. Execute:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()