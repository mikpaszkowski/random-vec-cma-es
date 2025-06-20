#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test skryptu dla nowej funkcjonalności wykresów pudełkowych w klasie ExperimentVisualizer.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path when running directly
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    experiments_dir = current_dir.parent
    root_dir = experiments_dir.parent
    
    for path in [str(root_dir), str(experiments_dir)]:
        if path not in sys.path:
            sys.path.insert(0, path)

# Try to import from experiments package, fallback to direct import
try:
    from experiments.utils.visualization import ExperimentVisualizer
except ImportError:
    try:
        from utils.visualization import ExperimentVisualizer
    except ImportError:
        # Last resort - add utils directory to path
        utils_dir = Path(__file__).parent.parent / "utils"
        if utils_dir.exists():
            sys.path.insert(0, str(utils_dir))
            from visualization import ExperimentVisualizer
        else:
            raise ImportError("Cannot find ExperimentVisualizer module")

def main():
    """
    Testuje funkcjonalność wykresów pudełkowych w klasie ExperimentVisualizer.
    """
    print("=== Test wykresów pudełkowych - ExperimentVisualizer ===")
    
    # Utwórz instancję wizualizatora
    visualizer = ExperimentVisualizer("results/final")
    
    # Sprawdź dostępne eksperymenty
    print("\n1. Sprawdzanie dostępnych eksperymentów...")
    available = visualizer.get_available_experiments()
    print(f"Dostępne funkcje: {available['functions']}")
    print(f"Dostępne wymiary: {available['dimensions']}")
    print(f"Dostępne algorytmy: {available['algorithms']}")
    print(f"Dostępne generatory: {available['generators']}")
    
    if not available['functions']:
        print("Brak dostępnych danych eksperymentów!")
        return
    
    # Utwórz przykładowy wykres pudełkowy
    print("\n2. Tworzenie przykładowego wykresu pudełkowego...")
    
    # Wybierz pierwszą dostępną funkcję i wymiar
    function = available['functions'][0]
    dimension = available['dimensions'][0]
    
    print(f"Tworzenie wykresów pudełkowych dla: {function} {dimension}D")
    
    try:
        visualizer.create_boxplots(
            function=function,
            dimension=dimension,
            algorithms=available['algorithms'],
            generators=available['generators'],
            save_plot=False,
            figsize=(16, 12)
        )
        print("✓ Wykresy pudełkowe utworzone pomyślnie!")
        
    except Exception as e:
        print(f"✗ Błąd podczas tworzenia wykresów pudełkowych: {e}")
        import traceback
        traceback.print_exc()
    
    # Test dla konkretnej funkcji jeśli dostępna
    if 'rosenbrock' in available['functions'] and 2 in available['dimensions']:
        print("\n3. Test dla funkcji Ackley 2D...")
        try:
            visualizer.create_boxplots(
                function='rosenbrock',
                dimension=2,
                save_plot=True
            )
            print("✓ Wykres Ackley 2D utworzony pomyślnie!")
        except Exception as e:
            print(f"✗ Błąd dla Ackley 2D: {e}")
    
    # Opcjonalnie: utwórz wszystkie wykresy pudełkowe
    print("\n4. Czy chcesz utworzyć wszystkie wykresy pudełkowe? (może to potrwać kilka minut)")
    response = input("Wpisz 'tak' aby kontynuować: ").lower().strip()
    
    if response in ['tak', 'yes', 'y', 't']:
        print("Tworzenie wszystkich wykresów pudełkowych...")
        try:
            visualizer.create_all_boxplots()
            print("✓ Wszystkie wykresy pudełkowe utworzone pomyślnie!")
        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wszystkich wykresów: {e}")
    else:
        print("Pomijanie tworzenia wszystkich wykresów pudełkowych.")
    
    print("\n=== Test zakończony ===")

if __name__ == "__main__":
    main() 