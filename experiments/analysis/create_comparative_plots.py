#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add parent directories to path when running directly
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

# Try to import from experiments package, fallback to direct import
try:
    from experiments.utils.visualization import ExperimentVisualizer
except ImportError:
    from utils.visualization import ExperimentVisualizer

def main(dirname="final"):
    """Test nowej funkcjonalności zbiorczych wykresów porównawczych"""
    
    print("=== Test zbiorczych wykresów porównawczych ===")
    
    # Utwórz wizualizator
    viz = ExperimentVisualizer(results_dir="results/" + dirname)
    
    # Sprawdź dostępne eksperymenty
    available = viz.get_available_experiments()
    print("Dostępne eksperymenty:")
    for key, values in available.items():
        print(f"  {key}: {values}")
    
    if not available['functions'] or not available['algorithms']:
        print("Błąd: Brak dostępnych danych!")
        return
    
    print("\n=== Tworzenie wykresów porównawczych ===")
    
    # Test 1: Wykres dla konkretnego wymiaru
    if available['dimensions']:
        test_dimension = available['dimensions'][0]  # Pierwszy dostępny wymiar
        print(f"\nTest 1: Wykres dla wymiaru {test_dimension}D")
        
        try:
            viz.create_comparative_convergence_plot(
                dimension=test_dimension,
                save_plot=True
            )
            print(f"✓ Wykres dla {test_dimension}D utworzony pomyślnie!")
        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wykresu dla {test_dimension}D: {e}")
    
    # Test 2: Wszystkie wykresy porównawcze
    print(f"\nTest 2: Wszystkie wykresy porównawcze")
    
    try:
        viz.create_all_comparative_plots()
        print("✓ Wszystkie wykresy porównawcze utworzone pomyślnie!")
    except Exception as e:
        print(f"✗ Błąd podczas tworzenia wszystkich wykresów: {e}")
    
    print(f"\nWykresy zapisane w katalogu: {viz.convergence_plots_dir}")

if __name__ == "__main__":
    main("final") 