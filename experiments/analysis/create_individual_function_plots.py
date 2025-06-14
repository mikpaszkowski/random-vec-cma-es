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

def create_individual_function_plots(dirname="final"):
    """Test nowej funkcjonalności osobnych wykresów dla każdej funkcji testowej"""
    
    print("=== Test osobnych wykresów zbieżności i sigma dla każdej funkcji ===")
    
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
    
    print("\n=== Tworzenie osobnych wykresów zbieżności dla każdej funkcji ===")
    
    # Test 1: Wykres zbieżności dla konkretnego wymiaru
    if available['dimensions']:
        test_dimension = available['dimensions'][0]  # Pierwszy dostępny wymiar
        print(f"\nTest 1: Wykresy zbieżności dla wymiaru {test_dimension}D")
        
        try:
            viz.create_individual_function_convergence_plots(
                dimension=test_dimension,
                save_plot=True
            )
            print(f"✓ Wykresy zbieżności dla {test_dimension}D utworzone pomyślnie!")
        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wykresów zbieżności dla {test_dimension}D: {e}")
    
    # Test 2: Wykresy zbieżności dla konkretnej funkcji
    if available['functions']:
        test_function = available['functions'][0]  # Pierwsza dostępna funkcja
        test_dimension = available['dimensions'][0] if available['dimensions'] else None
        
        if test_dimension:
            print(f"\nTest 2: Wykres zbieżności tylko dla funkcji {test_function}")
            
            try:
                viz.create_individual_function_convergence_plots(
                    dimension=test_dimension,
                    functions=[test_function],
                    save_plot=True
                )
                print(f"✓ Wykres zbieżności dla {test_function} utworzony pomyślnie!")
            except Exception as e:
                print(f"✗ Błąd podczas tworzenia wykresu zbieżności dla {test_function}: {e}")
    
    print("\n=== Tworzenie osobnych wykresów ewolucji sigma dla każdej funkcji ===")
    
    # Test 3: Wykres sigma dla konkretnego wymiaru
    if available['dimensions']:
        test_dimension = available['dimensions'][0]  # Pierwszy dostępny wymiar
        print(f"\nTest 3: Wykresy sigma dla wymiaru {test_dimension}D")
        
        try:
            viz.create_individual_function_sigma_plots(
                dimension=test_dimension,
                save_plot=True
            )
            print(f"✓ Wykresy sigma dla {test_dimension}D utworzone pomyślnie!")
        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wykresów sigma dla {test_dimension}D: {e}")
    
    # Test 4: Wykresy sigma dla konkretnej funkcji
    if available['functions']:
        test_function = available['functions'][0]  # Pierwsza dostępna funkcja
        test_dimension = available['dimensions'][0] if available['dimensions'] else None
        
        if test_dimension:
            print(f"\nTest 4: Wykres sigma tylko dla funkcji {test_function}")
            
            try:
                viz.create_individual_function_sigma_plots(
                    dimension=test_dimension,
                    functions=[test_function],
                    save_plot=True
                )
                print(f"✓ Wykres sigma dla {test_function} utworzony pomyślnie!")
            except Exception as e:
                print(f"✗ Błąd podczas tworzenia wykresu sigma dla {test_function}: {e}")
    
    # Test 5: Wszystkie wykresy zbieżności osobne
    print(f"\nTest 5: Wszystkie wykresy zbieżności osobne dla każdej funkcji")
    
    try:
        viz.create_all_individual_function_plots()
        print("✓ Wszystkie wykresy zbieżności osobne utworzone pomyślnie!")
    except Exception as e:
        print(f"✗ Błąd podczas tworzenia wszystkich wykresów zbieżności osobnych: {e}")
    
    # Test 6: Wszystkie wykresy sigma osobne
    print(f"\nTest 6: Wszystkie wykresy sigma osobne dla każdej funkcji")
    
    try:
        viz.create_all_individual_sigma_plots()
        print("✓ Wszystkie wykresy sigma osobne utworzone pomyślnie!")
    except Exception as e:
        print(f"✗ Błąd podczas tworzenia wszystkich wykresów sigma osobnych: {e}")
    
    print(f"\nWykresy zapisane w katalogach:")
    print(f"  Zbieżność: {viz.convergence_plots_dir}/individual_functions/")
    print(f"  Sigma: {viz.convergence_plots_dir}/individual_sigma/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test individual function convergence and sigma plots')
    parser.add_argument('--dirname', default='final', help='Results directory name')
    args = parser.parse_args()
    
    create_individual_function_plots(args.dirname) 