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
    """Test nowej funkcjonalności osobnych wykresów ewolucji sigma dla każdej funkcji testowej"""
    
    print("=== Test osobnych wykresów ewolucji sigma dla każdej funkcji ===")
    
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
    
    print("\n=== Tworzenie osobnych wykresów ewolucji sigma ===")
    
    # Test 1: Wykres sigma dla konkretnego wymiaru
    if available['dimensions']:
        test_dimension = available['dimensions'][0]  # Pierwszy dostępny wymiar
        print(f"\nTest 1: Wykresy sigma dla wymiaru {test_dimension}D")
        
        try:
            viz.create_individual_function_sigma_plots(
                dimension=test_dimension,
                save_plot=True
            )
            print(f"✓ Wykresy sigma dla {test_dimension}D utworzone pomyślnie!")
        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wykresów sigma dla {test_dimension}D: {e}")
    
    # Test 2: Wykres sigma dla konkretnej funkcji
    if available['functions']:
        test_function = available['functions'][0]  # Pierwsza dostępna funkcja
        test_dimension = available['dimensions'][0] if available['dimensions'] else None
        
        if test_dimension:
            print(f"\nTest 2: Wykres sigma tylko dla funkcji {test_function}")
            
            try:
                viz.create_individual_function_sigma_plots(
                    dimension=test_dimension,
                    functions=[test_function],
                    save_plot=True
                )
                print(f"✓ Wykres sigma dla {test_function} utworzony pomyślnie!")
            except Exception as e:
                print(f"✗ Błąd podczas tworzenia wykresu sigma dla {test_function}: {e}")
    
    # Test 3: Wykresy sigma dla wybranych funkcji
    if len(available['functions']) >= 2:
        selected_functions = available['functions'][:2]  # Pierwsze dwie funkcje
        test_dimension = available['dimensions'][0] if available['dimensions'] else None
        
        if test_dimension:
            print(f"\nTest 3: Wykresy sigma dla wybranych funkcji: {selected_functions}")
            
            try:
                viz.create_individual_function_sigma_plots(
                    dimension=test_dimension,
                    functions=selected_functions,
                    save_plot=True
                )
                print(f"✓ Wykresy sigma dla wybranych funkcji utworzone pomyślnie!")
            except Exception as e:
                print(f"✗ Błąd podczas tworzenia wykresów sigma dla wybranych funkcji: {e}")
    
    # Test 4: Wszystkie wykresy sigma osobne
    print(f"\nTest 4: Wszystkie wykresy sigma osobne dla każdej funkcji")
    
    try:
        viz.create_all_individual_sigma_plots()
        print("✓ Wszystkie wykresy sigma osobne utworzone pomyślnie!")
    except Exception as e:
        print(f"✗ Błąd podczas tworzenia wszystkich wykresów sigma osobnych: {e}")
    
    print(f"\nWykresy sigma zapisane w katalogu: {viz.convergence_plots_dir}/individual_sigma/")
    
    # Pokaż strukturę katalogów
    sigma_dir = viz.convergence_plots_dir / "individual_sigma"
    if sigma_dir.exists():
        print(f"\nUtworzono następujące pliki:")
        for file in sorted(sigma_dir.glob("*.png")):
            print(f"  - {file.name}")

def demo_sigma_comparison():
    """Demonstracja porównania ewolucji sigma między algorytmami"""
    
    print("\n" + "="*60)
    print("DEMONSTRACJA: Porównanie ewolucji sigma między algorytmami")
    print("="*60)
    
    viz = ExperimentVisualizer(results_dir="results/final")
    available = viz.get_available_experiments()
    
    if not available['functions']:
        print("Brak dostępnych danych do demonstracji")
        return
    
    # Wybierz pierwszą dostępną funkcję i wymiar
    demo_function = available['functions'][0]
    demo_dimension = available['dimensions'][0] if available['dimensions'] else None
    
    if demo_dimension:
        print(f"Tworzenie wykresu porównawczego sigma dla: {demo_function} {demo_dimension}D")
        print("Ten wykres pokazuje:")
        print("  - Standard CMA-ES (niebieska linia ciągła)")
        print("  - Modified CMA-ES (czerwona linia przerywana)")
        print("  - Dane uśrednione z generatorów MT i PCG")
        print("  - Mediana z przedziałem kwartylowym")
        
        try:
            viz.create_individual_function_sigma_plots(
                dimension=demo_dimension,
                functions=[demo_function],
                save_plot=True,
                figsize=(14, 8)
            )
            print("✓ Wykres demonstracyjny utworzony pomyślnie!")
        except Exception as e:
            print(f"✗ Błąd podczas tworzenia wykresu demonstracyjnego: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test individual function sigma evolution plots')
    parser.add_argument('--dirname', default='final', help='Results directory name')
    parser.add_argument('--demo', action='store_true', help='Run demonstration of sigma comparison')
    args = parser.parse_args()
    
    main(args.dirname)
    
    if args.demo:
        demo_sigma_comparison() 