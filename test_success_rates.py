#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test skryptu dla nowej funkcjonalności wykresów wskaźników sukcesu w klasie ExperimentVisualizer.
"""

from experiments.utils.visualization import ExperimentVisualizer

def main():
    """
    Testuje funkcjonalność wykresów wskaźników sukcesu w klasie ExperimentVisualizer.
    """
    print("=== Test wykresów wskaźników sukcesu - ExperimentVisualizer ===")
    
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
    
    # Utwórz wykres wskaźników sukcesu
    print("\n2. Tworzenie wykresów wskaźników sukcesu...")
    print("Analiza obejmuje heatmapy i wykresy słupkowe porównawcze")
    
    try:
        visualizer.create_success_rate_plots(
            algorithms=available['algorithms'],
            generators=available['generators'],
            save_plot=True,
            figsize=(18, 14)
        )
        print("✓ Wykresy wskaźników sukcesu utworzone pomyślnie!")
        
    except Exception as e:
        print(f"✗ Błąd podczas tworzenia wykresów wskaźników sukcesu: {e}")
        import traceback
        traceback.print_exc()
    
    # Test z ograniczonymi parametrami
    print("\n3. Test dla wybranych algorytmów...")
    try:
        visualizer.create_success_rate_plots(
            algorithms=['modified'],  # Tylko algorytm zmodyfikowany
            generators=['mt', 'pcg'],
            save_plot=False,  # Nie zapisuj, tylko wyświetl
            figsize=(16, 12)
        )
        print("✓ Wykres dla algoritmu Modified utworzony pomyślnie!")
    except Exception as e:
        print(f"✗ Błąd dla wybranych algorytmów: {e}")
    
    print("\n=== Test zakończony ===")

if __name__ == "__main__":
    main() 