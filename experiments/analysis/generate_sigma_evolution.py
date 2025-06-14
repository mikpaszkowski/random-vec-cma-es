#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """
    Parsuje nazwę pliku aby wyciągnąć funkcję, wymiar i typ algorytmu.
    
    Przykład: schwefel_10D_modified_pcg_seed30_convergence.csv
    Returns: (function_name, dimension, algorithm_type)
    """
    parts = filename.replace('_convergence.csv', '').split('_')
    
    if len(parts) >= 3:
        function_name = parts[0]
        dimension = parts[1]  # np. "10D"
        algorithm_type = parts[2]  # "standard" lub "modified"
        return function_name, dimension, algorithm_type
    
    return None, None, None

def plot_sigma_evolution(function_name=None):
    """
    Rysuje wykres ewolucji odchylenia standardowego (sigma) z osobnymi seriami
    dla różnych wymiarów i typów algorytmów.
    
    Args:
        function_name (str): Nazwa funkcji testowej (np. 'rosenbrock', 'schwefel', 'rastrigin')
                           Jeśli None, wyświetla wszystkie funkcje
    """
    
    # Ścieżka do danych
    results_dir = Path("results/final/raw_data")
    
    if not results_dir.exists():
        print(f"Błąd: Katalog {results_dir} nie istnieje!")
        return
    
    # Znajdź wszystkie pliki convergence.csv
    if function_name:
        # Filtruj tylko dla wybranej funkcji
        pattern = f"{function_name}_*_convergence.csv"
        csv_files = list(results_dir.glob(pattern))
        
        if not csv_files:
            print(f"Błąd: Nie znaleziono plików dla funkcji '{function_name}'!")
            print("Dostępne funkcje:")
            all_files = list(results_dir.glob("*_convergence.csv"))
            functions = set()
            for f in all_files:
                func, _, _ = parse_filename(f.name)
                if func:
                    functions.add(func)
            for func in sorted(functions):
                print(f"  - {func}")
            return
    else:
        # Wszystkie pliki
        print("Wszystkie pliki")
        csv_files = list(results_dir.glob("*_convergence.csv"))
    
    if not csv_files:
        print("Błąd: Nie znaleziono plików _convergence.csv!")
        return
    
    print(f"Znaleziono {len(csv_files)} plików z danymi eksperymentów" + 
          (f" dla funkcji '{function_name}'" if function_name else ""))
    
    # Grupuj pliki według funkcji, wymiaru i typu algorytmu
    grouped_data = defaultdict(list)
    
    for csv_file in csv_files:
        func, dim, algo_type = parse_filename(csv_file.name)
        
        if func and dim and algo_type:
            # Klucz grupowania
            if function_name and func != function_name:
                continue
                
            key = (func, dim, algo_type)
            grouped_data[key].append(csv_file)
    
    if not grouped_data:
        print("Błąd: Nie znaleziono prawidłowych plików do analizy!")
        return
    
    # Przygotuj wykres
    plt.figure(figsize=(14, 10))
    
    # Kolory i style dla różnych kombinacji
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    line_styles = ['-', '--', '-.', ':']
    
    series_info = []
    color_idx = 0
    
    # Przetwórz każdą grupę
    for (func, dim, algo_type), file_list in sorted(grouped_data.items()):
        print(f"Przetwarzam: {func} {dim} {algo_type} ({len(file_list)} plików)")
        
        # Zbierz dane z wszystkich plików w grupie
        group_data = []
        
        for csv_file in file_list:
            try:
                df = pd.read_csv(csv_file)
                
                # Sprawdź czy mamy wymagane kolumny
                if 'evaluations' not in df.columns or 'sigma' not in df.columns:
                    continue
                
                # Filtruj nieprawidłowe wartości
                valid_mask = (np.isfinite(df['sigma']) & (df['sigma'] > 0) & 
                             np.isfinite(df['evaluations']))
                
                if valid_mask.sum() < 2:
                    continue
                
                # Dodaj dane
                valid_df = df.loc[valid_mask, ['evaluations', 'sigma']].copy()
                group_data.append(valid_df)
                
            except Exception as e:
                print(f"Błąd podczas przetwarzania {csv_file.name}: {e}")
                continue
        
        if not group_data:
            print(f"Brak prawidłowych danych dla {func} {dim} {algo_type}")
            continue
        
        # Oblicz średnią krzywą dla tej grupy
        # Znajdź wspólną siatkę punktów ewaluacji
        all_evaluations = []
        for df in group_data:
            all_evaluations.extend(df['evaluations'].tolist())
        
        if not all_evaluations:
            continue
            
        min_eval = min(all_evaluations)
        max_eval = max(all_evaluations)
        common_evals = np.linspace(min_eval, max_eval, 100)
        
        # Interpoluj dane dla każdego eksperymentu w grupie
        interpolated_sigmas = []
        for df in group_data:
            if len(df) >= 2:
                interpolated_sigma = np.interp(common_evals, df['evaluations'], df['sigma'])
                interpolated_sigmas.append(interpolated_sigma)
        
        if interpolated_sigmas:
            # Oblicz średnią
            interpolated_sigmas = np.array(interpolated_sigmas)
            mean_sigma = np.mean(interpolated_sigmas, axis=0)
            std_sigma = np.std(interpolated_sigmas, axis=0)
            
            # Wybierz kolor i styl
            color = colors[color_idx % len(colors)]
            style = line_styles[(color_idx // len(colors)) % len(line_styles)]
            
            # Utwórz etykietę
            label = f"{func} {dim} {algo_type} (n={len(interpolated_sigmas)})"
            
            # Narysuj średnią krzywą
            line = plt.semilogy(common_evals, mean_sigma, color=color, 
                               linestyle=style, linewidth=2.5, label=label)
            
            # Dodaj przedział ufności (opcjonalnie, tylko dla pierwszych kilku)
            if color_idx < 4:
                plt.fill_between(common_evals, 
                               np.maximum(mean_sigma - std_sigma, 1e-10),
                               mean_sigma + std_sigma,
                               color=color, alpha=0.15)
            
            # Zapisz informacje o serii
            series_info.append({
                'function': func,
                'dimension': dim,
                'algorithm': algo_type,
                'experiments': len(interpolated_sigmas),
                'initial_sigma': np.mean([data[0] for data in interpolated_sigmas]),
                'final_sigma': np.mean([data[-1] for data in interpolated_sigmas]),
                'color': color,
                'style': style
            })
            
            color_idx += 1
    
    # Formatowanie wykresu
    plt.xlabel('Liczba ewaluacji funkcji', fontsize=12)
    plt.ylabel('Odchylenie standardowe (sigma) - skala log', fontsize=12)
    
    # Tytuł zależny od funkcji
    if function_name:
        plt.title(f'Ewolucja odchylenia standardowego - funkcja {function_name}', fontsize=14)
        output_filename = f'sigma_evolution_{function_name}_comparison.png'
    else:
        plt.title('Ewolucja odchylenia standardowego - porównanie algorytmów', fontsize=14)
        output_filename = 'sigma_evolution_comparison.png'
    
    plt.grid(True, alpha=0.3)
    
    # Legenda (jeśli nie za dużo serii)
    if len(series_info) <= 12:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Ustaw limity osi
    plt.ylim(bottom=1e-6)
    
    # Zapisz wykres
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Wykres zapisano jako: {output_filename}")
    
    # Pokaż szczegółowe statystyki
    print(f"\n=== Szczegółowe statystyki ===")
    print(f"Funkcja: {function_name if function_name else 'wszystkie'}")
    
    for info in series_info:
        reduction = info['initial_sigma'] / info['final_sigma'] if info['final_sigma'] > 0 else float('inf')
        print(f"{info['function']} {info['dimension']} {info['algorithm']}: "
              f"{info['experiments']} exp., "
              f"σ: {info['initial_sigma']:.3f} → {info['final_sigma']:.6f} "
              f"(redukcja {reduction:.1f}x)")
    
    plt.show()

def list_available_functions():
    """Lista dostępnych funkcji testowych z detalami"""
    results_dir = Path("results/final/raw_data")
    
    if not results_dir.exists():
        print(f"Błąd: Katalog {results_dir} nie istnieje!")
        return
    
    all_files = list(results_dir.glob("*_convergence.csv"))
    
    # Grupuj według funkcji, wymiaru i algorytmu
    grouped = defaultdict(int)
    
    for f in all_files:
        func, dim, algo_type = parse_filename(f.name)
        if func and dim and algo_type:
            key = (func, dim, algo_type)
            grouped[key] += 1
    
    print("Dostępne konfiguracje eksperymentów:")
    
    current_func = None
    for (func, dim, algo_type), count in sorted(grouped.items()):
        if func != current_func:
            if current_func is not None:
                print()
            print(f"📊 {func}:")
            current_func = func
        print(f"   {dim} {algo_type}: {count} eksperymentów")

def main():
    parser = argparse.ArgumentParser(description='Wizualizacja ewolucji odchylenia standardowego CMA-ES z porównaniem algorytmów')
    parser.add_argument('function', nargs='?', default=None, 
                       help='Nazwa funkcji testowej (np. rosenbrock, schwefel, rastrigin)')
    parser.add_argument('--list', action='store_true', 
                       help='Wyświetl listę dostępnych funkcji i konfiguracji')
    
    args = parser.parse_args()
    
    # if args.list:
    #     list_available_functions()
    #     return
    
    plot_sigma_evolution(args.function)

if __name__ == "__main__":
    # Jeśli uruchomione bezpośrednio bez argumentów
    if len(sys.argv) == 1:
        print("Użycie:")
        print("  python plot_sigma_evolution.py [nazwa_funkcji]")
        print("  python plot_sigma_evolution.py --list")
        print("\nPrzykłady:")
        print("  python plot_sigma_evolution.py rosenbrock")
        print("  python plot_sigma_evolution.py schwefel")
        print("  python plot_sigma_evolution.py  # wszystkie funkcje")
        print()
        list_available_functions()
    else:
        main() 