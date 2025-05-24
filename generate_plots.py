#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_convergence_plots(results_dir):
    """Generuje wykresy zbieżności na podstawie danych z eksperymentu."""
    
    # Znajdź pliki z danymi zbieżności
    raw_data_dir = os.path.join(results_dir, 'raw_data')
    convergence_files = [f for f in os.listdir(raw_data_dir) if f.endswith('_convergence.csv')]
    
    if not convergence_files:
        print("Nie znaleziono plików z danymi zbieżności")
        return
    
    # Stwórz wykres
    plt.figure(figsize=(12, 8))
    
    # Dla każdego pliku z danymi zbieżności
    for file in convergence_files:
        file_path = os.path.join(raw_data_dir, file)
        
        # Parsuj nazwę pliku: rosenbrock_2D_standard_mt_seed1_convergence.csv
        parts = file.replace('_convergence.csv', '').split('_')
        function = parts[0]
        dimension = parts[1]
        algorithm = parts[2]
        generator = parts[3]
        seed = parts[4]  # seed1 lub seed2
        
        # Wczytaj dane
        df = pd.read_csv(file_path)
        
        # Utwórz etykietę
        label = f"{algorithm}_{seed}"
        color = 'blue' if algorithm == 'standard' else 'red'
        linestyle = '-' if seed == 'seed1' else '--'
        
        # Narysuj krzywą
        plt.plot(df['evaluations'], df['best_fitness'], 
                color=color, linestyle=linestyle, label=label, alpha=0.8, linewidth=2)
    
    # Konfiguracja wykresu
    plt.xlabel('Liczba ewaluacji funkcji')
    plt.ylabel('Najlepsza wartość funkcji')
    plt.title(f'Krzywa zbieżności - {function} {dimension}')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Zapisz wykres
    output_path = os.path.join(results_dir, 'convergence_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Wykres zapisano w: {output_path}")
    
    # Nie pokazuj wykresu w trybie headless
    # plt.show()

def main():
    # Znajdź najnowszy katalog z wynikami
    results_base = 'results'
    if not os.path.exists(results_base):
        print("Brak katalogu results")
        return
    
    # Znajdź najnowszy katalog
    subdirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
    if not subdirs:
        print("Brak katalogów z wynikami")
        return
    
    latest_dir = max(subdirs)
    results_dir = os.path.join(results_base, latest_dir)
    
    print(f"Generuję wykresy dla: {results_dir}")
    generate_convergence_plots(results_dir)

if __name__ == "__main__":
    main() 