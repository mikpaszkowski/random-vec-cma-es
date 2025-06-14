#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt do generowania ogólnej tabeli porównawczej Standard vs Modified CMA-ES.

Ten skrypt agreguje wszystkie wyniki eksperymentów (niezależnie od funkcji, wymiarów, 
generatorów) i tworzy prostą tabelę porównawczą pokazującą ogólne różnice między algorytmami.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from tabulate import tabulate


def load_experiment_data(results_dir):
    """
    Wczytuje dane eksperymentów z katalogu wyników.
    
    Args:
        results_dir: Ścieżka do katalogu z wynikami
        
    Returns:
        Lista słowników z danymi eksperymentów
    """
    raw_data_dir = os.path.join(results_dir, 'raw_data')
    
    if not os.path.exists(raw_data_dir):
        raise ValueError(f"Katalog z danymi nie istnieje: {raw_data_dir}")
    
    experiments = []
    
    # Wczytaj wszystkie pliki JSON z raw_data
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(raw_data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    experiments.append(data)
            except Exception as e:
                print(f"Błąd przy wczytywaniu {filename}: {e}")
                continue
    
    print(f"Wczytano {len(experiments)} eksperymentów z {raw_data_dir}")
    return experiments


def aggregate_by_algorithm(experiments):
    """
    Agreguje wszystkie wyniki według typu algorytmu (Standard vs Modified).
    
    Args:
        experiments: Lista danych eksperymentów
        
    Returns:
        Słownik z zagregowanymi danymi dla każdego algorytmu
    """
    algorithm_data = {'standard': [], 'modified': []}
    
    for exp in experiments:
        algorithm = exp['algorithm'].lower()
        if algorithm in algorithm_data:
            algorithm_data[algorithm].append(exp)
    
    print(f"Standard: {len(algorithm_data['standard'])} eksperymentów")
    print(f"Modified: {len(algorithm_data['modified'])} eksperymentów")
    
    return algorithm_data


def calculate_metrics(experiments):
    """
    Oblicza metryki dla listy eksperymentów.
    
    Args:
        experiments: Lista eksperymentów
        
    Returns:
        Słownik z obliczonymi metrykami
    """
    if not experiments:
        return None
    
    # Wyciągnij dane
    fitness_values = [exp['result']['fun'] for exp in experiments]
    evaluations = [exp['result']['nfev'] for exp in experiments]
    success_flags = [exp.get('success', False) for exp in experiments]
    
    # Oblicz metryki
    fitness_array = np.array(fitness_values)
    mean_fitness = np.mean(fitness_array)
    std_fitness = np.std(fitness_array)
    
    # Coefficient of Variation (CV) = std/mean (jeśli mean != 0)
    cv = std_fitness / mean_fitness if mean_fitness != 0 else float('inf')
    
    # Success rate
    success_rate = np.mean(success_flags) * 100  # w procentach
    
    # Średnie ewaluacje
    mean_evaluations = np.mean(evaluations)
    
    # Mediana fitness (dodatkowa metryka)
    median_fitness = np.median(fitness_array)
    
    return {
        'mean_fitness': mean_fitness,
        'median_fitness': median_fitness,
        'std_fitness': std_fitness,
        'cv': cv,
        'success_rate': success_rate,
        'mean_evaluations': mean_evaluations,
        'total_experiments': len(experiments)
    }


def format_number(value, precision=2, use_scientific=True):
    """
    Formatuje liczbę w odpowiednim formacie.
    
    Args:
        value: Wartość do sformatowania
        precision: Liczba miejsc dziesiętnych
        use_scientific: Czy używać notacji naukowej dla małych/dużych liczb
        
    Returns:
        Sformatowana wartość jako string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    if use_scientific and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def create_comparison_table(standard_metrics, modified_metrics):
    """
    Tworzy tabelę porównawczą dla algorytmów Standard vs Modified.
    
    Args:
        standard_metrics: Metryki dla algorytmu Standard
        modified_metrics: Metryki dla algorytmu Modified
        
    Returns:
        DataFrame z tabelą porównawczą
    """
    # Przygotowanie danych do tabeli
    comparison_data = []
    
    # Średnia jakość (fitness)
    standard_fitness = standard_metrics['mean_fitness']
    modified_fitness = modified_metrics['mean_fitness']
    fitness_winner = "Modified" if modified_fitness < standard_fitness else "Standard"
    if abs(modified_fitness - standard_fitness) / max(abs(standard_fitness), abs(modified_fitness)) < 0.01:
        fitness_winner = "Tie"
    
    comparison_data.append({
        'Metryka': 'Średnia jakość',
        'Standard': format_number(standard_fitness),
        'Modified': format_number(modified_fitness),
        'Zwycięzca': fitness_winner
    })
    
    # Success rate
    standard_success = standard_metrics['success_rate']
    modified_success = modified_metrics['success_rate']
    success_winner = "Modified" if modified_success > standard_success else "Standard"
    if abs(modified_success - standard_success) < 1.0:  # różnica mniejsza niż 1%
        success_winner = "Tie"
    
    comparison_data.append({
        'Metryka': 'Success rate',
        'Standard': f"{standard_success:.1f}%",
        'Modified': f"{modified_success:.1f}%",
        'Zwycięzca': success_winner
    })
    
    # Średnie ewaluacje (mniej = lepiej)
    standard_evals = standard_metrics['mean_evaluations']
    modified_evals = modified_metrics['mean_evaluations']
    evals_winner = "Modified" if modified_evals < standard_evals else "Standard"
    if abs(modified_evals - standard_evals) / max(standard_evals, modified_evals) < 0.05:
        evals_winner = "Tie"
    
    comparison_data.append({
        'Metryka': 'Średnie ewaluacje',
        'Standard': f"{standard_evals:.0f}",
        'Modified': f"{modified_evals:.0f}",
        'Zwycięzca': evals_winner
    })
    
    # Stabilność (CV - mniej = lepiej)
    standard_cv = standard_metrics['cv']
    modified_cv = modified_metrics['cv']
    cv_winner = "Modified" if modified_cv < standard_cv else "Standard"
    if abs(modified_cv - standard_cv) / max(standard_cv, modified_cv) < 0.05:
        cv_winner = "Tie"
    
    comparison_data.append({
        'Metryka': 'Stabilność (CV)',
        'Standard': format_number(standard_cv, precision=3, use_scientific=False),
        'Modified': format_number(modified_cv, precision=3, use_scientific=False),
        'Zwycięzca': cv_winner
    })
    
    # Mediana jakości (dodatkowa metryka)
    standard_median = standard_metrics['median_fitness']
    modified_median = modified_metrics['median_fitness']
    median_winner = "Modified" if modified_median < standard_median else "Standard"
    if abs(modified_median - standard_median) / max(abs(standard_median), abs(modified_median)) < 0.01:
        median_winner = "Tie"
    
    comparison_data.append({
        'Metryka': 'Mediana jakości',
        'Standard': format_number(standard_median),
        'Modified': format_number(modified_median),
        'Zwycięzca': median_winner
    })
    
    return pd.DataFrame(comparison_data)


def create_detailed_stats(standard_metrics, modified_metrics):
    """
    Tworzy tabelę z szczegółowymi statystykami.
    
    Args:
        standard_metrics: Metryki dla algorytmu Standard
        modified_metrics: Metryki dla algorytmu Modified
        
    Returns:
        DataFrame ze szczegółowymi statystykami
    """
    stats_data = [
        {
            'Statystyka': 'Liczba eksperymentów',
            'Standard': standard_metrics['total_experiments'],
            'Modified': modified_metrics['total_experiments']
        },
        {
            'Statystyka': 'Średnia fitness',
            'Standard': format_number(standard_metrics['mean_fitness']),
            'Modified': format_number(modified_metrics['mean_fitness'])
        },
        {
            'Statystyka': 'Mediana fitness',
            'Standard': format_number(standard_metrics['median_fitness']),
            'Modified': format_number(modified_metrics['median_fitness'])
        },
        {
            'Statystyka': 'Odchylenie std fitness',
            'Standard': format_number(standard_metrics['std_fitness']),
            'Modified': format_number(modified_metrics['std_fitness'])
        },
        {
            'Statystyka': 'Success rate (%)',
            'Standard': f"{standard_metrics['success_rate']:.2f}",
            'Modified': f"{modified_metrics['success_rate']:.2f}"
        },
        {
            'Statystyka': 'Średnie ewaluacje',
            'Standard': f"{standard_metrics['mean_evaluations']:.0f}",
            'Modified': f"{modified_metrics['mean_evaluations']:.0f}"
        },
        {
            'Statystyka': 'Coefficient of Variation',
            'Standard': format_number(standard_metrics['cv'], precision=4, use_scientific=False),
            'Modified': format_number(modified_metrics['cv'], precision=4, use_scientific=False)
        }
    ]
    
    return pd.DataFrame(stats_data)


def save_results(comparison_df, stats_df, output_dir, base_filename="algorithm_comparison"):
    """
    Zapisuje wyniki w różnych formatach.
    
    Args:
        comparison_df: DataFrame z porównaniem
        stats_df: DataFrame ze szczegółowymi statystykami
        output_dir: Katalog docelowy
        base_filename: Podstawowa nazwa pliku
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV
    comparison_df.to_csv(os.path.join(output_dir, f"{base_filename}.csv"), index=False)
    stats_df.to_csv(os.path.join(output_dir, f"{base_filename}_detailed.csv"), index=False)
    
    # Markdown
    md_path = os.path.join(output_dir, f"{base_filename}.md")
    with open(md_path, 'w') as f:
        f.write("# Porównanie algorytmów Standard vs Modified CMA-ES\n\n")
        f.write("## Tabela porównawcza\n\n")
        f.write(tabulate(comparison_df, headers='keys', tablefmt='github', showindex=False))
        f.write("\n\n## Szczegółowe statystyki\n\n")
        f.write(tabulate(stats_df, headers='keys', tablefmt='github', showindex=False))
    
    print(f"Zapisano wyniki do: {output_dir}")


def generate_comparison_table(results_dir='results/final', output_dir=None, save=False):
    """
    Generuje ogólną tabelę porównawczą Standard vs Modified CMA-ES.
    
    Args:
        results_dir: Ścieżka do katalogu z wynikami eksperymentów
        output_dir: Katalog docelowy dla tabel (domyślnie: results_dir)
        save: Czy zapisać wyniki do plików
    """
    if output_dir is None:
        output_dir = results_dir
        
    try:
        # Wczytanie danych
        print(f"Wczytywanie danych z: {results_dir}")
        experiments = load_experiment_data(results_dir)
        
        if not experiments:
            print("Nie znaleziono danych eksperymentów!")
            return 1
        
        # Agregacja według algorytmu
        print("Agregacja wyników według algorytmu...")
        algorithm_data = aggregate_by_algorithm(experiments)
        
        # Obliczenie metryk
        print("Obliczanie metryk...")
        standard_metrics = calculate_metrics(algorithm_data['standard'])
        modified_metrics = calculate_metrics(algorithm_data['modified'])
        
        if not standard_metrics or not modified_metrics:
            print("Brak wystarczających danych dla porównania!")
            return 1
        
        # Tworzenie tabel
        comparison_df = create_comparison_table(standard_metrics, modified_metrics)
        stats_df = create_detailed_stats(standard_metrics, modified_metrics)
        
        # Wyświetlenie wyników
        print("\n" + "="*80)
        print("TABELA PORÓWNAWCZA: Standard vs Modified CMA-ES")
        print("="*80)
        print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
        
        print("\n" + "="*80)
        print("SZCZEGÓŁOWE STATYSTYKI")
        print("="*80)
        print(tabulate(stats_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Zapisanie wyników jeśli wymagane
        if save:
            save_results(comparison_df, stats_df, output_dir)
        
        # Podsumowanie zwycięstw
        winners = comparison_df['Zwycięzca'].value_counts()
        print(f"\n" + "="*80)
        print("PODSUMOWANIE ZWYCIĘSTW:")
        print("="*80)
        for algorithm, count in winners.items():
            print(f"{algorithm}: {count} metryki")
        
        return 0
        
    except Exception as e:
        print(f"Błąd: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Główna funkcja dla uruchomienia z linii poleceń."""
    parser = argparse.ArgumentParser(
        description="Generuje ogólną tabelę porównawczą Standard vs Modified CMA-ES"
    )
    parser.add_argument(
        'results_dir',
        help='Ścieżka do katalogu z wynikami eksperymentów'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Katalog docelowy dla tabel (domyślnie: results_dir)'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Zapisz wyniki do plików'
    )
    
    args = parser.parse_args()
    
    return generate_comparison_table(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        save=args.save
    )


if __name__ == "__main__":
    sys.exit(main()) 