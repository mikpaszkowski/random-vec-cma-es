#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt do generowania tabeli podsumowującej jakość rozwiązań dla algorytmów CMA-ES.

Ten skrypt odczytuje dane z eksperymentów i tworzy tabelę z miarami statystycznymi
(średnia, mediana, odchylenie standardowe, minimum, maksimum) dla każdej konfiguracji.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
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


def aggregate_results(experiments):
    """
    Agreguje wyniki eksperymentów według funkcji, wymiaru, algorytmu i generatora.
    
    Args:
        experiments: Lista danych eksperymentów
        
    Returns:
        DataFrame z zagregowanymi statystykami
    """
    # Grupowanie wyników
    groups = {}
    
    for exp in experiments:
        # Grupuj również według generatora
        key = (exp['function'], exp['dimension'], exp['algorithm'], exp.get('generator', 'unknown'))
        
        if key not in groups:
            groups[key] = []
        
        # Dodaj wartość funkcji celu (best fitness)
        groups[key].append(exp['result']['fun'])
    
    # Obliczanie statystyk dla każdej grupy
    summary_data = []
    
    for (function, dimension, algorithm, generator), fitness_values in groups.items():
        if len(fitness_values) == 0:
            continue
            
        fitness_array = np.array(fitness_values)
        
        summary_row = {
            'Funkcja': function.capitalize(),
            'Wymiar': f"{dimension}D",
            'Algorithm': algorithm.capitalize(),
            'Generator': generator.upper() if generator != 'unknown' else 'Unknown',
            'Średnia': np.mean(fitness_array),
            'Mediana': np.median(fitness_array),
            'Std Dev': np.std(fitness_array),
            'Min': np.min(fitness_array),
            'Max': np.max(fitness_array),
            'N': len(fitness_values)  # Skrócona nazwa dla liczby powtórzeń
        }
        
        summary_data.append(summary_row)
    
    # Konwersja do DataFrame i sortowanie
    df = pd.DataFrame(summary_data)
    
    if not df.empty:
        # Sortowanie: funkcja, wymiar, algorytm, generator
        df = df.sort_values(['Funkcja', 'Wymiar', 'Algorithm', 'Generator'])
        df = df.reset_index(drop=True)
    
    return df


def format_scientific_notation(value, precision=2):
    """
    Formatuje liczbę w notacji naukowej z zadaną precyzją.
    
    Args:
        value: Wartość do sformatowania
        precision: Liczba miejsc dziesiętnych
        
    Returns:
        Sformatowana wartość jako string
    """
    if value == 0:
        return "0.0"
    elif abs(value) >= 1e-3 and abs(value) < 1e3:
        return f"{value:.{precision}f}"
    else:
        return f"{value:.{precision}e}"


def create_formatted_table(df, format_type='grid'):
    """
    Tworzy sformatowaną tabelę z wynikami.
    
    Args:
        df: DataFrame z wynikami
        format_type: Typ formatowania tabeli ('grid', 'latex', 'markdown', etc.)
        
    Returns:
        Sformatowana tabela jako string
    """
    if df.empty:
        return "Brak danych do wyświetlenia."
    
    # Kopiuj DataFrame żeby nie modyfikować oryginału
    df_formatted = df.copy()
    
    # Formatuj kolumny numeryczne
    numeric_columns = ['Średnia', 'Mediana', 'Std Dev', 'Min', 'Max']
    
    for col in numeric_columns:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: format_scientific_notation(x, 2)
            )
    
    # Tworzenie tabeli
    return tabulate(
        df_formatted,
        headers='keys',
        tablefmt=format_type,
        showindex=False,
        stralign='center',
        numalign='right'
    )


def save_results(df, output_dir, base_filename="summary_table"):
    """
    Zapisuje wyniki w różnych formatach.
    
    Args:
        df: DataFrame z wynikami
        output_dir: Katalog docelowy
        base_filename: Podstawowa nazwa pliku
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV (surowe dane)
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Zapisano CSV: {csv_path}")
    
    # Markdown
    md_path = os.path.join(output_dir, f"{base_filename}.md")
    with open(md_path, 'w') as f:
        f.write("# Tabela podsumowująca jakość rozwiązań\n\n")
        f.write(create_formatted_table(df, 'github'))
    print(f"Zapisano Markdown: {md_path}")
    
    # LaTeX
    latex_path = os.path.join(output_dir, f"{base_filename}.tex")
    with open(latex_path, 'w') as f:
        f.write("% Tabela podsumowująca jakość rozwiązań\n")
        f.write(create_formatted_table(df, 'latex'))
    print(f"Zapisano LaTeX: {latex_path}")
    
    # Sformatowany tekst
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, 'w') as f:
        f.write("Tabela podsumowująca jakość rozwiązań\n")
        f.write("=" * 50 + "\n\n")
        f.write(create_formatted_table(df, 'grid'))
    print(f"Zapisano tekst: {txt_path}")


def main():
    """Główna funkcja skryptu."""
    parser = argparse.ArgumentParser(
        description="Generuje tabelę podsumowującą jakość rozwiązań"
    )
    parser.add_argument(
        'results_dir',
        help='Ścieżka do katalogu z wynikami eksperymentów'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Katalog docelowy dla tabel (domyślnie: bieżący katalog)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['grid', 'markdown', 'latex', 'simple', 'github'],
        default='grid',
        help='Format wyświetlania tabeli (domyślnie: grid)'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Zapisz wyniki do plików'
    )
    
    args = parser.parse_args()
    
    try:
        # Wczytanie danych
        print(f"Wczytywanie danych z: {args.results_dir}")
        experiments = load_experiment_data(args.results_dir)
        
        if not experiments:
            print("Nie znaleziono danych eksperymentów!")
            return 1
        
        # Agregacja wyników
        print("Agregacja wyników...")
        summary_df = aggregate_results(experiments)
        
        if summary_df.empty:
            print("Brak danych do zagregowania!")
            return 1
        
        # Wyświetlenie tabeli
        print(f"\nTabela podsumowująca jakość rozwiązań:")
        print("=" * 80)
        table = create_formatted_table(summary_df, args.format)
        print(table)
        
        # Zapisanie wyników jeśli wymagane
        if args.save:
            print(f"\nZapisywanie wyników do: {args.output_dir}")
            save_results(summary_df, args.output_dir)
        
        # Dodatkowe statystyki
        print(f"\nPodsumowanie:")
        print(f"- Liczba funkcji testowych: {summary_df['Funkcja'].nunique()}")
        print(f"- Liczba wymiarów: {summary_df['Wymiar'].nunique()}")
        print(f"- Liczba algorytmów: {summary_df['Algorithm'].nunique()}")
        print(f"- Liczba generatorów: {summary_df['Generator'].nunique()}")
        print(f"- Łączna liczba konfiguracji: {len(summary_df)}")
        
        return 0
        
    except Exception as e:
        print(f"Błąd: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 