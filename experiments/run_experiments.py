#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt uruchomieniowy do przeprowadzania eksperymentów porównujących standardowy i zmodyfikowany
algorytm CMA-ES z różnymi funkcjami testowymi i generatorami liczb losowych.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Dodanie ścieżki głównej projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import komponentów projektu
from experiments.experiment_runner import ExperimentRunner
from experiments.config import (
    DEFAULT_DIMENSIONS, DEFAULT_FUNCTIONS, DEFAULT_ALGORITHMS, DEFAULT_GENERATORS,
    DEFAULT_SEEDS, DEFAULT_MAX_EVALUATIONS, DEFAULT_FTOL, DEFAULT_XTOL,
    DEFAULT_INITIAL_SIGMA
)
from experiments.utils import generate_summary_plots, calculate_wilcoxon_tests


def parse_arguments():
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description='Eksperymenty z CMA-ES')
    
    parser.add_argument('--functions', nargs='+', choices=DEFAULT_FUNCTIONS, default=DEFAULT_FUNCTIONS,
                       help='Funkcje testowe do użycia')
    parser.add_argument('--dimensions', type=int, nargs='+', default=DEFAULT_DIMENSIONS,
                       help='Wymiary przestrzeni do testowania')
    parser.add_argument('--algorithms', nargs='+', choices=DEFAULT_ALGORITHMS, default=DEFAULT_ALGORITHMS,
                       help='Wersje algorytmu do testowania')
    parser.add_argument('--generators', nargs='+', choices=DEFAULT_GENERATORS, default=DEFAULT_GENERATORS,
                       help='Generatory liczb losowych do użycia')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='Ziarna do użycia (domyślnie: DEFAULT_SEEDS)')
    parser.add_argument('--max-evaluations', type=int, default=DEFAULT_MAX_EVALUATIONS,
                       help='Maksymalna liczba ewaluacji funkcji')
    parser.add_argument('--ftol', type=float, default=DEFAULT_FTOL,
                       help='Tolerancja dla wartości funkcji')
    parser.add_argument('--xtol', type=float, default=DEFAULT_XTOL,
                       help='Tolerancja dla parametrów')
    parser.add_argument('--initial-sigma', type=float, default=DEFAULT_INITIAL_SIGMA,
                       help='Początkowa wartość sigma')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Katalog wynikowy (domyślnie: results/YYYYMMDD_HHMMSS)')
    parser.add_argument('--parallel', action='store_true',
                       help='Czy uruchamiać eksperymenty równolegle (nie zaimplementowane)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Liczba procesów równoległych (nie zaimplementowane)')
    parser.add_argument('--only-plots', action='store_true',
                       help='Tylko generuj wykresy dla istniejących wyników')
    parser.add_argument('--only-stats', action='store_true',
                       help='Tylko generuj statystyki dla istniejących wyników')
    
    return parser.parse_args()


def main():
    """Główna funkcja uruchamiająca eksperymenty."""
    args = parse_arguments()
    
    # Ustalenie katalogu wynikowego
    if args.output_dir:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('results', timestamp)
        os.makedirs(output_dir, exist_ok=True)
    
    # Jeśli tylko generujemy wykresy lub statystyki
    if args.only_plots or args.only_stats:
        if args.only_plots:
            print(f"Generowanie wykresów dla wyników w katalogu: {output_dir}")
            generate_summary_plots(output_dir)
        if args.only_stats:
            print(f"Generowanie statystyk dla wyników w katalogu: {output_dir}")
            calculate_wilcoxon_tests(output_dir)
        return 0
    
    # Przygotowanie konfiguracji
    config = {
        'dimensions': args.dimensions,
        'functions': args.functions,
        'algorithms': args.algorithms,
        'generators': args.generators,
        'seeds': args.seeds or DEFAULT_SEEDS,
        'max_evaluations': args.max_evaluations,
        'ftol': args.ftol,
        'xtol': args.xtol,
        'initial_sigma': args.initial_sigma
    }
    
    # Utworzenie i uruchomienie eksperymentów
    runner = ExperimentRunner(config, output_dir)
    
    print(f"Rozpoczynam eksperymenty z następującą konfiguracją:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Wyniki będą zapisywane w: {output_dir}")
    
    # Uruchomienie eksperymentów
    try:
        results = runner.run_experiment_batch(**config)
        
        # Generowanie wykresów i statystyk
        print("Generowanie wykresów podsumowujących...")
        generate_summary_plots(output_dir)
        
        print("Przeprowadzanie analizy statystycznej...")
        calculate_wilcoxon_tests(output_dir)
        
        print(f"\nEksperymenty zakończone. Wyniki zapisane w: {output_dir}")
        return 0
    
    except KeyboardInterrupt:
        print("\nPrzerwano eksperymenty przez użytkownika.")
        return 1
    
    except Exception as e:
        print(f"\nBłąd podczas przeprowadzania eksperymentów: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 