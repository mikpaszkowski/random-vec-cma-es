#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt uruchomieniowy do przeprowadzania eksperymentów porównujących standardowy i zmodyfikowany
algorytm CMA-ES z różnymi funkcjami testowymi i generatorami liczb losowych.
"""

import os
import sys
import argparse
from datetime import datetime

# Dodanie ścieżki głównej projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import komponentów projektu
from experiments.analysis.create_statistical_analysis import SimpleStatisticalAnalysis
from experiments.experiment_runner import ExperimentRunner
from experiments.config import (
    DEFAULT_DIMENSIONS, DEFAULT_FUNCTIONS, DEFAULT_ALGORITHMS, DEFAULT_GENERATORS,
    DEFAULT_SEEDS, DEFAULT_MAX_EVALUATIONS, DEFAULT_FTOL, DEFAULT_XTOL,
    DEFAULT_INITIAL_SIGMA, get_ftarget_stop_value
)
from experiments.analysis.create_individual_function_plots import create_individual_function_plots
from experiments.analysis.create_comparative_plots import create_comparative_plots
from experiments.analysis.generate_summary_table import generate_summary_table
from experiments.analysis.generate_comparison_table import generate_comparison_table


def run_all_experiments(output_dir='results/final'):
  
    
    # Przygotowanie konfiguracji
    config = {
        'dimensions': DEFAULT_DIMENSIONS,
        'functions': DEFAULT_FUNCTIONS,
        'algorithms': DEFAULT_ALGORITHMS,
        'generators': DEFAULT_GENERATORS,
        'seeds': DEFAULT_SEEDS,
        'max_evaluations': DEFAULT_MAX_EVALUATIONS,
        'ftol': DEFAULT_FTOL,
        'xtol': DEFAULT_XTOL,
        'initial_sigma': DEFAULT_INITIAL_SIGMA,
        'population_size': None,
        'ftarget_stop_value': get_ftarget_stop_value(DEFAULT_FUNCTIONS, DEFAULT_DIMENSIONS)
    }
    
    # Utworzenie i uruchomienie eksperymentów z wybraną biblioteką
    runner = ExperimentRunner(config, output_dir, library="pycma")
    
    print(f"Rozpoczynam eksperymenty z następującą konfiguracją:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Wyniki będą zapisywane w: {output_dir}")
    
    # Uruchomienie eksperymentów
    start_time = datetime.now()
    results = runner.run_experiment_batch(**config)
    end_time = datetime.now()
    
    print(f"\nEksperymenty zakończone!")
    print(f"Czas wykonania: {end_time - start_time}")
    print(f"Liczba udanych eksperymentów: {len(results)}")
    print(f"Wyniki zapisane w: {output_dir}")


def main():
    """Główna funkcja uruchamiająca eksperymenty."""
    
    # Uruchomienie eksperymentów
    run_all_experiments(output_dir='results/final')
    
    # Uruchomienie generowania wykresów zbieżności oraz sigma
    create_individual_function_plots()
    
    # # Uruchomienie generowania wykresów zbiorczych dla zbieznosci
    create_comparative_plots()
    
    #Uruchomienie generowania tabeli z wynikami
    generate_summary_table(results_dir='results/final')
    
    # #Uruchomienie generowania tabeli z wynikami analizy statystycznej
    analyzer = SimpleStatisticalAnalysis("results/final/all_results_summary.csv")
    analyzer.run_analysis()
    
    #Uruchomienie generowania tabeli z wynikami
    generate_comparison_table(results_dir='results/final')
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 