#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfiguracje domyślne dla eksperymentów z algorytmem CMA-ES.
"""

import os
import numpy as np

# Domyślne wartości dla eksperymentów
DEFAULT_DIMENSIONS = [2, 10, 30]
DEFAULT_FUNCTIONS = ['rosenbrock', 'rastrigin', 'ackley', 'schwefel']
DEFAULT_ALGORITHMS = ['standard', 'modified']
DEFAULT_GENERATORS = ['mt', 'pcg']
DEFAULT_SEEDS = list(range(1, 31))  # 30 powtórzeń
DEFAULT_MAX_EVALUATIONS = 50000  # Zwiększone z 10000 dla trudnych funkcji
DEFAULT_FTOL = 1e-9  # Zmniejszone z 1e-6 dla większej cierpliwości w funkcjach multimodalnych
DEFAULT_XTOL = 1e-9  # Zmniejszone z 1e-6 dla `większej cierpliwości
DEFAULT_INITIAL_SIGMA = 2.0  # Zwiększone z 1.0 dla lepszej eksploracji

# Przedziały odstępu między zapisywaniem danych zbieżności (co ile ewaluacji)
CONVERGENCE_INTERVAL = 50 

# Tolerancje specyficzne dla różnych funkcji
FUNCTION_SPECIFIC_TOLERANCES = {
    'rosenbrock': {'ftol': 1e-8, 'xtol': 1e-8},
    'rastrigin': {'ftol': 1e-12, 'xtol': 1e-12},  # Bardzo tolerancyjne dla multimodalnej
    'ackley': {'ftol': 1e-10, 'xtol': 1e-10},
    'schwefel': {'ftol': 1e-12, 'xtol': 1e-12}  # Bardzo tolerancyjne dla multimodalnej
}

# Struktura katalogów dla wyników
RESULTS_DIR_STRUCTURE = {
    'raw_data': 'raw_data',
    'convergence_plots': 'plots/convergence',
    'comparison_plots': 'plots/comparison',
    'statistics': 'statistics',
    'seeds': 'seeds'
}

# Progi dokładności dla różnych funkcji
ACCURACY_THRESHOLDS = {
    'rosenbrock': {
        '2D': 1e-6,
        '10D': 1e-6,
        '30D': 1e-6
        },
    'rastrigin': {
        '2D': 1e-3,  # Złagodzone - funkcja multimodalna
        '10D': 1e-2,  # Złagodzone z 1e-5
        '30D': 1e-1   # Złagodzone z 1e-4
    },
    'ackley': {
        '2D': 1e-6,
        '10D': 1e-5,
        '30D': 1e-4
    },
    'schwefel': {
        '2D': 50.0,   # Złagodzone z 100.0
        '10D': 200.0, # Złagodzone z 500.0
        '30D': 500.0  # Złagodzone z 1500.0
    }
}

# Ustawienia punktów początkowych dla różnych funkcji - POPRAWIONE
def get_initial_point(function_name: str, dimension: int, random_generator=None):
    """
    Zwraca punkt początkowy dla danej funkcji z opcjonalnym generatorem losowym.
    
    Args:
        function_name: Nazwa funkcji
        dimension: Wymiar przestrzeni
        random_generator: Generator liczb losowych (jeśli None, używa np.random)
    
    Returns:
        Punkt początkowy jako numpy array
    """
    if random_generator is None:
        random_generator = np.random
    
    if function_name == 'rosenbrock':
        return np.full(dimension, -2.0)
    elif function_name == 'rastrigin':
        if hasattr(random_generator, 'uniform'):
            return random_generator.uniform(-2.0, 2.0, dimension)
        else:
            return np.random.uniform(-2.0, 2.0, dimension)
    elif function_name == 'ackley':
        if hasattr(random_generator, 'uniform'):
            return random_generator.uniform(-15.0, 15.0, dimension)
        else:
            return np.random.uniform(-15.0, 15.0, dimension)
    elif function_name == 'schwefel':
        if hasattr(random_generator, 'uniform'):
            return random_generator.uniform(300.0, 500.0, dimension)
        else:
            return np.random.uniform(300.0, 500.0, dimension)
    else:
        # Fallback dla nieznanych funkcji
        return np.zeros(dimension)

def get_ftarget_stop_value(function_name: str, dimension: int) -> float | None:
    """
    Zwraca docelową wartość funkcji celu (ftarget), przy której algorytm powinien się zatrzymać,
    dla danej funkcji i jej wymiarowości.

    Args:
        function_name: Nazwa funkcji testowej.
        dimension: Wymiarowość problemu.

    Returns:
        Wartość ftarget lub None, jeśli nie zdefiniowano dla danej funkcji.
    """
    if function_name == 'rosenbrock':
        return 1e-8
    elif function_name == 'rastrigin':
        return 1e-4
    elif function_name == 'ackley':
        return 1e-5
    elif function_name == 'schwefel':
        # Globalne minimum dla Schwefela (w zakresie [-500, 500]) to ok. -dimension * 418.9829
        # Chcemy być blisko tego minimum, np. z dokładnością 0.01
        return -dimension * 418.9829 + 0.01
    else:
        # Można tu zwrócić domyślną wartość, None, lub zgłosić błąd
        # dla nieznanych funkcji, w zależności od potrzeb.
        print(f"Ostrzeżenie: Nie zdefiniowano ftarget dla funkcji '{function_name}'.")
        return None
