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
DEFAULT_MAX_EVALUATIONS = 10000
DEFAULT_FTOL = 1e-6
DEFAULT_XTOL = 1e-6
DEFAULT_INITIAL_SIGMA = 1.0

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
    'rosenbrock': 1e-6,
    'rastrigin': 1e-6,
    'ackley': 1e-6,
    'schwefel': 1e-6
}

# Ustawienia punktów początkowych dla różnych funkcji
INITIAL_POINTS = {
    'rosenbrock': lambda dim: np.full(dim, -2.0),
    'rastrigin': lambda dim: np.full(dim, 2.0),
    'ackley': lambda dim: np.full(dim, 10.0),
    'schwefel': lambda dim: np.full(dim, 200.0)
}

# Przedziały odstępu między zapisywaniem danych zbieżności (co ile ewaluacji)
CONVERGENCE_INTERVAL = 50 