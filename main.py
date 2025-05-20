#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Główny skrypt uruchomieniowy dla projektu Random Vector CMA-ES.
"""

import argparse
import sys

def parse_arguments():
    """Parsuje argumenty linii poleceń."""
    parser = argparse.ArgumentParser(description='Random Vector CMA-ES')
    
    parser.add_argument('--algorithm', type=str, choices=['standard', 'modified'],
                        default='modified', help='Wersja algorytmu CMA-ES')
    parser.add_argument('--generator', type=str, choices=['mt', 'pcg'],
                        default='mt', help='Generator liczb losowych')
    parser.add_argument('--function', type=str, choices=['rosenbrock', 'rastrigin', 'ackley', 'schwefel'],
                        default='rosenbrock', help='Funkcja testowa')
    parser.add_argument('--dimension', type=int, choices=[2, 10, 30],
                        default=2, help='Wymiarowość problemu')
    parser.add_argument('--seed', type=int, default=None,
                        help='Ziarno generatora liczb losowych')
    return parser.parse_args()

def main():
    """Główna funkcja programu."""
    args = parse_arguments()
    
    print(f"Uruchamianie algorytmu {args.algorithm} z generatorem {args.generator}")
    print(f"Funkcja testowa: {args.function}, wymiar: {args.dimension}")
    
    # TODO: Implementacja logiki uruchomieniowej
    # 1. Załaduj odpowiedni algorytm (standardowy lub zmodyfikowany)
    # 2. Skonfiguruj generator liczb losowych
    # 3. Przygotuj funkcję testową
    # 4. Uruchom optymalizację
    # 5. Wyświetl/zapisz wyniki
    
    print("Funkcjonalność jeszcze nie zaimplementowana.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
