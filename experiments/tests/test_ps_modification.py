#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt testujący modyfikację wektora ps w CMA-ES.
Sprawdza czy modyfikacja rzeczywiście działa i jak wpływa na zachowanie algorytmu.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Dodaj ścieżkę do modułów projektu
sys.path.insert(0, os.path.abspath('.'))

from algorithms.cma_es_standard import StandardCMAES
from algorithms.cma_es_modified import ModifiedCMAES
from functions.benchmark import get_function


class PSInspector:
    """Klasa do monitorowania wektora ps podczas optymalizacji."""
    
    def __init__(self):
        self.ps_history = []
        self.iteration_history = []
    
    def record_ps(self, cma_instance, iteration):
        """Zapisuje aktualny wektor ps."""
        ps = cma_instance.get_ps()
        self.ps_history.append(ps.copy())
        self.iteration_history.append(iteration)
        return ps


def test_ps_initialization():
    """Test sprawdzający inicjalizację wektora ps."""
    print("=== Test inicjalizacji wektora ps ===")
    
    # Parametry testowe
    dimension = 5
    initial_mean = np.zeros(dimension)
    initial_sigma = 1.0
    seed = 42
    
    # Generator z ustaloną wartością seed
    np.random.seed(seed)
    
    # Funkcja testowa (nie jest istotna dla tego testu)
    test_function = lambda x: np.sum(x**2)
    
    print(f"Wymiar: {dimension}")
    print(f"Seed: {seed}")
    print()
    
    # Test StandardCMAES
    print("--- StandardCMAES ---")
    np.random.seed(seed)  # Reset seed
    standard_cma = StandardCMAES(test_function, initial_mean, initial_sigma)
    
    # Sprawdź ps przed pierwszą iteracją
    ps_before = standard_cma.get_ps()
    print(f"PS przed pierwszą iteracją: {ps_before}")
    print(f"Czy ps to wektor zerowy: {np.allclose(ps_before, 0)}")
    
    # Jedna iteracja ask/tell
    solutions = standard_cma.ask()
    fitnesses = np.array([test_function(x) for x in solutions])
    standard_cma.tell(solutions, fitnesses)
    
    ps_after = standard_cma.get_ps()
    print(f"PS po pierwszej iteracji: {ps_after}")
    print(f"Norma ps: {np.linalg.norm(ps_after):.6f}")
    print()
    
    # Test ModifiedCMAES
    print("--- ModifiedCMAES ---")
    np.random.seed(seed)  # Reset seed
    
    # Stwórz generator z tym samym seed
    rng = np.random.RandomState(seed)
    
    modified_cma = ModifiedCMAES(test_function, initial_mean, initial_sigma, random_generator=rng)
    
    # Sprawdź ps przed pierwszą iteracją
    ps_before_mod = modified_cma.get_ps()
    print(f"PS przed pierwszą iteracją: {ps_before_mod}")
    print(f"Czy ps to wektor zerowy: {np.allclose(ps_before_mod, 0)}")
    print(f"Norma ps: {np.linalg.norm(ps_before_mod):.6f}")
    
    # Jedna iteracja ask/tell
    solutions = modified_cma.ask()
    fitnesses = np.array([test_function(x) for x in solutions])
    modified_cma.tell(solutions, fitnesses)
    
    ps_after_mod = modified_cma.get_ps()
    print(f"PS po pierwszej iteracji: {ps_after_mod}")
    print(f"Norma ps: {np.linalg.norm(ps_after_mod):.6f}")
    print()
    
    # Porównanie
    print("--- Porównanie ---")
    print(f"Standard ps po tell: {ps_after}")
    print(f"Modified ps po tell: {ps_after_mod}")
    print(f"Czy są różne: {not np.allclose(ps_after, ps_after_mod)}")
    print(f"Różnica norm: {np.linalg.norm(ps_after_mod) - np.linalg.norm(ps_after):.6f}")


def test_ps_evolution():
    """Test ewolucji wektora ps przez kilka iteracji."""
    print("\n=== Test ewolucji wektora ps ===")
    
    # Parametry
    dimension = 2
    initial_mean = np.array([2.0, -1.0])
    initial_sigma = 1.0
    seed = 123
    max_iterations = 10
    
    # Funkcja Rosenbrocka
    test_function = get_function('rosenbrock', dimension)
    
    print(f"Funkcja: {test_function.name}")
    print(f"Wymiar: {dimension}")
    print(f"Punkt startowy: {initial_mean}")
    print(f"Iteracje: {max_iterations}")
    print()
    
    # Inspektorzy ps
    standard_inspector = PSInspector()
    modified_inspector = PSInspector()
    
    # Standard CMA-ES
    print("--- Uruchamianie Standard CMA-ES ---")
    np.random.seed(seed)
    standard_cma = StandardCMAES(test_function, initial_mean, initial_sigma)
    
    for i in range(max_iterations):
        ps_before = standard_inspector.record_ps(standard_cma, i)
        
        solutions = standard_cma.ask()
        fitnesses = np.array([test_function(x) for x in solutions])
        standard_cma.tell(solutions, fitnesses)
        
        if i < 3:  # Pokaż pierwsze iteracje
            ps_after = standard_cma.get_ps()
            print(f"Iter {i}: ps_before={ps_before}, ps_after={ps_after}")
    
    print()
    
    # Modified CMA-ES
    print("--- Uruchamianie Modified CMA-ES ---")
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    modified_cma = ModifiedCMAES(test_function, initial_mean, initial_sigma, random_generator=rng)
    
    for i in range(max_iterations):
        ps_before = modified_inspector.record_ps(modified_cma, i)
        
        solutions = modified_cma.ask()
        fitnesses = np.array([test_function(x) for x in solutions])
        modified_cma.tell(solutions, fitnesses)
        
        if i < 3:  # Pokaż pierwsze iteracje
            ps_after = modified_cma.get_ps()
            print(f"Iter {i}: ps_before={ps_before}, ps_after={ps_after}")
    
    # Analiza różnic
    print("\n--- Analiza różnic w ewolucji ps ---")
    for i in range(min(len(standard_inspector.ps_history), len(modified_inspector.ps_history))):
        ps_std = standard_inspector.ps_history[i]
        ps_mod = modified_inspector.ps_history[i]
        norm_diff = np.linalg.norm(ps_mod) - np.linalg.norm(ps_std)
        print(f"Iter {i}: |ps_std|={np.linalg.norm(ps_std):.4f}, |ps_mod|={np.linalg.norm(ps_mod):.4f}, diff={norm_diff:.4f}")
    
    return standard_inspector, modified_inspector


def test_internal_cma_state():
    """Test sprawdzający wewnętrzny stan obiektu CMA."""
    print("\n=== Test wewnętrznego stanu CMA ===")
    
    dimension = 3
    initial_mean = np.zeros(dimension)
    initial_sigma = 1.0
    seed = 42
    
    test_function = lambda x: np.sum(x**2)
    
    print("Sprawdzanie dostępu do wewnętrznych struktur CMA...")
    
    # Test na ModifiedCMAES
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    modified_cma = ModifiedCMAES(test_function, initial_mean, initial_sigma, random_generator=rng)
    
    print(f"Czy obiekt es istnieje: {hasattr(modified_cma, 'es')}")
    print(f"Typ obiektu es: {type(modified_cma.es)}")
    
    # Sprawdź strukturę adapt_sigma
    if hasattr(modified_cma.es, 'adapt_sigma'):
        print(f"Obiekt adapt_sigma istnieje: True")
        print(f"Typ adapt_sigma: {type(modified_cma.es.adapt_sigma)}")
        
        if hasattr(modified_cma.es.adapt_sigma, 'ps'):
            ps = modified_cma.es.adapt_sigma.ps
            print(f"Wektor ps istnieje: True")
            print(f"Typ ps: {type(ps)}")
            print(f"Kształt ps: {ps.shape}")
            print(f"Wartość ps przed iteracją: {ps}")
        else:
            print("Wektor ps nie istnieje w adapt_sigma")
    else:
        print("Obiekt adapt_sigma nie istnieje")
    
    # Pierwsza iteracja
    print("\nPo pierwszej iteracji ask/tell:")
    solutions = modified_cma.ask()
    fitnesses = np.array([test_function(x) for x in solutions])
    modified_cma.tell(solutions, fitnesses)
    
    if hasattr(modified_cma.es, 'adapt_sigma') and hasattr(modified_cma.es.adapt_sigma, 'ps'):
        ps_after = modified_cma.es.adapt_sigma.ps
        print(f"Wartość ps po iteracji: {ps_after}")
        print(f"Czy ps zostało zmodyfikowane: {not np.allclose(ps_after, 0)}")


def test_optimization_comparison():
    """Porównanie pełnej optymalizacji między standard a modified."""
    print("\n=== Test porównania optymalizacji ===")
    
    dimension = 2
    initial_mean = np.array([2.0, -1.0])
    initial_sigma = 1.0
    seed = 42
    max_evaluations = 500
    
    test_function = get_function('rastrigin', dimension)
    
    print(f"Funkcja: {test_function.name}")
    print(f"Maksymalne ewaluacje: {max_evaluations}")
    print()
    
    # Standard CMA-ES
    print("--- Standard CMA-ES ---")
    np.random.seed(seed)
    standard_cma = StandardCMAES(test_function, initial_mean, initial_sigma)
    
    result_std = standard_cma.optimize(
        max_evaluations=max_evaluations,
        verbose=True,
        ftol=1e-8,
        xtol=1e-8
    )
    
    print(f"Wynik Standard: {result_std['fun']:.6e}")
    print(f"Ewaluacje: {result_std['nfev']}")
    print(f"Iteracje: {result_std['nit']}")
    print()
    
    # Modified CMA-ES
    print("--- Modified CMA-ES ---")
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    modified_cma = ModifiedCMAES(test_function, initial_mean, initial_sigma, random_generator=rng)
    
    result_mod = modified_cma.optimize(
        max_evaluations=max_evaluations,
        verbose=True,
        ftol=1e-8,
        xtol=1e-8
    )
    
    print(f"Wynik Modified: {result_mod['fun']:.6e}")
    print(f"Ewaluacje: {result_mod['nfev']}")
    print(f"Iteracje: {result_mod['nit']}")
    print()
    
    # Porównanie
    print("--- Porównanie wyników ---")
    print(f"Różnica w wyniku: {result_mod['fun'] - result_std['fun']:.6e}")
    print(f"Różnica w ewaluacjach: {result_mod['nfev'] - result_std['nfev']}")
    print(f"Różnica w iteracjach: {result_mod['nit'] - result_std['nit']}")
    
    return result_std, result_mod


def main():
    """Główna funkcja testowa."""
    print("TESTY MODYFIKACJI WEKTORA PS W CMA-ES")
    print("=" * 50)
    
    try:
        # Test 1: Inicjalizacja
        test_ps_initialization()
        
        # Test 2: Ewolucja ps
        standard_inspector, modified_inspector = test_ps_evolution()
        
        # Test 3: Wewnętrzny stan
        test_internal_cma_state()
        
        # Test 4: Porównanie optymalizacji
        result_std, result_mod = test_optimization_comparison()
        
        print("\n" + "=" * 50)
        print("PODSUMOWANIE TESTÓW")
        print("=" * 50)
        print("Wszystkie testy zakończone pomyślnie!")
        
    except Exception as e:
        print(f"\nBŁĄD podczas testów: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 