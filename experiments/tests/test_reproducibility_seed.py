#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Szczegółowa diagnostyka problemu z powtarzalnością eksperymentów CMA-ES.
"""

import numpy as np
import sys
import os
import cma

# Dodaj ścieżkę do modułów projektu
sys.path.insert(0, os.path.abspath('.'))

from algorithms.cma_es_standard import StandardCMAES
from algorithms.cma_es_modified import ModifiedCMAES
from algorithms.cma_es_modified import ImprovedCMAES
from functions.benchmark import get_function


def test_numpy_seed_control():
    """Test kontroli nad seedami numpy."""
    print("=== TEST KONTROLI NAD SEEDAMI NUMPY ===")
    
    seed = 12345
    
    # Test 1: Czy numpy.random.seed działa
    print("--- Test 1: numpy.random.seed ---")
    results = []
    for i in range(3):
        np.random.seed(seed)
        random_values = np.random.randn(5)
        results.append(random_values)
        print(f"Run {i+1}: {random_values}")
    
    all_same = all(np.allclose(results[0], r) for r in results[1:])
    print(f"Czy wszystkie identyczne: {all_same}")
    print()
    
    # Test 2: Czy RandomState działa
    print("--- Test 2: RandomState ---")
    results_rs = []
    for i in range(3):
        rng = np.random.RandomState(seed)
        random_values = rng.randn(5)
        results_rs.append(random_values)
        print(f"Run {i+1}: {random_values}")
    
    all_same_rs = all(np.allclose(results_rs[0], r) for r in results_rs[1:])
    print(f"Czy wszystkie identyczne: {all_same_rs}")
    print()


def test_pycma_seed_control():
    """Test kontroli nad seedami w pycma."""
    print("=== TEST KONTROLI NAD SEEDAMI PYCMA ===")
    
    dimension = 2
    initial_mean = np.array([1.0, -0.5])
    initial_sigma = 0.8
    seed = 54321
    
    test_function = lambda x: np.sum(x**2)
    
    print("--- Test 1: Czy pycma respektuje seed w opcjach ---")
    results = []
    
    for i in range(3):
        print(f"\nRun {i+1}:")
        
        # Reset numpy seed
        np.random.seed(seed)
        
        # Utwórz CMA-ES z seedem
        options = {'seed': seed, 'verbose': -9}
        es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, options)
        
        # Jedna iteracja
        solutions = es.ask()
        fitnesses = [test_function(x) for x in solutions]
        es.tell(solutions, fitnesses)
        
        best_fitness = min(fitnesses)
        results.append(best_fitness)
        
        print(f"  Pierwsze rozwiązanie: {solutions[0]}")
        print(f"  Najlepsza wartość: {best_fitness:.10f}")
        print(f"  Seed usado por CMA: {es.opts.get('seed', 'BRAK')}")
    
    all_same = len(set(f"{r:.10f}" for r in results)) == 1
    print(f"\nCzy wszystkie wyniki identyczne: {all_same}")
    if not all_same:
        print(f"Różnice: {[abs(r - results[0]) for r in results[1:]]}")
    print()


def test_cmaes_class_reproducibility():
    """Test powtarzalności naszych klas CMA-ES."""
    print("=== TEST POWTARZALNOŚCI KLAS CMA-ES ===")
    
    dimension = 2
    initial_mean = np.array([1.0, -0.5])
    initial_sigma = 0.8
    max_evaluations = 100  # Krótki test
    seed = 98765
    
    test_function = get_function('rosenbrock', dimension)
    
    # Test dla każdej implementacji
    implementations = [
        (StandardCMAES, "StandardCMAES"),
        (ModifiedCMAES, "ModifiedCMAES"),  
        (ImprovedCMAES, "ImprovedCMAES")
    ]
    
    for impl_class, impl_name in implementations:
        print(f"--- Test {impl_name} ---")
        
        results = []
        
        for run in range(3):
            print(f"\nRun {run+1}:")
            
            # KLUCZOWE: Reset wszystkich seedów
            np.random.seed(seed)
            
            # Utwórz instancję
            if impl_class in [ModifiedCMAES, ImprovedCMAES]:
                rng = np.random.RandomState(seed)
                if impl_class == ImprovedCMAES:
                    cma_instance = impl_class(
                        test_function, initial_mean, initial_sigma,
                        random_generator=rng, ps_scale_factor=0.5
                    )
                else:
                    cma_instance = impl_class(
                        test_function, initial_mean, initial_sigma,
                        random_generator=rng
                    )
            else:
                cma_instance = impl_class(
                    test_function, initial_mean, initial_sigma
                )
            
            # WAŻNE: Ustaw seed bezpośrednio w opcjach CMA-ES
            cma_instance.options['seed'] = seed
            
            result = cma_instance.optimize(
                max_evaluations=max_evaluations,
                verbose=False
            )
            
            results.append(result['fun'])
            print(f"  Wynik: {result['fun']:.10f}")
            print(f"  Ewaluacje: {result['nfev']}")
            print(f"  CMA seed: {cma_instance.es.opts.get('seed', 'BRAK')}")
        
        # Sprawdź powtarzalność
        all_same = len(set(f"{r:.8f}" for r in results)) == 1
        print(f"\n✅ {impl_name}: POWTARZALNE" if all_same else f"❌ {impl_name}: NIEPOWTARZALNE")
        
        if not all_same:
            print(f"   Wartości: {[f'{r:.10f}' for r in results]}")
            print(f"   Różnice: {[abs(r - results[0]) for r in results[1:]]}")
        print()


def test_seed_propagation():
    """Test propagacji seedów w różnych częściach algorytmu."""
    print("=== TEST PROPAGACJI SEEDÓW ===")
    
    dimension = 2
    initial_mean = np.array([0.0, 0.0])
    initial_sigma = 1.0
    seed = 11111
    
    test_function = lambda x: np.sum(x**2)
    
    print("--- Test propagacji seed przez StandardCMAES ---")
    
    # Test 1: Tylko numpy seed
    print("\n1. Tylko np.random.seed:")
    np.random.seed(seed)
    cma1 = StandardCMAES(test_function, initial_mean, initial_sigma)
    
    # Test 2: Seed w opcjach
    print("\n2. Seed w opcjach CMA:")  
    np.random.seed(seed)
    cma2 = StandardCMAES(test_function, initial_mean, initial_sigma)
    cma2.options['seed'] = seed
    
    # Test 3: Seed ustawiony przed optimize
    print("\n3. Seed ustawiony przed optimize:")
    np.random.seed(seed)
    cma3 = StandardCMAES(test_function, initial_mean, initial_sigma)
    
    # Porównaj pierwsze rozwiązania
    for i, (cma, name) in enumerate([(cma1, "Tylko numpy"), (cma2, "Opcje"), (cma3, "Przed optimize")], 1):
        if i == 3:
            # Dla trzeciego, ustaw seed przed optimize
            cma.options['seed'] = seed
        
        solutions = cma.ask()
        print(f"{name}: pierwsze rozwiązanie = {solutions[0]}")
        print(f"  CMA seed: {cma.es.opts.get('seed', 'BRAK')}")


def create_reproducible_cmaes():
    """Tworzy w pełni powtarzalną wersję CMA-ES."""
    print("\n=== TWORZENIE W PEŁNI POWTARZALNEJ WERSJI ===")
    
    class ReproducibleCMAES(StandardCMAES):
        """W pełni powtarzalna wersja CMA-ES."""
        
        def __init__(self, objective_function, initial_mean, initial_sigma, 
                     seed=None, **kwargs):
            self.master_seed = seed
            super().__init__(objective_function, initial_mean, initial_sigma, **kwargs)
            
            if seed is not None:
                self.options['seed'] = seed
        
        def optimize(self, **kwargs):
            """Optymalizacja z gwarantowaną powtarzalnością."""
            
            # KLUCZOWE: Ustaw seed na początku optimize
            if self.master_seed is not None:
                np.random.seed(self.master_seed)
                self.options['seed'] = self.master_seed
            
            return super().optimize(**kwargs)
    
    # Test nowej klasy
    print("--- Test ReproducibleCMAES ---")
    
    dimension = 2
    initial_mean = np.array([1.0, -1.0])
    initial_sigma = 1.0
    seed = 77777
    
    test_function = get_function('rastrigin', dimension)
    
    results = []
    for run in range(3):
        cma = ReproducibleCMAES(
            test_function, initial_mean, initial_sigma,
            seed=seed
        )
        
        result = cma.optimize(max_evaluations=200, verbose=False)
        results.append(result['fun'])
        print(f"Run {run+1}: {result['fun']:.10f}")
    
    all_same = len(set(f"{r:.8f}" for r in results)) == 1
    print(f"✅ ReproducibleCMAES: POWTARZALNE" if all_same else f"❌ ReproducibleCMAES: NIEPOWTARZALNE")
    
    return ReproducibleCMAES


def diagnose_randomness_sources():
    """Diagnozuje źródła losowości w CMA-ES."""
    print("\n=== DIAGNOZA ŹRÓDEŁ LOSOWOŚCI ===")
    
    dimension = 2
    initial_mean = np.array([0.0, 0.0])
    initial_sigma = 1.0
    seed = 33333
    
    test_function = lambda x: np.sum(x**2)
    
    print("--- Śledzenie losowości w pycma ---")
    
    # Utwórz CMA-ES i śledź losowość
    np.random.seed(seed)
    options = {'seed': seed, 'verbose': -9}
    es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, options)
    
    print(f"1. Po inicjalizacji:")
    print(f"   CMA seed: {es.opts.get('seed')}")
    print(f"   Numpy state sample: {np.random.randn():.6f}")
    
    # Reset i test ask()
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, options)
    
    solutions1 = es.ask()
    print(f"\n2. Po pierwszym ask():")
    print(f"   Pierwsze rozwiązanie: {solutions1[0]}")
    
    # Reset i drugi test ask()
    np.random.seed(seed)
    es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, options)
    
    solutions2 = es.ask()
    print(f"\n3. Po drugim ask() (po reset):")
    print(f"   Pierwsze rozwiązanie: {solutions2[0]}")
    print(f"   Czy identyczne: {np.allclose(solutions1[0], solutions2[0])}")
    
    # Sprawdź czy problem jest w tell()
    fitnesses = [test_function(x) for x in solutions1]
    es.tell(solutions1, fitnesses)
    
    solutions3 = es.ask()
    print(f"\n4. Po tell() i kolejnym ask():")
    print(f"   Nowe rozwiązanie: {solutions3[0]}")


def main():
    """Główna funkcja diagnostyczna."""
    print("DIAGNOSTYKA PROBLEMU POWTARZALNOŚCI CMA-ES")
    print("=" * 60)
    
    try:
        # Test 1: Kontrola nad numpy
        test_numpy_seed_control()
        
        # Test 2: Kontrola nad pycma
        test_pycma_seed_control()
        
        # Test 3: Nasze implementacje
        test_cmaes_class_reproducibility()
        
        # Test 4: Propagacja seedów
        test_seed_propagation()
        
        # Test 5: Źródła losowości
        diagnose_randomness_sources()
        
        # Test 6: Powtarzalna wersja
        ReproducibleCMAES = create_reproducible_cmaes()
        
        print("\n" + "=" * 60)
        print("PODSUMOWANIE DIAGNOSTYKI")
        print("=" * 60)
        print("Sprawdź wyniki powyżej aby zidentyfikować problemy z powtarzalnością.")
        print("Główne podejrzenia:")
        print("1. pycma może używać wewnętrznych generatorów losowych")
        print("2. Seed może nie być propagowany poprawnie")
        print("3. Różne części algorytmu mogą używać różnych źródeł losowości")
        
    except Exception as e:
        print(f"BŁĄD podczas diagnostyki: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 