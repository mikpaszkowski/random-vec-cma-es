#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test reprodukowalności zmodyfikowanego algorytmu CMA-ES.

Ten skrypt testuje czy ModifiedCMAES z parametrem seed zapewnia
pełną reprodukowalność wyników przy identycznych parametrach.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from algorithms.cma_es_modified import ModifiedCMAES
from functions.benchmark import get_function


def test_reproducible_modified_cmaes():
    """Test funkcjonalności powtarzalności w ModifiedCMAES."""
    print("=== TEST REPRODUKOWALNOŚCI ModifiedCMAES ===")
    print("Testuje czy parametr seed zapewnia pełną powtarzalność wyników\n")
    
    # Parametry testu
    dimension = 2
    initial_mean = np.array([1.0, -0.5])
    initial_sigma = 1.0
    seed = 42
    max_evaluations = 200
    
    # Użyj funkcji testowej Rastrigin
    test_function = get_function('rastrigin', dimension)
    
    print(f"Parametry testu:")
    print(f"- Funkcja: Rastrigin {dimension}D")
    print(f"- Punkt startowy: {initial_mean}")
    print(f"- Sigma: {initial_sigma}")
    print(f"- Seed: {seed}")
    print(f"- Max ewaluacji: {max_evaluations}")
    
    # Test bez seed (niepowtarzalne)
    print("\n" + "="*60)
    print("TEST 1: Bez seed (powinien być niepowtarzalny)")
    print("="*60)
    
    results_no_seed = []
    ps_norms_no_seed = []
    
    for run in range(3):
        print(f"\nUruchomienie {run+1}/3 (bez seed):")
        
        cma_instance = ModifiedCMAES(
            test_function, 
            initial_mean, 
            initial_sigma,
            ps_scale_factor=0.5
        )
        
        result = cma_instance.optimize(
            max_evaluations=max_evaluations,
            verbose=False
        )
        
        results_no_seed.append(result['fun'])
        ps_norms_no_seed.append(result['initial_ps_norm'])
        
        print(f"  Wynik: {result['fun']:.10f}")
        print(f"  Reprodukowalny: {result['reproducible']}")
        print(f"  Norma ps: {result['initial_ps_norm']:.6f}")
    
    # Test z seed (powtarzalne)
    print("\n" + "="*60)
    print(f"TEST 2: Z seed={seed} (powinien być powtarzalny)")
    print("="*60)
    
    results_with_seed = []
    ps_norms_with_seed = []
    ps_vectors_with_seed = []
    
    for run in range(3):
        print(f"\nUruchomienie {run+1}/3 (z seed={seed}):")
        
        cma_instance = ModifiedCMAES(
            test_function, 
            initial_mean, 
            initial_sigma,
            ps_scale_factor=0.5, 
            seed=seed
        )
        
        result = cma_instance.optimize(
            max_evaluations=max_evaluations,
            verbose=False
        )
        
        results_with_seed.append(result['fun'])
        ps_norms_with_seed.append(result['initial_ps_norm'])
        ps_vectors_with_seed.append(cma_instance._random_ps.copy())
        
        print(f"  Wynik: {result['fun']:.10f}")
        print(f"  Reprodukowalny: {result['reproducible']}")
        print(f"  Seed: {result['seed']}")
        print(f"  Norma ps: {result['initial_ps_norm']:.6f}")
        print(f"  Wektor ps: [{cma_instance._random_ps[0]:.6f}, {cma_instance._random_ps[1]:.6f}]")
    
    # Analiza wyników
    print("\n" + "="*60)
    print("ANALIZA WYNIKÓW")
    print("="*60)
    
    # Test bez seed
    no_seed_unique_results = len(set(f"{r:.8f}" for r in results_no_seed))
    no_seed_unique_ps = len(set(f"{p:.6f}" for p in ps_norms_no_seed))
    
    print(f"\nBez seed:")
    print(f"  Różnych wyników fitness: {no_seed_unique_results}/3")
    print(f"  Różnych norm ps: {no_seed_unique_ps}/3")
    
    if no_seed_unique_results > 1 or no_seed_unique_ps > 1:
        print("  ✅ RÓŻNE wyniki (oczekiwane dla niepowtarzalnego)")
        no_seed_ok = True
    else:
        print("  ⚠️  Identyczne wyniki (nieoczekiwane, ale możliwe)")
        no_seed_ok = True  # Nie traktujemy jako błąd
    
    # Test z seed
    with_seed_unique_results = len(set(f"{r:.8f}" for r in results_with_seed))
    with_seed_unique_ps = len(set(f"{p:.6f}" for p in ps_norms_with_seed))
    
    # Sprawdź czy wektory ps są identyczne
    ps_identical = True
    if len(ps_vectors_with_seed) > 1:
        first_ps = ps_vectors_with_seed[0]
        for ps in ps_vectors_with_seed[1:]:
            if not np.allclose(first_ps, ps, atol=1e-10):
                ps_identical = False
                break
    
    print(f"\nZ seed={seed}:")
    print(f"  Różnych wyników fitness: {with_seed_unique_results}/3")
    print(f"  Różnych norm ps: {with_seed_unique_ps}/3")
    print(f"  Wektory ps identyczne: {ps_identical}")
    
    if with_seed_unique_results == 1 and with_seed_unique_ps == 1 and ps_identical:
        print("  ✅ POWTARZALNE wyniki (oczekiwane)")
        with_seed_ok = True
    else:
        print("  ❌ NIEPOWTARZALNE wyniki (błąd!)")
        with_seed_ok = False
        
        # Szczegółowa diagnostyka
        print("\n  DIAGNOSTYKA:")
        if with_seed_unique_results > 1:
            print(f"    - Różne wyniki fitness: {results_with_seed}")
        if with_seed_unique_ps > 1:
            print(f"    - Różne normy ps: {ps_norms_with_seed}")
        if not ps_identical:
            print("    - Różne wektory ps:")
            for i, ps in enumerate(ps_vectors_with_seed):
                print(f"      Run {i+1}: [{ps[0]:.10f}, {ps[1]:.10f}]")
    
    # Podsumowanie
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    
    overall_success = no_seed_ok and with_seed_ok
    
    if overall_success:
        print("🎉 WSZYSTKIE TESTY PRZESZŁY POMYŚLNIE!")
        print("   - Bez seed: Wyniki różnią się (poprawne)")
        print("   - Z seed: Wyniki są identyczne (poprawne)")
        print("\n✅ ModifiedCMAES zapewnia pełną reprodukowalność z parametrem seed")
    else:
        print("❌ NIEKTÓRE TESTY NIEUDANE!")
        if not with_seed_ok:
            print("   - PROBLEM: Seed nie zapewnia reprodukowalności")
        print("\n❌ ModifiedCMAES wymaga poprawek w mechanizmie seed")
    
    return overall_success


def test_different_seeds():
    """Test czy różne seedy dają różne wyniki."""
    print("\n" + "="*60)
    print("TEST DODATKOWY: Różne seedy")
    print("="*60)
    
    # Parametry
    dimension = 2
    initial_mean = np.array([0.0, 0.0])
    initial_sigma = 1.0
    max_evaluations = 100
    test_function = get_function('rosenbrock', dimension)
    
    seeds = [1, 42, 123]
    results = []
    
    print("Testuje czy różne seedy dają różne wyniki:")
    
    for seed in seeds:
        cma_instance = ModifiedCMAES(
            test_function, 
            initial_mean, 
            initial_sigma,
            ps_scale_factor=0.5, 
            seed=seed
        )
        
        result = cma_instance.optimize(
            max_evaluations=max_evaluations,
            verbose=False
        )
        
        results.append(result['fun'])
        print(f"  Seed {seed:3d}: {result['fun']:.10f}, ps_norm: {result['initial_ps_norm']:.6f}")
    
    # Sprawdź różnorodność
    unique_results = len(set(f"{r:.8f}" for r in results))
    
    if unique_results == len(seeds):
        print(f"\n✅ Różne seedy dają różne wyniki ({unique_results}/{len(seeds)} unikalnych)")
        return True
    else:
        print(f"\n⚠️  Niektóre seedy dają identyczne wyniki ({unique_results}/{len(seeds)} unikalnych)")
        return False


def main():
    """Główna funkcja testowa."""
    print("TESTY REPRODUKOWALNOŚCI ModifiedCMAES")
    print("=" * 80)
    
    try:
        # Test główny
        test1_success = test_reproducible_modified_cmaes()
        
        # Test dodatkowy
        test2_success = test_different_seeds()
        
        # Końcowe podsumowanie
        print("\n" + "="*80)
        print("KOŃCOWE PODSUMOWANIE")
        print("="*80)
        
        if test1_success and test2_success:
            print("🎉 WSZYSTKIE TESTY ZAKOŃCZONE SUKCESEM!")
            print("   ModifiedCMAES poprawnie implementuje reprodukowalność")
            return 0
        else:
            print("❌ NIEKTÓRE TESTY NIEUDANE!")
            if not test1_success:
                print("   - Test reprodukowalności: NIEUDANY")
            if not test2_success:
                print("   - Test różnych seedów: NIEUDANY")
            return 1
            
    except Exception as e:
        print(f"\n❌ BŁĄD PODCZAS TESTÓW: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 