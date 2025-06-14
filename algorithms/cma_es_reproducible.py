#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
W pełni powtarzalne implementacje CMA-ES.
Bazuje na diagnostyce problemu z powtarzalnością.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path if running directly
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import numpy as np
import cma
from typing import Callable, Dict, Any, Optional, Tuple

# Try to import from algorithms package, fallback to direct import
try:
    from algorithms.cma_es_standard import StandardCMAES
except ImportError:
    from cma_es_standard import StandardCMAES


class ReproducibleStandardCMAES(StandardCMAES):
    """W pełni powtarzalna wersja StandardCMAES."""
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 initial_mean: np.ndarray,
                 initial_sigma: float = 1.0,
                 population_size: Optional[int] = None,
                 seed: Optional[int] = None):
        """
        Inicjalizacja z gwarantowaną powtarzalnością.
        
        Args:
            seed: Seed dla pełnej kontroli nad losowością.
        """
        super().__init__(objective_function, initial_mean, initial_sigma, population_size)
        
        self.master_seed = seed
        if seed is not None:
            self.options['seed'] = seed
    
    def optimize(self, **kwargs):
        """Optymalizacja z gwarantowaną powtarzalnością."""
        
        # KLUCZOWE: Ustaw wszystkie seedy na początku optimize
        if self.master_seed is not None:
            np.random.seed(self.master_seed)
            self.options['seed'] = self.master_seed
        
        return super().optimize(**kwargs)


class ReproducibleModifiedCMAES(StandardCMAES):
    """W pełni powtarzalna wersja ModifiedCMAES z losowym ps."""
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 initial_mean: np.ndarray,
                 initial_sigma: float = 1.0,
                 population_size: Optional[int] = None,
                 seed: Optional[int] = None,
                 ps_scale_factor: float = 0.5):
        """
        Inicjalizacja z losowym ps i gwarantowaną powtarzalnością.
        
        Args:
            seed: Seed dla pełnej kontroli nad losowością.
            ps_scale_factor: Współczynnik skalowania losowego ps.
        """
        super().__init__(objective_function, initial_mean, initial_sigma, population_size)
        
        self.master_seed = seed
        self.ps_scale_factor = ps_scale_factor
        
        if seed is not None:
            self.options['seed'] = seed
        
        # Flagi dla ps
        self._ps_initialized = False
        self._random_ps = None
        
        # Generuj losowy ps
        self._generate_random_ps()
    
    def _generate_random_ps(self):
        """Generuje powtarzalny losowy ps."""
        if self.master_seed is not None:
            rng = np.random.RandomState(self.master_seed + 1)  # +1 aby odróżnić od głównego seed
        else:
            rng = np.random
        
        # Generuj i skaluj ps
        raw_ps = rng.randn(self.dimension)
        expected_norm = np.sqrt(self.dimension)
        target_norm = self.ps_scale_factor * expected_norm
        
        self._random_ps = raw_ps * (target_norm / np.linalg.norm(raw_ps))
        
        print(f"DEBUG: Wygenerowano powtarzalny ps o normie {np.linalg.norm(self._random_ps):.4f}")
    
    def _apply_random_ps(self):
        """Aplikuje losowy ps."""
        if (not self._ps_initialized and 
            hasattr(self.es, 'adapt_sigma') and 
            hasattr(self.es.adapt_sigma, 'ps')):
            
            self.es.adapt_sigma.ps = self._random_ps.copy()
            self._ps_initialized = True
            print(f"DEBUG: Zastąpiono ps powtarzalnym wektorem: {self._random_ps}")
    
    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """Tell z aplikacją losowego ps."""
        super().tell(solutions, fitnesses)
        self._apply_random_ps()
    
    def optimize(self, **kwargs):
        """Optymalizacja z losowym ps i gwarantowaną powtarzalnością."""
        
        # KLUCZOWE: Ustaw wszystkie seedy
        if self.master_seed is not None:
            np.random.seed(self.master_seed)
            self.options['seed'] = self.master_seed
        
        # Resetuj flagi ps
        self._ps_initialized = False
        
        result = super().optimize(**kwargs)
        
        # Dodaj informacje o ps do wyniku
        result['ps_scale_factor'] = self.ps_scale_factor
        result['initial_ps_norm'] = np.linalg.norm(self._random_ps) if self._random_ps is not None else 0.0
        
        return result
    
    def get_ps(self) -> np.ndarray:
        """Zwraca aktualny wektor ps."""
        if hasattr(self.es, 'adapt_sigma') and hasattr(self.es.adapt_sigma, 'ps'):
            return self.es.adapt_sigma.ps.copy()
        elif self._random_ps is not None:
            return self._random_ps.copy()
        else:
            return np.zeros(self.dimension)


class ReproducibleImprovedCMAES(ReproducibleModifiedCMAES):
    """
    Najlepsza, w pełni powtarzalna implementacja z wszystkimi ulepszeniami.
    Dziedziczy po ReproducibleModifiedCMAES i dodaje dodatkowe funkcje.
    """
    
    def optimize(self, 
                max_evaluations: int = 1000, 
                ftol: float = 1e-8, 
                xtol: float = 1e-8, 
                ftarget_stop: Optional[float] = None,
                verbose: bool = False,
                convergence_interval: int = 100) -> Dict[str, Any]:
        """
        Optymalizacja z pełną funkcjonalnością i powtarzalnością.
        """
        # KLUCZOWE: Ustaw wszystkie seedy
        if self.master_seed is not None:
            np.random.seed(self.master_seed)
            self.options['seed'] = self.master_seed
        
        # Resetuj statystyki
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Przygotuj opcje
        options = self.options.copy()
        options['maxfevals'] = max_evaluations
        options['tolfun'] = ftol
        options['tolx'] = xtol
        
        if ftarget_stop is not None:
            options['ftarget'] = ftarget_stop
        
        if verbose:
            options['verbose'] = 1
        else:
            options['verbose'] = -9
        
        # Utwórz nowy obiekt CMA-ES
        self.es = cma.CMAEvolutionStrategy(
            self.initial_mean, 
            self.initial_sigma, 
            inopts=options
        )
        
        # Resetuj flagi ps
        self._ps_initialized = False
        
        # Przygotuj dane zbieżności
        convergence_data = []
        last_saved = 0
        
        # Główna pętla optymalizacji
        while not self.es.stop() and self.evaluations < max_evaluations:
            # Generuj populację
            solutions = self.ask()
            
            # Oceń populację
            fitnesses = np.array([self.objective_function(x) for x in solutions])
            
            # Aktualizuj stan (aplikuje losowy ps)
            self.tell(solutions, fitnesses)
            
            # Zapisz dane zbieżności
            should_save = (
                self.iterations == 1 or
                self.evaluations - last_saved >= convergence_interval or
                self.iterations % 5 == 0
            )
            
            if should_save:
                convergence_data.append({
                    'evaluations': self.evaluations,
                    'best_fitness': float(self.best_fitness),
                    'sigma': float(self.es.sigma)
                })
                last_saved = self.evaluations
            
            # Wyświetl postęp
            if verbose and self.iterations % 10 == 0:
                ps_norm = np.linalg.norm(self.get_ps())
                print(f"Iteracja {self.iterations}, ewaluacje {self.evaluations}, "
                      f"najlepsza wartość {self.best_fitness:.6e}, sigma {self.es.sigma:.6e}, "
                      f"ps_norm {ps_norm:.4f}")
        
        # Komunikat o zatrzymaniu
        if verbose:
            if self.es.stop():
                print(f"Zatrzymano przez CMA-ES: {self.es.stop()}")
            elif self.evaluations >= max_evaluations:
                print(f"Zatrzymano: osiągnięto maksymalną liczbę ewaluacji ({max_evaluations})")
        
        # Zapisz ostatni punkt zbieżności
        if not convergence_data or self.evaluations > last_saved:
            convergence_data.append({
                'evaluations': self.evaluations,
                'best_fitness': float(self.best_fitness),
                'sigma': float(self.es.sigma)
            })
        
        # Dodaj punkt początkowy
        if convergence_data and self.best_fitness != float('inf'):
            initial_point = {
                'evaluations': 0,
                'best_fitness': convergence_data[0]['best_fitness'],
                'sigma': float(self.initial_sigma)
            }
            convergence_data.insert(0, initial_point)
        
        # Przygotuj wynik
        result = {
            'x': self.best_solution,
            'fun': self.best_fitness,
            'nfev': self.evaluations,
            'nit': self.iterations,
            'sigma': self.es.sigma,
            'success': True,
            'convergence_data': convergence_data,
            'ps_scale_factor': self.ps_scale_factor,
            'initial_ps_norm': np.linalg.norm(self._random_ps) if self._random_ps is not None else 0.0,
            'seed': self.master_seed,
            'reproducible': True
        }
        
        return result


def demo_reproducibility():
    """Demonstracja powtarzalności wszystkich implementacji."""
    print("=== DEMONSTRACJA POWTARZALNOŚCI ===")
    
    from functions.benchmark import get_function
    
    # Parametry
    dimension = 2
    initial_mean = np.array([1.0, -0.5])
    initial_sigma = 1.0
    seed = 42
    max_evaluations = 200
    
    test_function = get_function('rastrigin', dimension)
    
    # Test wszystkich implementacji
    implementations = [
        (ReproducibleStandardCMAES, "ReproducibleStandardCMAES", {}),
        (ReproducibleModifiedCMAES, "ReproducibleModifiedCMAES", {'ps_scale_factor': 0.5}),
        (ReproducibleImprovedCMAES, "ReproducibleImprovedCMAES", {'ps_scale_factor': 0.5}),
    ]
    
    for impl_class, name, kwargs in implementations:
        print(f"\n--- Test {name} ---")
        
        results = []
        for run in range(3):
            cma_instance = impl_class(
                test_function, initial_mean, initial_sigma,
                seed=seed, **kwargs
            )
            
            result = cma_instance.optimize(
                max_evaluations=max_evaluations,
                verbose=False
            )
            
            results.append(result['fun'])
            print(f"Run {run+1}: {result['fun']:.10f}")
        
        # Sprawdź powtarzalność
        all_same = len(set(f"{r:.8f}" for r in results)) == 1
        print(f"✅ {name}: POWTARZALNE" if all_same else f"❌ {name}: NIEPOWTARZALNE")


if __name__ == "__main__":
    demo_reproducibility() 