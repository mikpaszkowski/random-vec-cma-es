#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ulepszona implementacja zmodyfikowanego algorytmu CMA-ES.
Poprawia problemy z obecnej implementacji ModifiedCMAES.
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


class ModifiedCMAES(StandardCMAES):
    """
    Ulepszona implementacja zmodyfikowanego CMA-ES z losowym ps.
    
    Poprawki względem ModifiedCMAES:
    1. Skalowanie losowego ps aby dopasować się do typowych wartości
    2. Uproszczone zarządzanie stanem
    3. Bezpieczniejsze wywołania optimize()
    4. Dodana funkcjonalność reproducible z kontrolą seed
    """
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 initial_mean: np.ndarray,
                 initial_sigma: float = 1.0,
                 population_size: Optional[int] = None,
                 random_generator = None,
                 ps_scale_factor: float = 0.5,
                 seed: Optional[int] = None):
        """
        Inicjalizacja ulepszonej wersji algorytmu CMA-ES.
        
        Args:
            objective_function: Funkcja celu do optymalizacji.
            initial_mean: Początkowy wektor średniej.
            initial_sigma: Początkowa wartość sigma (odchylenie standardowe).
            population_size: Rozmiar populacji. Jeśli None, zostanie obliczony automatycznie.
            random_generator: Generator liczb losowych.
            ps_scale_factor: Współczynnik skalowania losowego ps (domyślnie 0.5).
            seed: Seed dla pełnej kontroli nad losowością i powtarzalnością.
        """
        # Wywołaj inicjalizację klasy bazowej
        super().__init__(objective_function, initial_mean, initial_sigma, population_size, random_generator)
        
        # Zachowaj parametry
        self.master_seed = seed
        self.ps_scale_factor = ps_scale_factor
        
        # Ustaw seed w opcjach CMA-ES jeśli podany
        if seed is not None:
            self.options['seed'] = seed
        
        # Konfiguruj generator losowy
        if seed is not None:
            self.random_generator = np.random.RandomState(seed + 1)  # +1 aby odróżnić od głównego seed
        elif random_generator is not None:
            self.random_generator = random_generator
        else:
            self.random_generator = np.random
        
        # Flaga oznaczająca czy ps zostało już zainicjalizowane
        self._ps_initialized = False
        self._random_ps = None
        
        # Generuj losowy ps przy inicjalizacji
        self._generate_random_ps()
    
    def _generate_random_ps(self):
        """Generuje powtarzalny losowy wektor ps z odpowiednim skalowaniem."""
        if hasattr(self.random_generator, 'randn'):
            raw_ps = self.random_generator.randn(self.dimension)
        else:
            raw_ps = np.random.randn(self.dimension)
        
        # Skalowanie ps aby lepiej pasowało do typowych wartości CMA-ES
        # Typowe ps ma normę około 1.0-1.5, więc skalujemy N(0,1) który ma E[|x|] ≈ sqrt(d)
        expected_norm = np.sqrt(self.dimension)
        target_norm = self.ps_scale_factor * expected_norm
        
        self._random_ps = raw_ps * (target_norm / np.linalg.norm(raw_ps))
        
        seed_info = f" (seed: {self.master_seed})" if self.master_seed is not None else ""
        print(f"DEBUG: Wygenerowano {'powtarzalny ' if self.master_seed is not None else ''}losowy ps o normie {np.linalg.norm(self._random_ps):.4f}{seed_info}")
    
    def _apply_random_ps(self):
        """Aplikuje losowy ps do obiektu CMA-ES."""
        if (not self._ps_initialized and 
            hasattr(self.es, 'adapt_sigma') and 
            hasattr(self.es.adapt_sigma, 'ps')):
            
            self.es.adapt_sigma.ps = self._random_ps.copy()
            self._ps_initialized = True
            seed_info = f" (seed: {self.master_seed})" if self.master_seed is not None else ""
            print(f"DEBUG: Zastąpiono ps {'powtarzalnym ' if self.master_seed is not None else ''}wektorem: {self._random_ps}{seed_info}")
    
    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """
        Aktualizuje stan algorytmu i aplikuje losowy ps po pierwszym tell().
        """
        # Wywołaj standardową metodę tell
        super().tell(solutions, fitnesses)
        
        # Zastąp ps losowym wektorem po pierwszym tell()
        self._apply_random_ps()
    
    def optimize(self, 
                max_evaluations: int = 1000, 
                ftol: float = 1e-8, 
                xtol: float = 1e-8, 
                ftarget_stop: Optional[float] = None,
                verbose: bool = False,
                convergence_interval: int = 100) -> Dict[str, Any]:
        """
        Przeprowadza pełną optymalizację z losowym ps i opcjonalną powtarzalnością.
        
        POPRAWA: Uproszczona logika - zawsze tworzymy nowy obiekt es.
        DODANO: Pełna kontrola nad powtarzalnością przez seed.
        """
        # KLUCZOWE: Ustaw wszystkie seedy na początku optimize dla powtarzalności
        if self.master_seed is not None:
            np.random.seed(self.master_seed)
            self.options['seed'] = self.master_seed
        
        # Resetuj statystyki
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # KLUCZOWA POPRAWA: Zawsze twórz nowy obiekt es dla spójności
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
        
        # Resetuj flagi inicjalizacji ps
        self._ps_initialized = False
        
        # Przygotuj dane do śledzenia zbieżności
        convergence_data = []
        last_saved = 0
        
        # Główna pętla optymalizacji
        while not self.es.stop() and self.evaluations < max_evaluations:
            # Generuj populację
            solutions = self.ask()
            
            # Oceń populację
            fitnesses = np.array([self.objective_function(x) for x in solutions])
            
            # Aktualizuj stan (w tym aplikuj losowy ps)
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
        
        # Komunikat o powodach zatrzymania
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
            'reproducible': self.master_seed is not None
        }
        
        return result
    
    def get_ps(self) -> np.ndarray:
        """
        Zwraca aktualny wektor ścieżki ewolucyjnej pσ.
        """
        if hasattr(self.es, 'adapt_sigma') and hasattr(self.es.adapt_sigma, 'ps'):
            return self.es.adapt_sigma.ps.copy()
        elif self._random_ps is not None:
            return self._random_ps.copy()
        else:
            return np.zeros(self.dimension)


class CustomCMAES:
    """
    Alternatywne podejście: własna klasa dziedzicząca bezpośrednio z CMAEvolutionStrategy.
    To jest najbezpieczniejsze podejście, ale wymaga więcej kodu.
    """
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 initial_mean: np.ndarray,
                 initial_sigma: float = 1.0,
                 ps_scale_factor: float = 0.5,
                 random_seed: Optional[int] = None):
        """
        Inicjalizacja niestandardowego CMA-ES z kontrolą nad ps.
        """
        self.objective_function = objective_function
        self.initial_mean = np.array(initial_mean, dtype=np.float64)
        self.initial_sigma = initial_sigma
        self.dimension = len(initial_mean)
        self.ps_scale_factor = ps_scale_factor
        
        # Opcje CMA-ES
        self.options = {}
        if random_seed is not None:
            self.options['seed'] = random_seed
        
        # Statystyki
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Generator losowy
        self.rng = np.random.RandomState(random_seed)
    
    def _create_custom_cma(self, options):
        """Tworzy niestandardowy obiekt CMA-ES z modyfikacją ps."""
        
        # Utwórz standardowy CMA-ES
        es = cma.CMAEvolutionStrategy(self.initial_mean, self.initial_sigma, options)
        
        # Wygeneruj losowy ps z odpowiednim skalowaniem
        raw_ps = self.rng.randn(self.dimension)
        expected_norm = np.sqrt(self.dimension)
        target_norm = self.ps_scale_factor * expected_norm
        scaled_ps = raw_ps * (target_norm / np.linalg.norm(raw_ps))
        
        # HOOK: Zastąp inicjalizację ps w pierwszym ask/tell
        original_tell = es.tell
        ps_set = [False]  # Użyj listy dla closure
        
        def custom_tell(solutions, fitnesses):
            result = original_tell(solutions, fitnesses)
            if not ps_set[0] and hasattr(es.adapt_sigma, 'ps'):
                es.adapt_sigma.ps = scaled_ps.copy()
                ps_set[0] = True
                print(f"CUSTOM: Ustawiono ps na: {scaled_ps}")
            return result
        
        es.tell = custom_tell
        return es, scaled_ps
    
    def optimize(self, max_evaluations: int = 1000, verbose: bool = False):
        """Optymalizacja z niestandardowym ps."""
        
        options = self.options.copy()
        options['maxfevals'] = max_evaluations
        if verbose:
            options['verbose'] = 1
        else:
            options['verbose'] = -9
        
        # Utwórz niestandardowy CMA-ES
        self.es, initial_ps = self._create_custom_cma(options)
        
        # Resetuj statystyki
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Główna pętla
        while not self.es.stop() and self.evaluations < max_evaluations:
            self.iterations += 1
            
            solutions = self.es.ask()
            fitnesses = np.array([self.objective_function(x) for x in solutions])
            self.es.tell(solutions, fitnesses)
            
            self.evaluations += len(fitnesses)
            
            # Aktualizuj najlepsze rozwiązanie
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = solutions[best_idx].copy()
            
            if verbose and self.iterations % 10 == 0:
                ps_norm = np.linalg.norm(self.es.adapt_sigma.ps) if hasattr(self.es.adapt_sigma, 'ps') else 0
                print(f"Iter {self.iterations}: best={self.best_fitness:.6e}, ps_norm={ps_norm:.4f}")
        
        return {
            'x': self.best_solution,
            'fun': self.best_fitness,
            'nfev': self.evaluations,
            'nit': self.iterations,
            'initial_ps': initial_ps,
            'initial_ps_norm': np.linalg.norm(initial_ps)
        } 