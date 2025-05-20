#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standardowa implementacja algorytmu CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
W tej wersji wektor ścieżki ewolucyjnej (pσ) jest inicjalizowany standardowo jako wektor zerowy.
Implementacja oparta jest na bibliotece pycma.
"""

import numpy as np
import cma
from typing import Callable, Dict, Any, Optional, Tuple


class StandardCMAES:
    """
    Standardowa implementacja algorytmu CMA-ES wykorzystująca bibliotekę pycma.
    """
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 initial_mean: np.ndarray,
                 initial_sigma: float = 1.0,
                 population_size: Optional[int] = None,
                 random_generator = None):
        """
        Inicjalizacja algorytmu CMA-ES.
        
        Args:
            objective_function: Funkcja celu do optymalizacji.
            initial_mean: Początkowy wektor średniej.
            initial_sigma: Początkowa wartość sigma (odchylenie standardowe).
            population_size: Rozmiar populacji. Jeśli None, zostanie obliczony automatycznie.
            random_generator: Generator liczb losowych. Może być używany do ustawienia ziarna (seed) dla algorytmu.
        """
        # Zapisz parametry
        self.objective_function = objective_function
        self.initial_mean = np.array(initial_mean, dtype=np.float64)
        self.initial_sigma = initial_sigma
        self.dimension = len(initial_mean)
        
        # Inicjalizacja opcji CMA-ES
        self.options = cma.CMAOptions()
        
        # Ustawienie rozmiaru populacji
        if population_size is not None:
            self.options['popsize'] = population_size
        
        # Ustawienie ziarna losowego, jeśli dostarczono generator
        if random_generator is not None and hasattr(random_generator, 'get_seed'):
            seed = random_generator.get_seed()
            if seed is not None:
                self.options['seed'] = seed
        
        # Inicjalizacja obiektu CMAEvolutionStrategy
        self.es = cma.CMAEvolutionStrategy(
            self.initial_mean, 
            self.initial_sigma, 
            inopts=self.options
        )
        
        # Inicjalizacja statystyk
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def ask(self) -> np.ndarray:
        """
        Generuje nową populację osobników.
        
        Returns:
            Macierz punktów (wiersze to osobniki, kolumny to wymiary).
        """
        return self.es.ask()
    
    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """
        Aktualizuje stan algorytmu na podstawie wyników oceny populacji.
        
        Args:
            solutions: Macierz punktów (wiersze to osobniki, kolumny to wymiary).
            fitnesses: Wektor wartości funkcji celu dla każdego osobnika.
        """
        # Aktualizuj statystyki
        self.evaluations += len(fitnesses)
        self.iterations += 1
        
        # Znajdź najlepsze rozwiązanie w aktualnej populacji
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_solution = solutions[best_idx].copy()
        
        # Przekaż wyniki do CMA-ES
        self.es.tell(solutions, fitnesses)
    
    def optimize(self, 
                max_evaluations: int = 1000, 
                ftol: float = 1e-8, 
                xtol: float = 1e-8, 
                verbose: bool = False) -> Dict[str, Any]:
        """
        Przeprowadza pełną optymalizację.
        
        Args:
            max_evaluations: Maksymalna liczba ewaluacji funkcji celu.
            ftol: Tolerancja zbieżności dla wartości funkcji.
            xtol: Tolerancja zbieżności dla parametrów.
            verbose: Czy wyświetlać postęp optymalizacji.
        
        Returns:
            Słownik z wynikami optymalizacji.
        """
        # Resetuj statystyki
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Inicjalizacja obiektu CMAEvolutionStrategy od nowa (czyste uruchomienie)
        self.es = cma.CMAEvolutionStrategy(
            self.initial_mean, 
            self.initial_sigma, 
            inopts=self.options
        )
        
        # Wykonaj główną pętlę optymalizacji
        previous_best = float('inf')
        
        while not self.es.stop() and self.evaluations < max_evaluations:
            # Generuj populację
            solutions = self.es.ask()
            
            # Oceniaj populację
            fitnesses = np.array([self.objective_function(x) for x in solutions])
            
            # Aktualizuj stan algorytmu
            self.tell(solutions, fitnesses)
            
            # Wyświetl postęp
            if verbose and self.iterations % 10 == 0:
                print(f"Iteracja {self.iterations}, ewaluacje {self.evaluations}, "
                      f"najlepsza wartość {self.best_fitness:.6e}, sigma {self.es.sigma:.6e}")
            
            # Sprawdź kryteria stopu
            if np.abs(previous_best - self.best_fitness) < ftol:
                if verbose:
                    print(f"Zatrzymano: zbieżność funkcji (delta={np.abs(previous_best - self.best_fitness):.6e})")
                break
            
            if self.es.sigma < xtol:
                if verbose:
                    print(f"Zatrzymano: zbieżność parametrów (sigma={self.es.sigma:.6e})")
                break
            
            previous_best = self.best_fitness
        
        # Przygotuj wynik w formacie zgodnym z poprzednią implementacją
        result = {
            'x': self.best_solution,
            'fun': self.best_fitness,
            'nfev': self.evaluations,
            'nit': self.iterations,
            'sigma': self.es.sigma,
            'success': True
        }
        
        return result
    
    def get_ps(self) -> np.ndarray:
        """
        Zwraca aktualny wektor ścieżki ewolucyjnej pσ.
        
        Returns:
            Wektor ścieżki ewolucyjnej pσ.
        """
        return self.es.adapt_sigma.ps
