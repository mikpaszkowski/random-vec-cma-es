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
                verbose: bool = False,
                convergence_interval: int = 100) -> Dict[str, Any]:
        """
        Przeprowadza pełną optymalizację.
        
        Args:
            max_evaluations: Maksymalna liczba ewaluacji funkcji celu.
            ftol: Tolerancja zbieżności dla wartości funkcji.
            xtol: Tolerancja zbieżności dla parametrów.
            verbose: Czy wyświetlać postęp optymalizacji.
            convergence_interval: Odstęp między zapisywaniem danych do krzywej zbieżności.
        
        Returns:
            Słownik z wynikami optymalizacji.
        """
        # Resetuj statystyki
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # POPRAWIONE: Ustaw opcje CMA-ES zamiast implementować własne kryteria stopu
        options = self.options.copy()
        options['maxfevals'] = max_evaluations
        options['tolfun'] = ftol  # Używamy wbudowanej tolerancji funkcji CMA-ES
        options['tolx'] = xtol    # Używamy wbudowanej tolerancji parametrów CMA-ES
        if verbose:
            options['verbose'] = 1
        else:
            options['verbose'] = -9  # Wyłącz wszystkie komunikaty
        
        # Inicjalizacja obiektu CMAEvolutionStrategy od nowa z nowymi opcjami
        self.es = cma.CMAEvolutionStrategy(
            self.initial_mean, 
            self.initial_sigma, 
            inopts=options
        )
        
        # Przygotuj dane do śledzenia zbieżności
        convergence_data = []
        last_saved = 0
        
        # Wykonaj główną pętlę optymalizacji - użyj TYLKO kryteriów CMA-ES
        while not self.es.stop() and self.evaluations < max_evaluations:
            # Generuj populację
            solutions = self.es.ask()
            
            # Oceniaj populację
            fitnesses = np.array([self.objective_function(x) for x in solutions])
            
            # Aktualizuj stan algorytmu
            self.tell(solutions, fitnesses)
            
            # Zapisz dane zbieżności
            should_save = (
                self.iterations == 1 or  # Pierwsza iteracja
                self.evaluations - last_saved >= convergence_interval or  # Normalny interwał
                self.iterations % 5 == 0  # Co 5 iteracji dla lepszej granulacji
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
                print(f"Iteracja {self.iterations}, ewaluacje {self.evaluations}, "
                      f"najlepsza wartość {self.best_fitness:.6e}, sigma {self.es.sigma:.6e}")
        
        # Komunikat o powodach zatrzymania
        if verbose:
            if self.es.stop():
                print(f"Zatrzymano przez CMA-ES: {self.es.stop()}")
            elif self.evaluations >= max_evaluations:
                print(f"Zatrzymano: osiągnięto maksymalną liczbę ewaluacji ({max_evaluations})")
        
        # ZAWSZE zapisz ostatni punkt zbieżności
        if not convergence_data or self.evaluations > last_saved:
            convergence_data.append({
                'evaluations': self.evaluations,
                'best_fitness': float(self.best_fitness),
                'sigma': float(self.es.sigma)
            })
        
        # Dodaj punkt początkowy na początku listy, ale tylko jeśli mamy rzeczywiste dane
        if convergence_data and self.best_fitness != float('inf'):
            # Wstaw początkowy punkt na początku z pierwszą rzeczywistą wartością funkcji
            # zamiast inf, użyj pierwszej rzeczywistej wartości
            initial_point = {
                'evaluations': 0,
                'best_fitness': convergence_data[0]['best_fitness'],  # Użyj pierwszej prawdziwej wartości
                'sigma': float(self.initial_sigma)
            }
            convergence_data.insert(0, initial_point)
        
        # Przygotuj wynik w formacie zgodnym z poprzednią implementacją
        result = {
            'x': self.best_solution,
            'fun': self.best_fitness,
            'nfev': self.evaluations,
            'nit': self.iterations,
            'sigma': self.es.sigma,
            'success': True,
            'convergence_data': convergence_data
        }
        
        return result
    
    def get_ps(self) -> np.ndarray:
        """
        Zwraca aktualny wektor ścieżki ewolucyjnej pσ.
        
        Returns:
            Wektor ścieżki ewolucyjnej pσ.
        """
        # W bibliotece pycma (cma) ścieżka ewolucyjna sigma jest dostępna jako atrybut ps
        # bezpośrednio w obiekcie CMAEvolutionStrategy, a nie w adapt_sigma
        if hasattr(self.es, 'ps'):
            return self.es.ps.copy()
        else:
            # Jeśli wektor ps nie istnieje, zwróć wektor zerowy
            return np.zeros(self.dimension)
