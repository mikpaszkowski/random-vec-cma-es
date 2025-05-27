#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standardowa implementacja algorytmu CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
używająca biblioteki cmaes CyberAgentAILab.
Ta implementacja jest alternatywą dla pycma i oferuje podobny interfejs ask-tell.
"""

import numpy as np
import cmaes
from typing import Callable, Dict, Any, Optional, Tuple


class StandardCMAESLib:
    """
    Standardowa implementacja algorytmu CMA-ES wykorzystująca bibliotekę cmaes CyberAgentAILab.
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
        
        # Ustalenie ziarna losowego
        seed = None
        if random_generator is not None and hasattr(random_generator, 'get_seed'):
            seed = random_generator.get_seed()
        
        # Inicjalizacja obiektu CMA
        self.optimizer = cmaes.CMA(
            mean=self.initial_mean.copy(),
            sigma=self.initial_sigma,
            population_size=population_size,
            seed=seed
        )
        
        # Inicjalizacja statystyk
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.current_population = []
        self.current_fitness_values = []
    
    def ask(self) -> np.ndarray:
        """
        Generuje nową populację osobników.
        
        Returns:
            Macierz punktów (wiersze to osobniki, kolumny to wymiary).
        """
        # Wygeneruj całą populację
        self.current_population = []
        for _ in range(self.optimizer.population_size):
            x = self.optimizer.ask()
            self.current_population.append(x)
        
        return np.array(self.current_population)
    
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
        
        # Przygotuj dane dla biblioteki cmaes - lista tupli (rozwiązanie, fitness)
        solutions_with_fitness = [(solutions[i], fitnesses[i]) for i in range(len(solutions))]
        
        # Przekaż wyniki do CMA-ES
        self.optimizer.tell(solutions_with_fitness)
    
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
        
        # Reinicjalizuj optimizer z nowym ziarnem jeśli potrzeba
        seed = None
        if hasattr(self, 'random_generator') and self.random_generator is not None:
            if hasattr(self.random_generator, 'get_seed'):
                seed = self.random_generator.get_seed()
        
        self.optimizer = cmaes.CMA(
            mean=self.initial_mean.copy(),
            sigma=self.initial_sigma,
            population_size=self.optimizer.population_size,
            seed=seed
        )
        
        # Przygotuj dane do śledzenia zbieżności
        convergence_data = []
        last_saved = 0
        
        # Pomocnicze zmienne dla kryteriów stopu
        prev_best = float('inf')
        stagnation_count = 0
        max_stagnation = 100  # Maksymalna liczba iteracji bez poprawy
        
        # Wykonaj główną pętlę optymalizacji
        while not self.optimizer.should_stop() and self.evaluations < max_evaluations:
            # Generuj populację
            solutions = self.ask()
            
            # Oceniaj populację
            fitnesses = np.array([self.objective_function(x) for x in solutions])
            
            # Aktualizuj stan algorytmu
            self.tell(solutions, fitnesses)
            
            # Sprawdź kryterium stopu dla ftol
            if abs(self.best_fitness - prev_best) < ftol:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best = self.best_fitness
            
            if stagnation_count >= max_stagnation:
                if verbose:
                    print(f"Zatrzymano: brak poprawy przez {max_stagnation} iteracji (ftol={ftol})")
                break
            
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
                    'sigma': float(self.initial_sigma)  # cmaes nie eksportuje sigma w prosty sposób
                })
                last_saved = self.evaluations
            
            # Wyświetl postęp
            if verbose and self.iterations % 10 == 0:
                print(f"Iteracja {self.iterations}, ewaluacje {self.evaluations}, "
                      f"najlepsza wartość {self.best_fitness:.6e}")
        
        # Komunikat o powodach zatrzymania
        if verbose:
            if self.optimizer.should_stop():
                print(f"Zatrzymano przez bibliotekę cmaes")
            elif self.evaluations >= max_evaluations:
                print(f"Zatrzymano: osiągnięto maksymalną liczbę ewaluacji ({max_evaluations})")
        
        # ZAWSZE zapisz ostatni punkt zbieżności
        if not convergence_data or self.evaluations > last_saved:
            convergence_data.append({
                'evaluations': self.evaluations,
                'best_fitness': float(self.best_fitness),
                'sigma': float(self.initial_sigma)
            })
        
        # Dodaj punkt początkowy na początku listy
        if convergence_data and self.best_fitness != float('inf'):
            initial_point = {
                'evaluations': 0,
                'best_fitness': convergence_data[0]['best_fitness'],
                'sigma': float(self.initial_sigma)
            }
            convergence_data.insert(0, initial_point)
        
        # Przygotuj wyniki w formacie zgodnym z pycma
        result = {
            'xbest': self.best_solution if self.best_solution is not None else self.initial_mean.copy(),
            'fbest': self.best_fitness,
            'fun': self.best_fitness,  # Alias dla zgodności
            'x': self.best_solution if self.best_solution is not None else self.initial_mean.copy(),
            'nfev': self.evaluations,
            'nit': self.iterations,
            'convergence_data': convergence_data
        }
        
        return result
    
    def get_ps(self) -> np.ndarray:
        """
        Zwraca wektor ścieżki ewolucyjnej pσ.
        
        Uwaga: Biblioteka cmaes nie eksportuje wewnętrznych zmiennych stanu jak ps.
        Zwracamy wektor zerowy dla zachowania zgodności interfejsu.
        
        Returns:
            Wektor ścieżki ewolucyjnej (wektor zerowy dla tej implementacji).
        """
        # Biblioteka cmaes nie eksportuje ps, zwracamy wektor zerowy
        return np.zeros(self.dimension)
    
    @property
    def sigma(self) -> float:
        """
        Zwraca aktualną wartość sigma.
        
        Uwaga: Biblioteka cmaes nie eksportuje sigma w prosty sposób.
        Zwracamy wartość początkową.
        
        Returns:
            Aktualna wartość sigma (wartość początkowa dla tej implementacji).
        """
        return self.initial_sigma 