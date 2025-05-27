#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zmodyfikowana implementacja algorytmu CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
używająca biblioteki cmaes CyberAgentAILab.
W tej wersji symulujemy losową inicjalizację ścieżki ewolucyjnej przez modyfikację początkowej macierzy kowariancji.
"""

import numpy as np
import cmaes
from typing import Callable, Dict, Any, Optional, Tuple
from algorithms.cmaes_standard import StandardCMAESLib


class ModifiedCMAESLib(StandardCMAESLib):
    """
    Zmodyfikowana implementacja algorytmu CMA-ES używająca biblioteki cmaes CyberAgentAILab.
    
    Ponieważ biblioteka cmaes nie pozwala na bezpośrednią modyfikację wektora ścieżki ewolucyjnej pσ,
    symulujemy efekt losowej inicjalizacji przez:
    1. Modyfikację początkowej macierzy kowariancji
    2. Przechowywanie symulowanego wektora ps dla zachowania zgodności interfejsu
    """
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 initial_mean: np.ndarray,
                 initial_sigma: float = 1.0,
                 population_size: Optional[int] = None,
                 random_generator = None):
        """
        Inicjalizacja zmodyfikowanej wersji algorytmu CMA-ES.
        
        Args:
            objective_function: Funkcja celu do optymalizacji.
            initial_mean: Początkowy wektor średniej.
            initial_sigma: Początkowa wartość sigma (odchylenie standardowe).
            population_size: Rozmiar populacji. Jeśli None, zostanie obliczony automatycznie.
            random_generator: Generator liczb losowych. Może być używany do ustawienia ziarna (seed)
                              oraz do generowania losowej modyfikacji.
        """
        # Zachowaj generator losowy
        self.random_generator = random_generator if random_generator is not None else np.random
        
        # Zapisz parametry podstawowe
        self.objective_function = objective_function
        self.initial_mean = np.array(initial_mean, dtype=np.float64)
        self.initial_sigma = initial_sigma
        self.dimension = len(initial_mean)
        
        # Wygeneruj losowy wektor ps dla symulacji
        self.simulated_ps = self._generate_random_ps()
        
        # Ustalenie ziarna losowego
        seed = None
        if random_generator is not None and hasattr(random_generator, 'get_seed'):
            seed = random_generator.get_seed()
        
        # Wygeneruj zmodyfikowaną macierz kowariancji na podstawie losowego ps
        modified_cov = self._create_modified_covariance_matrix()
        
        # Inicjalizacja obiektu CMA z zmodyfikowaną macierzą kowariancji
        self.optimizer = cmaes.CMA(
            mean=self.initial_mean.copy(),
            sigma=self.initial_sigma,
            population_size=population_size,
            seed=seed,
            cov=modified_cov  # Użyj zmodyfikowanej macierzy kowariancji
        )
        
        # Inicjalizacja statystyk
        self.evaluations = 0
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.current_population = []
        self.current_fitness_values = []
    
    def _generate_random_ps(self) -> np.ndarray:
        """
        Generuje losowy wektor ścieżki ewolucyjnej pσ.
        
        Returns:
            Losowy wektor pσ z rozkładu normalnego.
        """
        if hasattr(self.random_generator, 'randn'):
            return self.random_generator.randn(self.dimension)
        else:
            return np.random.randn(self.dimension)
    
    def _create_modified_covariance_matrix(self) -> np.ndarray:
        """
        Tworzy zmodyfikowaną macierz kowariancji bazując na losowym wektorze ps.
        
        Idea: Macierz kowariancji C w CMA-ES jest związana z ps przez aktualizację rang-μ.
        Aby symulować efekt losowej inicjalizacji ps, modyfikujemy początkową macierz
        kowariancji poprzez dodanie komponentu zewnętrznego produktu losowego wektora.
        
        Returns:
            Zmodyfikowana macierz kowariancji.
        """
        # Rozpocznij od macierzy jednostkowej
        base_cov = np.eye(self.dimension)
        
        # Normalizuj symulowany ps
        ps_normalized = self.simulated_ps / (np.linalg.norm(self.simulated_ps) + 1e-12)
        
        # Dodaj komponent związany z ps jako aktualizację rang-1
        # Używamy małego współczynnika żeby nie zdominować macierzy
        alpha = 0.1  # Siła modyfikacji
        ps_update = alpha * np.outer(ps_normalized, ps_normalized)
        
        # Zmodyfikowana macierz kowariancji
        modified_cov = base_cov + ps_update
        
        # Upewnij się, że macierz jest dodatnio określona
        eigenvals, eigenvecs = np.linalg.eigh(modified_cov)
        eigenvals = np.maximum(eigenvals, 1e-12)  # Zapewnienie pozytywności
        modified_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return modified_cov
    
    def optimize(self, 
                max_evaluations: int = 1000, 
                ftol: float = 1e-8, 
                xtol: float = 1e-8, 
                verbose: bool = False,
                convergence_interval: int = 100) -> Dict[str, Any]:
        """
        Przeprowadza pełną optymalizację z zmodyfikowaną inicjalizacją.
        
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
        
        # Wygeneruj nowy losowy ps dla każdej optymalizacji
        self.simulated_ps = self._generate_random_ps()
        
        # Ustalenie ziarna losowego
        seed = None
        if hasattr(self, 'random_generator') and self.random_generator is not None:
            if hasattr(self.random_generator, 'get_seed'):
                seed = self.random_generator.get_seed()
        
        # Utwórz nową zmodyfikowaną macierz kowariancji
        modified_cov = self._create_modified_covariance_matrix()
        
        # Reinicjalizuj optimizer z nową zmodyfikowaną macierzą kowariancji
        self.optimizer = cmaes.CMA(
            mean=self.initial_mean.copy(),
            sigma=self.initial_sigma,
            population_size=self.optimizer.population_size,
            seed=seed,
            cov=modified_cov
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
        Zwraca symulowany wektor ścieżki ewolucyjnej pσ.
        
        Returns:
            Symulowany wektor ścieżki ewolucyjnej.
        """
        return self.simulated_ps.copy() 