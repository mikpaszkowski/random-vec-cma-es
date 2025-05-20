#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zmodyfikowana implementacja algorytmu CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
W tej wersji wektor ścieżki ewolucyjnej pσ jest inicjalizowany losowo zamiast wektorem zerowym.
Implementacja oparta jest na bibliotece pycma z modyfikacją inicjalizacji.
"""

import numpy as np
import cma
from typing import Callable, Dict, Any, Optional, Tuple
from algorithms.cma_es_standard import StandardCMAES


class ModifiedCMAES(StandardCMAES):
    """
    Zmodyfikowana implementacja algorytmu CMA-ES, gdzie wektor ścieżki ewolucyjnej pσ
    jest inicjalizowany losowo zamiast wektorem zerowym.
    
    Dziedziczy po klasie StandardCMAES i nadpisuje tylko inicjalizację w celu zachowania
    spójności logiki algorytmu przy minimalnej modyfikacji kodu.
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
                              oraz do generowania losowego wektora pσ.
        """
        # Wywołaj inicjalizację klasy bazowej
        super().__init__(objective_function, initial_mean, initial_sigma, population_size, random_generator)
        
        # Zachowaj generator losowy do późniejszej modyfikacji
        self.random_generator = random_generator if random_generator is not None else np.random
        
        # MODYFIKACJA: Po inicjalizacji standardowej, modyfikujemy wektor ścieżki ewolucyjnej pσ
        # aby był losowy zamiast zerowego
        self.initialize_random_ps()
    
    def initialize_random_ps(self):
        """
        Inicjalizuje ścieżkę ewolucyjną pσ losowo.
        """
        if hasattr(self.es, 'adapt_sigma') and hasattr(self.es.adapt_sigma, 'ps'):
            # Generuj losowy wektor z rozkładu normalnego
            if hasattr(self.random_generator, 'randn'):
                # Użyj naszego generatora
                random_ps = self.random_generator.randn(self.dimension)
            else:
                # Użyj standardowego generatora numpy
                random_ps = np.random.randn(self.dimension)
            
            # Przypisz losowy wektor do ścieżki ewolucyjnej pσ
            self.es.adapt_sigma.ps = random_ps
            
            # Możemy rozważyć różne strategie inicjalizacji:
            # 1. Losowa inicjalizacja z rozkładu normalnego (zastosowana powyżej)
            # 2. Znormalizowana losowa inicjalizacja, aby zachować oczekiwaną długość
            # self.es.adapt_sigma.ps = random_ps / np.linalg.norm(random_ps) * np.sqrt(self.dimension)
    
    def optimize(self, 
                max_evaluations: int = 1000, 
                ftol: float = 1e-8, 
                xtol: float = 1e-8, 
                verbose: bool = False) -> Dict[str, Any]:
        """
        Przeprowadza pełną optymalizację z losową inicjalizacją ścieżki ewolucyjnej pσ.
        
        Args:
            max_evaluations: Maksymalna liczba ewaluacji funkcji celu.
            ftol: Tolerancja zbieżności dla wartości funkcji.
            xtol: Tolerancja zbieżności dla parametrów.
            verbose: Czy wyświetlać postęp optymalizacji.
        
        Returns:
            Słownik z wynikami optymalizacji.
        """
        # Najpierw wykonujemy standardową inicjalizację z klasy bazowej
        result = super().optimize(max_evaluations, ftol, xtol, verbose)
        
        # Modyfikacja jest wykonywana w ramach konstruktora i przez resetowanie w metodzie optimize
        # klasy bazowej, która wywołuje inicjalizację CMAEvolutionStrategy, a następnie
        # initialize_random_ps() jest wywoływana przez konstruktor.
        
        return result
