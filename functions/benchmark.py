#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zestaw funkcji testowych (benchmark) do optymalizacji.
"""

import numpy as np
from typing import Callable, Dict, Any, Tuple


class TestFunction:
    """
    Bazowa klasa dla funkcji testowych.
    """
    
    def __init__(self, dimension: int = 2):
        """
        Inicjalizacja funkcji testowej.
        
        Args:
            dimension: Wymiar przestrzeni przeszukiwań.
        """
        self.dimension = dimension
        self.evaluations_counter = 0
        self.name = "AbstractTestFunction"
        
    def __call__(self, x: np.ndarray) -> float:
        """
        Oblicza wartość funkcji testowej.
        
        Args:
            x: Punkt, w którym obliczana jest wartość funkcji.
            
        Returns:
            Wartość funkcji w punkcie x.
        """
        self.evaluations_counter += 1
        return self._evaluate(x)
    
    def _evaluate(self, x: np.ndarray) -> float:
        """
        Metoda abstrakcyjna do implementacji w klasach pochodnych.
        
        Args:
            x: Punkt, w którym obliczana jest wartość funkcji.
            
        Returns:
            Wartość funkcji w punkcie x.
        """
        raise NotImplementedError("Funkcja _evaluate musi być zaimplementowana w klasie pochodnej.")
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zwraca dolne i górne ograniczenia przestrzeni przeszukiwań.
        
        Returns:
            Krotka (lower_bounds, upper_bounds).
        """
        raise NotImplementedError("Funkcja get_bounds musi być zaimplementowana w klasie pochodnej.")
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """
        Zwraca globalne minimum funkcji.
        
        Returns:
            Krotka (global_minimum_point, global_minimum_value).
        """
        raise NotImplementedError("Funkcja get_global_minimum musi być zaimplementowana w klasie pochodnej.")
    
    def reset_counter(self) -> None:
        """
        Resetuje licznik ewaluacji funkcji.
        """
        self.evaluations_counter = 0


class Rosenbrock(TestFunction):
    """
    Funkcja Rosenbrocka: suma_{i=1}^{n-1} [100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    
    Globalne minimum: f(1, 1, ..., 1) = 0
    """
    
    def __init__(self, dimension: int = 2):
        """
        Inicjalizacja funkcji Rosenbrocka.
        
        Args:
            dimension: Wymiar przestrzeni przeszukiwań.
        """
        super().__init__(dimension)
        self.name = "Rosenbrock"
    
    def _evaluate(self, x: np.ndarray) -> float:
        """
        Oblicza wartość funkcji Rosenbrocka.
        
        Args:
            x: Punkt, w którym obliczana jest wartość funkcji.
            
        Returns:
            Wartość funkcji w punkcie x.
        """
        if len(x) != self.dimension:
            raise ValueError(f"Wymiar punktu ({len(x)}) nie zgadza się z wymiarem funkcji ({self.dimension}).")
        
        result = 0.0
        for i in range(self.dimension - 1):
            result += 100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1.0)**2
        
        return result
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zwraca dolne i górne ograniczenia przestrzeni przeszukiwań.
        
        Returns:
            Krotka (lower_bounds, upper_bounds).
        """
        lower_bounds = np.full(self.dimension, -5.0)
        upper_bounds = np.full(self.dimension, 10.0)
        return (lower_bounds, upper_bounds)
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """
        Zwraca globalne minimum funkcji.
        
        Returns:
            Krotka (global_minimum_point, global_minimum_value).
        """
        global_minimum_point = np.ones(self.dimension)
        global_minimum_value = 0.0
        return (global_minimum_point, global_minimum_value)


class Rastrigin(TestFunction):
    """
    Funkcja Rastrigina: 10*n + sum_{i=1}^n [x_i^2 - 10*cos(2*pi*x_i)]
    
    Globalne minimum: f(0, 0, ..., 0) = 0
    """
    
    def __init__(self, dimension: int = 2):
        """
        Inicjalizacja funkcji Rastrigina.
        
        Args:
            dimension: Wymiar przestrzeni przeszukiwań.
        """
        super().__init__(dimension)
        self.name = "Rastrigin"
    
    def _evaluate(self, x: np.ndarray) -> float:
        """
        Oblicza wartość funkcji Rastrigina.
        
        Args:
            x: Punkt, w którym obliczana jest wartość funkcji.
            
        Returns:
            Wartość funkcji w punkcie x.
        """
        if len(x) != self.dimension:
            raise ValueError(f"Wymiar punktu ({len(x)}) nie zgadza się z wymiarem funkcji ({self.dimension}).")
        
        result = 10.0 * self.dimension
        for i in range(self.dimension):
            result += x[i]**2 - 10.0 * np.cos(2.0 * np.pi * x[i])
        
        return result
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zwraca dolne i górne ograniczenia przestrzeni przeszukiwań.
        
        Returns:
            Krotka (lower_bounds, upper_bounds).
        """
        lower_bounds = np.full(self.dimension, -5.12)
        upper_bounds = np.full(self.dimension, 5.12)
        return (lower_bounds, upper_bounds)
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """
        Zwraca globalne minimum funkcji.
        
        Returns:
            Krotka (global_minimum_point, global_minimum_value).
        """
        global_minimum_point = np.zeros(self.dimension)
        global_minimum_value = 0.0
        return (global_minimum_point, global_minimum_value)


class Ackley(TestFunction):
    """
    Funkcja Ackleya: -20*exp(-0.2*sqrt(1/n*sum_{i=1}^n x_i^2)) - exp(1/n*sum_{i=1}^n cos(2*pi*x_i)) + 20 + e
    
    Globalne minimum: f(0, 0, ..., 0) = 0
    """
    
    def __init__(self, dimension: int = 2):
        """
        Inicjalizacja funkcji Ackleya.
        
        Args:
            dimension: Wymiar przestrzeni przeszukiwań.
        """
        super().__init__(dimension)
        self.name = "Ackley"
    
    def _evaluate(self, x: np.ndarray) -> float:
        """
        Oblicza wartość funkcji Ackleya.
        
        Args:
            x: Punkt, w którym obliczana jest wartość funkcji.
            
        Returns:
            Wartość funkcji w punkcie x.
        """
        if len(x) != self.dimension:
            raise ValueError(f"Wymiar punktu ({len(x)}) nie zgadza się z wymiarem funkcji ({self.dimension}).")
        
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x)))
        result = term1 + term2 + 20.0 + np.e
        
        return result
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zwraca dolne i górne ograniczenia przestrzeni przeszukiwań.
        
        Returns:
            Krotka (lower_bounds, upper_bounds).
        """
        lower_bounds = np.full(self.dimension, -32.768)
        upper_bounds = np.full(self.dimension, 32.768)
        return (lower_bounds, upper_bounds)
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """
        Zwraca globalne minimum funkcji.
        
        Returns:
            Krotka (global_minimum_point, global_minimum_value).
        """
        global_minimum_point = np.zeros(self.dimension)
        global_minimum_value = 0.0
        return (global_minimum_point, global_minimum_value)


class Schwefel(TestFunction):
    """
    Funkcja Schwefela: 418.9829*n - sum_{i=1}^n [x_i * sin(sqrt(|x_i|))]
    
    Globalne minimum: f(420.9687, 420.9687, ..., 420.9687) = 0
    """
    
    def __init__(self, dimension: int = 2):
        """
        Inicjalizacja funkcji Schwefela.
        
        Args:
            dimension: Wymiar przestrzeni przeszukiwań.
        """
        super().__init__(dimension)
        self.name = "Schwefel"
    
    def _evaluate(self, x: np.ndarray) -> float:
        """
        Oblicza wartość funkcji Schwefela.
        
        Args:
            x: Punkt, w którym obliczana jest wartość funkcji.
            
        Returns:
            Wartość funkcji w punkcie x.
        """
        if len(x) != self.dimension:
            raise ValueError(f"Wymiar punktu ({len(x)}) nie zgadza się z wymiarem funkcji ({self.dimension}).")
        
        result = 418.9829 * self.dimension
        for i in range(self.dimension):
            result -= x[i] * np.sin(np.sqrt(abs(x[i])))
        
        return result
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zwraca dolne i górne ograniczenia przestrzeni przeszukiwań.
        
        Returns:
            Krotka (lower_bounds, upper_bounds).
        """
        lower_bounds = np.full(self.dimension, -500.0)
        upper_bounds = np.full(self.dimension, 500.0)
        return (lower_bounds, upper_bounds)
    
    def get_global_minimum(self) -> Tuple[np.ndarray, float]:
        """
        Zwraca globalne minimum funkcji.
        
        Returns:
            Krotka (global_minimum_point, global_minimum_value).
        """
        global_minimum_point = np.full(self.dimension, 420.9687)
        global_minimum_value = 0.0
        return (global_minimum_point, global_minimum_value)


def get_function(function_name: str, dimension: int = 2) -> TestFunction:
    """
    Fabryka funkcji testowych.
    
    Args:
        function_name: Nazwa funkcji testowej.
        dimension: Wymiar przestrzeni przeszukiwań.
        
    Returns:
        Instancja żądanej funkcji testowej.
        
    Raises:
        ValueError: Jeśli podano nieznaną nazwę funkcji.
    """
    function_map = {
        'rosenbrock': Rosenbrock,
        'rastrigin': Rastrigin,
        'ackley': Ackley,
        'schwefel': Schwefel
    }
    
    function_name = function_name.lower()
    if function_name in function_map:
        return function_map[function_name](dimension)
    else:
        raise ValueError(f"Nieznana funkcja testowa: {function_name}. "
                         f"Dostępne funkcje: {', '.join(function_map.keys())}.") 