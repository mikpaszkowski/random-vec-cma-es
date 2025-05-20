#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementacja generatora liczb pseudolosowych Mersenne Twister.
Wykorzystuje implementację z biblioteki NumPy.
"""

import numpy as np


class MersenneTwister:
    """
    Wrapper dla generatora Mersenne Twister z biblioteki NumPy.
    """
    
    def __init__(self, seed=None):
        """
        Inicjalizacja generatora.
        
        Args:
            seed: Ziarno inicjalizacyjne dla generatora. Jeśli None, zostanie użyte losowe ziarno.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def get_seed(self):
        """
        Zwraca aktualne ziarno generatora.
        
        Returns:
            Ziarno generatora.
        """
        return self.seed
    
    def randn(self, *args):
        """
        Generuje liczby pseudolosowe z rozkładu normalnego.
        
        Args:
            *args: Wymiary tablicy wynikowej.
            
        Returns:
            Tablica liczb pseudolosowych z rozkładu normalnego N(0, 1).
        """
        return self.rng.randn(*args)
    
    def rand(self, *args):
        """
        Generuje liczby pseudolosowe z rozkładu jednostajnego [0, 1).
        
        Args:
            *args: Wymiary tablicy wynikowej.
            
        Returns:
            Tablica liczb pseudolosowych z rozkładu jednostajnego [0, 1).
        """
        return self.rng.rand(*args)
    
    def randint(self, low, high=None, size=None):
        """
        Generuje liczby całkowite pseudolosowe z przedziału [low, high).
        
        Args:
            low: Dolna granica przedziału (włącznie). Jeśli high=None, to [0, low).
            high: Górna granica przedziału (wyłącznie).
            size: Wymiary tablicy wynikowej.
            
        Returns:
            Tablica liczb całkowitych pseudolosowych.
        """
        return self.rng.randint(low, high, size)
    
    def random_sample(self, size=None):
        """
        Generuje liczby pseudolosowe z rozkładu jednostajnego [0, 1).
        
        Args:
            size: Wymiary tablicy wynikowej.
            
        Returns:
            Tablica liczb pseudolosowych.
        """
        return self.rng.random_sample(size)
    
    def choice(self, a, size=None, replace=True, p=None):
        """
        Losuje elementy z tablicy a.
        
        Args:
            a: Tablica, z której mają być losowane elementy.
            size: Wymiary wynikowej tablicy.
            replace: Czy losowanie z powtórzeniami.
            p: Prawdopodobieństwa poszczególnych elementów.
            
        Returns:
            Tablica wylosowanych elementów.
        """
        return self.rng.choice(a, size, replace, p)
    
    def shuffle(self, x):
        """
        Tasuje elementy w tablicy x.
        
        Args:
            x: Tablica do tasowania.
        """
        return self.rng.shuffle(x)
    
    def permutation(self, x):
        """
        Zwraca permutację sekwencji x.
        
        Args:
            x: Sekwencja do permutowania lub liczba całkowita.
            
        Returns:
            Permutacja sekwencji x lub permutacja liczb od 0 do x-1.
        """
        return self.rng.permutation(x)
    
    def random(self):
        """
        Generuje liczbę pseudolosową z rozkładu jednostajnego [0, 1).
        
        Returns:
            Liczba pseudolosowa.
        """
        return self.rng.random()
