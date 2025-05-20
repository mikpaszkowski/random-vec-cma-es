#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementacja generatora liczb pseudolosowych PCG (Permuted Congruential Generator).
Wykorzystuje implementację z biblioteki pcg-numpy.
"""

import numpy as np
import warnings

try:
    import pcg
    PCG_AVAILABLE = True
except ImportError:
    PCG_AVAILABLE = False
    warnings.warn("Biblioteka pcg-numpy nie jest dostępna. Generator PCG będzie emulowany przez Mersenne Twister.", 
                  ImportWarning)


class PCG:
    """
    Wrapper dla generatora PCG (Permuted Congruential Generator).
    Jeśli biblioteka pcg-numpy nie jest dostępna, używa generatora Mersenne Twister jako fallback.
    """
    
    def __init__(self, seed=None):
        """
        Inicjalizacja generatora.
        
        Args:
            seed: Ziarno inicjalizacyjne dla generatora. Jeśli None, zostanie użyte losowe ziarno.
        """
        self.seed = seed
        
        if PCG_AVAILABLE:
            self.rng = pcg.PCG64(seed)
        else:
            # Fallback do Mersenne Twister jeśli biblioteka pcg nie jest dostępna
            self.rng = np.random.RandomState(seed)
            print("Uwaga: Używam Mersenne Twister jako fallback zamiast PCG!")
        
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
        if PCG_AVAILABLE:
            # W przypadku PCG64, wygeneruj liczbę z przedziału [0,1) a następnie przekształć
            # do rozkładu normalnego używając odwrotnej funkcji dystrybuanty
            return self.rng.standard_normal(args if args else None)
        else:
            return self.rng.randn(*args)
    
    def rand(self, *args):
        """
        Generuje liczby pseudolosowe z rozkładu jednostajnego [0, 1).
        
        Args:
            *args: Wymiary tablicy wynikowej.
            
        Returns:
            Tablica liczb pseudolosowych z rozkładu jednostajnego [0, 1).
        """
        if PCG_AVAILABLE:
            return self.rng.random(args if args else None)
        else:
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
        if PCG_AVAILABLE:
            return self.rng.integers(low, high, size=size)
        else:
            return self.rng.randint(low, high, size)
    
    def random_sample(self, size=None):
        """
        Generuje liczby pseudolosowe z rozkładu jednostajnego [0, 1).
        
        Args:
            size: Wymiary tablicy wynikowej.
            
        Returns:
            Tablica liczb pseudolosowych.
        """
        if PCG_AVAILABLE:
            return self.rng.random(size)
        else:
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
        if PCG_AVAILABLE:
            return self.rng.choice(a, size=size, replace=replace, p=p)
        else:
            return self.rng.choice(a, size, replace, p)
    
    def shuffle(self, x):
        """
        Tasuje elementy w tablicy x.
        
        Args:
            x: Tablica do tasowania.
        """
        if PCG_AVAILABLE:
            return self.rng.shuffle(x)
        else:
            return self.rng.shuffle(x)
    
    def permutation(self, x):
        """
        Zwraca permutację sekwencji x.
        
        Args:
            x: Sekwencja do permutowania lub liczba całkowita.
            
        Returns:
            Permutacja sekwencji x lub permutacja liczb od 0 do x-1.
        """
        if PCG_AVAILABLE:
            return self.rng.permutation(x)
        else:
            return self.rng.permutation(x)
    
    def random(self):
        """
        Generuje liczbę pseudolosową z rozkładu jednostajnego [0, 1).
        
        Returns:
            Liczba pseudolosowa.
        """
        if PCG_AVAILABLE:
            return self.rng.random()
        else:
            return self.rng.random()
