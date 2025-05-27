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
        # Generuj losowy wektor z rozkładu normalnego
        if hasattr(self.random_generator, 'randn'):
            # Użyj naszego generatora
            random_ps = self.random_generator.randn(self.dimension)
        else:
            # Użyj standardowego generatora numpy
            random_ps = np.random.randn(self.dimension)
        
        # POPRAWKA: ps znajduje się w es.adapt_sigma.ps, nie w es.ps
        # ps jest inicjalizowane dopiero po pierwszym tell(), więc zawsze zapisz do _pending_random_ps
        self._pending_random_ps = random_ps.copy()
    
    def ask(self) -> np.ndarray:
        """
        Generuje nową populację osobników z uwzględnieniem losowej inicjalizacji ps.
        
        Returns:
            Macierz punktów (wiersze to osobniki, kolumny to wymiary).
        """
        # Wywołaj standardową metodę ask
        solutions = super().ask()
        
        # Nie rób nic tutaj - ps nie jest jeszcze zainicjalizowane po ask()
        return solutions
    
    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """
        Aktualizuje stan algorytmu na podstawie wyników oceny populacji.
        POPRAWKA: Ustawia losowe ps po pierwszym wywołaniu tell().
        
        Args:
            solutions: Macierz punktów (wiersze to osobniki, kolumny to wymiary).
            fitnesses: Wektor wartości funkcji celu dla każdego osobnika.
        """
        # Wywołaj standardową metodę tell - to zainicjalizuje ps
        super().tell(solutions, fitnesses)
        
        # POPRAWKA: Po tell() sprawdź czy ps zostało zainicjalizowane i zastąp je losowym
        if (hasattr(self, '_pending_random_ps') and 
            hasattr(self.es, 'adapt_sigma') and 
            hasattr(self.es.adapt_sigma, 'ps')):
            
            # Zastąp standardowe ps (wektor zer) naszym losowym ps
            self.es.adapt_sigma.ps = self._pending_random_ps.copy()
            delattr(self, '_pending_random_ps')  # Usuń oczekujący ps
    
    def optimize(self, 
                max_evaluations: int = 1000, 
                ftol: float = 1e-8, 
                xtol: float = 1e-8, 
                verbose: bool = False,
                convergence_interval: int = 100) -> Dict[str, Any]:
        """
        Przeprowadza pełną optymalizację z losową inicjalizacją ścieżki ewolucyjnej pσ.
        
        Args:
            max_evaluations: Maksymalna liczba ewaluacji funkcji celu.
            ftol: Tolerancja zbieżności dla wartości funkcji.
            xtol: Tolerancja zbieżności dla parametrów.
            verbose: Czy wyświetlać postęp optymalizacji.
            convergence_interval: Odstęp między zapisywaniem danych do krzywej zbieżności.
        
        Returns:
            Słownik z wynikami optymalizacji.
        """
        # Resetuj statystyki (ale NIE obiekt es, żeby zachować losowe ps)
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
        
        # KRYTYCZNA POPRAWKA: NIE resetuj obiektu es - to niszczy losowe ps!
        # Zamiast tego, tylko resetuj niezbędne statystyki wewnętrzne
        
        # Jeśli es nie istnieje (pierwsza optymalizacja), utwórz go
        if not hasattr(self, 'es') or self.es is None:
            self.es = cma.CMAEvolutionStrategy(
                self.initial_mean, 
                self.initial_sigma, 
                inopts=options
            )
            # Po utworzeniu es, ustaw losowe ps
            self.initialize_random_ps()
        else:
            # es już istnieje - nie resetuj go, tylko zaktualizuj opcje
            # Niestety CMA-ES nie pozwala na zmianę opcji w trakcie działania
            # Musimy stworzyć nowy obiekt, ale zachować stan ps
            current_ps = self.get_ps().copy()
            
            self.es = cma.CMAEvolutionStrategy(
                self.initial_mean, 
                self.initial_sigma, 
                inopts=options
            )
            
            # Przywróć zachowane losowe ps
            if hasattr(self.es, 'ps'):
                self.es.ps = current_ps.copy()
            else:
                self._pending_random_ps = current_ps.copy()
        
        # Przygotuj dane do śledzenia zbieżności
        convergence_data = []
        last_saved = 0
        
        # Wykonaj główną pętlę optymalizacji - użyj TYLKO kryteriów CMA-ES
        while not self.es.stop() and self.evaluations < max_evaluations:
            # Generuj populację (ask automatycznie ustawi losowe ps przy pierwszym wywołaniu)
            solutions = self.ask()
            
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
        # Najpierw sprawdź czy istnieje oczekujący losowy ps
        if hasattr(self, '_pending_random_ps'):
            return self._pending_random_ps.copy()
        
        # POPRAWKA: ps znajduje się w es.adapt_sigma.ps, nie w es.ps
        if hasattr(self, 'es') and self.es is not None and hasattr(self.es, 'adapt_sigma') and hasattr(self.es.adapt_sigma, 'ps'):
            return self.es.adapt_sigma.ps.copy()
        else:
            # Jeśli wektor ps nie istnieje, zwróć wektor zerowy
            return np.zeros(self.dimension)
