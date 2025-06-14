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
                convergence_interval: int = 100,
                ftarget_stop: Optional[float] = None) -> Dict[str, Any]:
        """
        Przeprowadza pełną optymalizację.
        
        Args:
            max_evaluations: Maksymalna liczba ewaluacji funkcji celu.
            ftol: Tolerancja zbieżności dla wartości funkcji.
            xtol: Tolerancja zbieżności dla parametrów.
            verbose: Czy wyświetlać postęp optymalizacji.
            convergence_interval: Odstęp między zapisywaniem danych do krzywej zbieżności.
            ftarget_stop: Wartość funkcji celu, która powoduje zatrzymanie optymalizacji.
        
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
        
        # Ustawienia verbosity - kontroluje komunikaty konsolowe
        if verbose:
            options['verbose'] = 1
        else:
            options['verbose'] = -9  # Wyłącz komunikaty konsolowe

        # KLUCZOWE: Zawsze włącz logowanie plików dla wykresów
        # Te opcje są niezależne od 'verbose' i kontrolują zapisywanie danych do plików
        # WAŻNE: Ustawiamy te opcje PO ustawieniu verbose, aby nie zostały nadpisane
        options['verb_log'] = 1  # Poziom logowania do plików (1 = każda iteracja)
        options['verb_filenameprefix'] = 'outcmaes/'  # Katalog i prefix plików
        options['verb_disp'] = 1  # Wyświetlaj informacje co iterację (może pomóc z logowaniem)
        options['verb_time'] = True  # Dodatkowe informacje o czasie
        
        if ftarget_stop is not None:
            options['ftarget'] = ftarget_stop
        
        # Inicjalizacja obiektu CMAEvolutionStrategy od nowa z nowymi opcjami
        self.es = cma.CMAEvolutionStrategy(
            self.initial_mean, 
            self.initial_sigma, 
            inopts=options
        )
        
        # WYMUSZENIE: Upewnij się że verb_log jest ustawione poprawnie po inicjalizacji
        self.es.opts['verb_log'] = 1
        self.es.opts['verb_filenameprefix'] = 'outcmaes/'
        
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
            
            # KLUCZOWE: Zapisz dane do plików w każdej iteracji
            if hasattr(self.es, 'logger') and self.es.logger:
                try:
                    self.es.logger.add()  # To jest klucz do zapisywania danych co iterację!
                except Exception as log_error:
                    if verbose:
                        print(f"Ostrzeżenie: Problem z logowaniem w iteracji {self.iterations}: {log_error}")
            
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
            
            if ftarget_stop is not None and self.best_fitness <= ftarget_stop:
                if verbose:
                    print(f"Zatrzymano: osiągnięto docelową wartość funkcji {ftarget_stop}")
                break
        
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
        
        # KLUCZOWE: Zapisz dane logowania do plików (wywoła to również tworzenie katalogu outcmaes)
        try:
            if hasattr(self.es, 'logger') and self.es.logger:
                # Zapisz dane do plików - to tworzy katalog outcmaes i wszystkie pliki danych
                self.es.logger.save()
                # Możemy też dodać informacje o zakończeniu
                if hasattr(self.es.logger, 'add'):
                    self.es.logger.add()
                    
                if verbose:
                    import os
                    if os.path.exists('outcmaes'):
                        print(f"Katalog outcmaes został utworzony z plikami: {os.listdir('outcmaes')}")
                    else:
                        print("Ostrzeżenie: Katalog outcmaes nie został utworzony pomimo wywołania logger.save()")
        except Exception as e:
            # ZAWSZE pokaż błędy logowania, niezależnie od verbose
            print(f"Ostrzeżenie: Nie udało się zapisać danych logowania: {e}")
            import traceback
            print(f"Szczegóły błędu: {traceback.format_exc()}")
        
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
        # POPRAWKA: ps znajduje się w es.adapt_sigma.ps, nie w es.ps
        if hasattr(self.es, 'adapt_sigma') and hasattr(self.es.adapt_sigma, 'ps'):
            return self.es.adapt_sigma.ps.copy()
        else:
            # Jeśli wektor ps nie istnieje (przed pierwszą iteracją), zwróć wektor zerowy
            return np.zeros(self.dimension)

    def plot(self, **kwargs) -> None:
        """
        Generuje standardowe wykresy CMA-ES przy użyciu metody plot() z obiektu CMAEvolutionStrategy.
        Wymaga, aby optymalizacja została wcześniej uruchomiona.
        Dane do wykresów są zazwyczaj wczytywane z katalogu 'outcmaes/'.
        
        Args:
            **kwargs: Dodatkowe argumenty przekazywane do metody `self.es.plot()`.
                      Pozwala na customizację wykresów zgodnie z opcjami pycma.
                      Na przykład: `fig=None`, `iabscissa=0`, `iteridx=-1`, `x_opt=None`, `f_opt=None`, `ax=None` etc.
        
        Raises:
            AttributeError: Jeśli metoda `optimize` nie została jeszcze wywołana lub obiekt `self.es` nie istnieje.
            Exception: Jeśli `pycma` napotka problem podczas generowania wykresu (np. brak danych).
        """
        if not hasattr(self, 'es') or self.es is None:
            raise AttributeError("Obiekt CMAEvolutionStrategy 'es' nie istnieje. "
                                 "Uruchom najpierw metodę 'optimize'.")
        
        print("Generowanie wykresów CMA-ES. Dane są zazwyczaj wczytywane z katalogu 'outcmaes/'...")
        try:
            self.es.result_pretty()
            self.es.plot(**kwargs)
        except Exception as e:
            print(f"Wystąpił błąd podczas generowania wykresów: {e}")
            print("Upewnij się, że optymalizacja została przeprowadzona i dane zostały zapisane (zwykle w 'outcmaes/').")
