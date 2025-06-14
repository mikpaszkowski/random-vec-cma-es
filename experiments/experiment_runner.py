#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechanizm uruchamiania eksperymentów dla porównania standardowego i zmodyfikowanego algorytmu CMA-ES.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Dodanie ścieżki głównej projektu do sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import komponentów projektu
from algorithms import StandardCMAES, ModifiedCMAES
from functions import get_function
from generators import get_generator
from experiments.data_collector import DataCollector
from experiments.config import (
    DEFAULT_DIMENSIONS, DEFAULT_FUNCTIONS, DEFAULT_ALGORITHMS, DEFAULT_GENERATORS,
    DEFAULT_SEEDS, DEFAULT_MAX_EVALUATIONS, DEFAULT_FTOL, DEFAULT_XTOL,
    ACCURACY_THRESHOLDS, FUNCTION_SPECIFIC_TOLERANCES, get_initial_point,
    get_ftarget_stop_value
)


class ExperimentRunner:
    """
    Klasa odpowiedzialna za uruchamianie eksperymentów porównujących algorytmy CMA-ES.
    """
    
    def __init__(self, config_dict=None, output_dir=None, library='pycma'):
        """
        Inicjalizacja obiektu ExperimentRunner.
        
        Args:
            config_dict: Słownik z konfiguracją eksperymentu.
            output_dir: Katalog wyjściowy dla wyników. Jeśli None, zostanie utworzony katalog z bieżącą datą i czasem.
            library: Biblioteka CMA-ES do użycia ('pycma' lub 'cmaes'). Domyślnie 'pycma'.
        """
        self.config = config_dict or {}
        self.library = library
        self.data_collector = DataCollector(output_dir)
        self.output_dir = self.data_collector.base_dir
        
        # Walidacja wyboru biblioteki
        if library not in ['pycma', 'cmaes']:
            raise ValueError(f"Nieznana biblioteka: {library}. Dostępne opcje: 'pycma', 'cmaes'")
    
    def _get_algorithm_class(self, algorithm_name: str):
        """
        Zwraca odpowiednią klasę algorytmu na podstawie wybranej biblioteki.
        
        Args:
            algorithm_name: Nazwa algorytmu ('standard' lub 'modified').
            
        Returns:
            Klasa algorytmu.
        """
        if self.library == 'pycma':
            if algorithm_name == 'standard':
                return StandardCMAES
            else:  # 'modified'
                return ModifiedCMAES
        else:
            raise ValueError(f"Nieznana biblioteka: {self.library}")
    
    def run_single_experiment(self, function_name, dimension, algorithm_name, 
                             generator_name, seed, **kwargs):
        """
        Uruchamia pojedynczy eksperyment.
        
        Args:
            function_name: Nazwa funkcji testowej.
            dimension: Wymiar przestrzeni.
            algorithm_name: Nazwa algorytmu ('standard' lub 'modified').
            generator_name: Nazwa generatora liczb losowych ('mt' lub 'pcg').
            seed: Ziarno generatora.
            **kwargs: Dodatkowe parametry konfiguracyjne.
            
        Returns:
            Słownik z wynikami eksperymentu lub None w przypadku błędu.
        """
        print(f"Uruchamiam ({self.library}): {function_name} {dimension}D, {algorithm_name}, {generator_name}, seed={seed}")
        
        try:
            # Przygotowanie środowiska
            function = get_function(function_name, dimension)
            generator = get_generator(generator_name, seed)
            
            # Wybór punktu początkowego - używaj deterministycznego generatora
            initial_mean = get_initial_point(function_name, dimension, generator)
            
            # Ustalenie parametrów
            initial_sigma = kwargs.get('initial_sigma', 1.0)
            max_evaluations = kwargs.get('max_evaluations', DEFAULT_MAX_EVALUATIONS)
            
            # Pobranie wartości ftarget_stop_value
            ftarget_stop = get_ftarget_stop_value(function_name, dimension)
            
            # Użyj specyficznych tolerancji dla funkcji jeśli dostępne
            function_tolerances = FUNCTION_SPECIFIC_TOLERANCES.get(function_name, {})
            ftol = kwargs.get('ftol', function_tolerances.get('ftol', DEFAULT_FTOL))
            xtol = kwargs.get('xtol', function_tolerances.get('xtol', DEFAULT_XTOL))
            
            convergence_interval = kwargs.get('convergence_interval', 100)
            
            # Utworzenie algorytmu - używanie odpowiedniej klasy na podstawie biblioteki
            AlgorithmClass = self._get_algorithm_class(algorithm_name)
            algorithm = AlgorithmClass(
                function,
                initial_mean=initial_mean,
                initial_sigma=initial_sigma,
                random_generator=generator,
                population_size=kwargs.get('population_size', None)
            )
                
            # Zapis początkowej wartości ps - BEZPOŚREDNIO po utworzeniu algorytmu
            # bez żadnych dummy cycles, które zużywałyby stan algorytmu
            initial_ps = algorithm.get_ps().copy()
            
            # Przeprowadzenie optymalizacji - BEZPOŚREDNIO bez żadnych resetów
            start_time = time.time()
            result = algorithm.optimize(
                max_evaluations=max_evaluations,
                ftol=ftol,
                xtol=xtol,
                ftarget_stop=ftarget_stop,
                verbose=False,
                convergence_interval=convergence_interval
            )
            end_time = time.time()
            
            # Po zakończeniu optymalizacji, pobierz ostateczną wartość ps
            final_ps = algorithm.get_ps().copy()
            
            # Pobierz dane zbieżności z wyniku
            convergence_data = result.get('convergence_data', [])
            
            # Sprawdzenie czy osiągnięto zadaną dokładność
            threshold = ACCURACY_THRESHOLDS.get(function_name, {})
            print(f"Threshold: {threshold}")
            threshold = threshold.get(f"{dimension}D", 1e-6)
            print(f"Threshold: {threshold}")
            success = result['fun'] < threshold
            
            # Przygotuj dane wynikowe
            experiment_data = {
                'function': function_name,
                'dimension': dimension,
                'algorithm': algorithm_name,
                'generator': generator_name,
                'seed': seed,
                'library': self.library,  # Dodaj informację o użytej bibliotece
                'result': result,
                'success': success,
                'execution_time': end_time - start_time,
                'initial_ps': initial_ps.tolist(),
                'final_ps': final_ps.tolist(),
                'convergence_data': convergence_data
            }
            
            # Zapisanie wyników
            self.data_collector.save_experiment_result(experiment_data)
            # algorithm.plot()
            
            return experiment_data
            
        except Exception as e:
            print(f"Błąd w eksperymencie: {function_name}, {dimension}D, {algorithm_name}, {generator_name}, seed={seed}")
            print(f"Błąd: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_experiment_batch(self, **kwargs):
        """
        Uruchamia serię eksperymentów zgodnie z konfiguracją.
        
        Args:
            **kwargs: Parametry konfiguracyjne eksperymentu.
            
        Returns:
            Lista wyników eksperymentów.
        """
        # Przygotowanie parametrów
        dimensions = kwargs.get('dimensions', DEFAULT_DIMENSIONS)
        functions = kwargs.get('functions', DEFAULT_FUNCTIONS)
        algorithms = kwargs.get('algorithms', DEFAULT_ALGORITHMS)
        generators = kwargs.get('generators', DEFAULT_GENERATORS)
        seeds = kwargs.get('seeds', DEFAULT_SEEDS)
        
        # Liczenie całkowitej liczby eksperymentów
        total_experiments = len(dimensions) * len(functions) * len(algorithms) * len(generators) * len(seeds)
        print(f"Rozpoczynam serię {total_experiments} eksperymentów z biblioteką {self.library}...")
        
        # Zapisanie konfiguracji
        config_path = os.path.join(self.output_dir, 'experiment_config.json')
        with open(config_path, 'w') as f:
            import json
            config = {
                'library': self.library,  # Dodaj informację o bibliotece
                'dimensions': dimensions,
                'functions': functions,
                'algorithms': algorithms,
                'generators': generators,
                'seeds': seeds,
                'max_evaluations': kwargs.get('max_evaluations', DEFAULT_MAX_EVALUATIONS),
                'ftol': kwargs.get('ftol', DEFAULT_FTOL),
                'xtol': kwargs.get('xtol', DEFAULT_XTOL),
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            json.dump(config, f, indent=2)
        
        # Licznik postępu
        completed = 0
        
        # Iteracja po wszystkich kombinacjach
        results = []
        for dimension in dimensions:
            for function_name in functions:
                for algorithm_name in algorithms:
                    for generator_name in generators:
                        for seed in seeds:
                            # Uruchomienie eksperymentu
                            result = self.run_single_experiment(
                                function_name, dimension, algorithm_name, 
                                generator_name, seed, **kwargs
                            )
                            
                            if result:
                                results.append(result)
                            
                            # Aktualizacja postępu
                            completed += 1
                            progress = (completed / total_experiments) * 100
                            print(f"Postęp: {completed}/{total_experiments} ({progress:.2f}%)")
                            
                            # Zapisywanie postępu
                            with open(os.path.join(self.output_dir, 'progress.txt'), 'w') as f:
                                f.write(f"Ukończono {completed}/{total_experiments} ({progress:.2f}%)\n")
                                f.write(f"Biblioteka: {self.library}\n")
                                f.write(f"Ostatni eksperyment: {function_name} {dimension}D, {algorithm_name}, {generator_name}, seed={seed}\n")
        
        # Zapisanie zbiorczych statystyk
        self.data_collector.save_summary(results)
        
        # Aktualizacja konfiguracji o czas zakończenia
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return results 