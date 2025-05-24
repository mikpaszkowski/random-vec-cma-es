#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moduł odpowiedzialny za zbieranie i zapisywanie danych eksperymentalnych.
"""

import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from experiments.config import RESULTS_DIR_STRUCTURE


class DataCollector:
    """
    Klasa odpowiedzialna za zbieranie i zapisywanie danych eksperymentalnych.
    """
    
    def __init__(self, base_dir=None):
        """
        Inicjalizacja kolektora danych.
        
        Args:
            base_dir: Katalog bazowy dla wyników. Jeśli None, zostanie utworzony katalog z bieżącą datą i czasem.
        """
        if base_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.base_dir = os.path.join('results', timestamp)
        else:
            self.base_dir = base_dir
        
        # Utworzenie struktury katalogów
        for dir_name in RESULTS_DIR_STRUCTURE.values():
            os.makedirs(os.path.join(self.base_dir, dir_name), exist_ok=True)
        
        # Inicjalizacja rejestrów danych
        self.experiment_results = []
    
    def save_experiment_result(self, experiment_data):
        """
        Zapisuje wyniki pojedynczego eksperymentu.
        
        Args:
            experiment_data: Słownik zawierający dane eksperymentu.
        """
        # Ekstrakcja metadanych
        function = experiment_data['function']
        dimension = experiment_data['dimension']
        algorithm = experiment_data['algorithm']
        generator = experiment_data['generator']
        seed = experiment_data['seed']
        
        # Utworzenie nazwy pliku
        filename = f"{function}_{dimension}D_{algorithm}_{generator}_seed{seed}.json"
        filepath = os.path.join(self.base_dir, RESULTS_DIR_STRUCTURE['raw_data'], filename)
        
        # Dodaj do rejestru
        self.experiment_results.append(experiment_data)
        
        # Zapisanie danych w formacie JSON
        with open(filepath, 'w') as f:
            # Konwersja danych NumPy do formatu JSON
            json_compatible_data = self._convert_to_json_compatible(experiment_data)
            json.dump(json_compatible_data, f, indent=2)
        
        # Zapisanie krzywej zbieżności jako CSV
        if 'convergence_data' in experiment_data:
            csv_filename = f"{function}_{dimension}D_{algorithm}_{generator}_seed{seed}_convergence.csv"
            csv_filepath = os.path.join(self.base_dir, RESULTS_DIR_STRUCTURE['raw_data'], csv_filename)
            
            with open(csv_filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['evaluations', 'best_fitness', 'sigma'])
                writer.writeheader()
                for data_point in experiment_data['convergence_data']:
                    writer.writerow(data_point)
        
        # Zapisanie informacji o ziarnie
        seed_filepath = os.path.join(self.base_dir, RESULTS_DIR_STRUCTURE['seeds'], f"{generator}_seeds.txt")
        with open(seed_filepath, 'a') as f:
            f.write(f"{function}_{dimension}D_{algorithm}: {seed}\n")
    
    def save_summary(self, results=None):
        """
        Zapisuje zbiorcze wyniki wielu eksperymentów.
        
        Args:
            results: Lista wyników eksperymentów. Jeśli None, używa wewnętrznego rejestru.
        """
        if results is None:
            results = self.experiment_results
        
        if not results:
            print("Brak wyników do zapisania.")
            return
        
        # Grupowanie wyników
        summary = {}
        for result in results:
            key = (result['function'], result['dimension'], result['algorithm'], result['generator'])
            if key not in summary:
                summary[key] = []
            summary[key].append(result)
        
        # Zapisywanie zagregowanych danych dla każdej konfiguracji
        for key, group_results in summary.items():
            function, dimension, algorithm, generator = key
            
            # Obliczanie statystyk
            best_fitnesses = [r['result']['fun'] for r in group_results]
            evaluations = [r['result']['nfev'] for r in group_results]
            success_rate = sum(1 for r in group_results if r.get('success', False)) / len(group_results)
            
            # Tworzenie zagregowanych danych
            aggregated_data = {
                'function': function,
                'dimension': dimension,
                'algorithm': algorithm,
                'generator': generator,
                'repetitions': len(group_results),
                'best_fitness_mean': float(np.mean(best_fitnesses)),
                'best_fitness_std': float(np.std(best_fitnesses)),
                'best_fitness_min': float(np.min(best_fitnesses)),
                'best_fitness_max': float(np.max(best_fitnesses)),
                'best_fitness_median': float(np.median(best_fitnesses)),
                'evaluations_mean': float(np.mean(evaluations)),
                'evaluations_std': float(np.std(evaluations)),
                'success_rate': float(success_rate)
            }
            
            # Zapisanie zagregowanych danych
            filename = f"{function}_{dimension}D_{algorithm}_{generator}_summary.json"
            filepath = os.path.join(self.base_dir, RESULTS_DIR_STRUCTURE['statistics'], filename)
            
            with open(filepath, 'w') as f:
                json.dump(aggregated_data, f, indent=2)
        
        # Zapisanie wszystkich zagregowanych danych w jednym pliku CSV
        all_summaries = []
        for key, group_results in summary.items():
            function, dimension, algorithm, generator = key
            
            # Obliczanie statystyk
            best_fitnesses = [r['result']['fun'] for r in group_results]
            evaluations = [r['result']['nfev'] for r in group_results]
            success_rate = sum(1 for r in group_results if r.get('success', False)) / len(group_results)
            
            all_summaries.append({
                'function': function,
                'dimension': dimension,
                'algorithm': algorithm,
                'generator': generator,
                'repetitions': len(group_results),
                'best_fitness_mean': float(np.mean(best_fitnesses)),
                'best_fitness_std': float(np.std(best_fitnesses)),
                'best_fitness_min': float(np.min(best_fitnesses)),
                'best_fitness_max': float(np.max(best_fitnesses)),
                'best_fitness_median': float(np.median(best_fitnesses)),
                'evaluations_mean': float(np.mean(evaluations)),
                'evaluations_std': float(np.std(evaluations)),
                'success_rate': float(success_rate)
            })
        
        # Zapisz do CSV
        csv_filepath = os.path.join(self.base_dir, 'all_results_summary.csv')
        with open(csv_filepath, 'w', newline='') as f:
            if all_summaries:
                writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
                writer.writeheader()
                for summary_row in all_summaries:
                    writer.writerow(summary_row)
    
    def _convert_to_json_compatible(self, data):
        """
        Konwertuje typy danych niekompatybilne z JSON na formaty kompatybilne.
        
        Args:
            data: Dane do konwersji.
            
        Returns:
            Dane w formacie kompatybilnym z JSON.
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, dict):
            return {k: self._convert_to_json_compatible(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_json_compatible(item) for item in data]
        else:
            return data 