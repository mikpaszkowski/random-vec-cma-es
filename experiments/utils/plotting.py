#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Narzędzia do wizualizacji wyników eksperymentów.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from experiments.config import RESULTS_DIR_STRUCTURE


def generate_convergence_plot(results_dir, function, dimension, algorithm, generator, seeds):
    """
    Generuje wykres krzywych zbieżności dla danej konfiguracji.
    
    Args:
        results_dir: Katalog z wynikami eksperymentów.
        function: Nazwa funkcji testowej.
        dimension: Wymiar przestrzeni.
        algorithm: Nazwa algorytmu.
        generator: Nazwa generatora liczb losowych.
        seeds: Lista ziaren dla których mają być generowane wykresy.
    """
    plt.figure(figsize=(10, 6))
    
    convergence_data_all = []
    legend_labels = []
    
    for seed in seeds:
        # Wczytaj dane eksperymentu
        json_filename = f"{function}_{dimension}D_{algorithm}_{generator}_seed{seed}.json"
        json_filepath = os.path.join(results_dir, RESULTS_DIR_STRUCTURE['raw_data'], json_filename)
        
        if os.path.exists(json_filepath):
            try:
                with open(json_filepath, 'r') as f:
                    data = json.load(f)
                    
                if 'convergence_data' in data and data['convergence_data']:
                    # Pobierz dane zbieżności
                    conv_data = data['convergence_data']
                    
                    # Wyodrębnij dane
                    evaluations = [point['evaluations'] for point in conv_data]
                    best_fitness = [point['best_fitness'] for point in conv_data]
                    
                    # FILTRUJ wartości inf i nan - usuń je z danych
                    valid_indices = []
                    for i, fitness in enumerate(best_fitness):
                        if np.isfinite(fitness) and fitness > 0:  # Tylko skończone i dodatnie wartości dla semilogy
                            valid_indices.append(i)
                    
                    if len(valid_indices) < 2:
                        print(f"Zbyt mało prawidłowych punktów dla ziarna {seed} (tylko {len(valid_indices)})")
                        continue
                    
                    # Filtruj dane
                    evaluations = [evaluations[i] for i in valid_indices]
                    best_fitness = [best_fitness[i] for i in valid_indices]
                    
                    # Upewnij się, że dane są posortowane według liczby ewaluacji
                    sorted_indices = np.argsort(evaluations)
                    evaluations = [evaluations[i] for i in sorted_indices]
                    best_fitness = [best_fitness[i] for i in sorted_indices]
                    
                    # Sprawdź czy mamy wystarczająco danych do rysowania
                    if len(evaluations) >= 2:
                        # Rysuj krzywą zbieżności dla tego ziarna
                        plt.semilogy(evaluations, best_fitness, alpha=0.3, linewidth=0.5)
                        
                        legend_labels.append(f"Ziarno {seed}")
                        convergence_data_all.append({
                            'evaluations': evaluations,
                            'best_fitness': best_fitness,
                            'seed': seed
                        })
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Błąd podczas odczytu pliku {json_filepath}: {str(e)}")
                continue
    
    # Oblicz średnią krzywą zbieżności, jeśli mamy dane
    if convergence_data_all:
        # Znajdź maksymalną liczbę ewaluacji
        max_evals = max([max(data['evaluations']) for data in convergence_data_all])
        
        # Zdefiniuj wspólną siatkę punktów ewaluacji dla interpolacji
        all_evaluations = []
        for data in convergence_data_all:
            all_evaluations.extend(data['evaluations'])
        common_evaluations = np.unique(all_evaluations)
        common_evaluations.sort()
        
        # Interpoluj dane dla każdego ziarna na wspólną siatkę
        interpolated_data = []
        for data in convergence_data_all:
            # Sprawdź czy mamy wystarczająco punktów do interpolacji
            if len(data['evaluations']) < 2:
                print(f"Zbyt mało punktów do interpolacji dla ziarna {data['seed']}")
                continue
                
            try:
                # Import funkcji do interpolacji
                from scipy.interpolate import interp1d
                
                # Upewnij się, że mamy unikalne ewaluacje (wymagane przez interp1d)
                eval_unique, indices = np.unique(data['evaluations'], return_index=True)
                fitness_unique = [data['best_fitness'][i] for i in indices]
                
                # Sprawdź czy nadal mamy wystarczająco punktów
                if len(eval_unique) < 2:
                    print(f"Zbyt mało unikalnych punktów do interpolacji dla ziarna {data['seed']}")
                    continue
                
                # Stwórz funkcję interpolacji
                f = interp1d(eval_unique, fitness_unique, kind='linear', bounds_error=False, 
                           fill_value=(fitness_unique[0], fitness_unique[-1]))
                interpolated_fitness = f(common_evaluations)
                
                # Sprawdź czy interpolacja dała prawidłowe wyniki
                if np.any(np.isfinite(interpolated_fitness)):
                    interpolated_data.append(interpolated_fitness)
                
            except Exception as e:
                print(f"Błąd podczas interpolacji dla ziarna {data['seed']}: {str(e)}")
                continue
        
        if interpolated_data and len(interpolated_data) > 0:
            # Oblicz średnią i odchylenie standardowe
            interpolated_data = np.array(interpolated_data)
            mean_fitness = np.nanmean(interpolated_data, axis=0)
            std_fitness = np.nanstd(interpolated_data, axis=0)
            
            # Filtruj punkty ze średniej (usuń inf i nan)
            valid_mean_indices = np.isfinite(mean_fitness) & (mean_fitness > 0)
            if np.any(valid_mean_indices):
                # Narysuj średnią krzywą zbieżności z przedziałem ufności
                valid_evals = common_evaluations[valid_mean_indices]
                valid_mean = mean_fitness[valid_mean_indices]
                valid_std = std_fitness[valid_mean_indices]
                
                plt.semilogy(valid_evals, valid_mean, 'r-', linewidth=2, label='Średnia')
                plt.fill_between(valid_evals, 
                               np.maximum(valid_mean - valid_std, np.full_like(valid_mean, 1e-10)),
                               valid_mean + valid_std,
                               color='r', alpha=0.2, label='Przedział ufności')
    
    plt.title(f"Krzywa zbieżności: {function} {dimension}D, {algorithm}, {generator}")
    plt.xlabel("Liczba ewaluacji")
    plt.ylabel("Wartość funkcji celu (log)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Ustaw limity osi Y aby uniknąć problemów z wartościami blisko zera
    plt.ylim(bottom=1e-10)
    
    # Dodaj legendę tylko jeśli są dane
    if legend_labels:
        plt.legend()
    
    # Dostosuj limity osi X
    if convergence_data_all:
        max_eval = max([max(data['evaluations']) for data in convergence_data_all])
        plt.xlim(0, max_eval)
    else:
        plt.xlim(0, 1)
        # Dodaj tekst informujący o braku danych
        plt.text(0.5, 0.5, 'Brak danych do wyświetlenia', 
                transform=plt.gca().transAxes, ha='center', va='center',
                fontsize=14, color='red')
    
    # Zapisz wykres
    os.makedirs(os.path.join(results_dir, RESULTS_DIR_STRUCTURE['convergence_plots']), exist_ok=True)
    plot_filename = f"{function}_{dimension}D_{algorithm}_{generator}_convergence.png"
    plot_filepath = os.path.join(results_dir, RESULTS_DIR_STRUCTURE['convergence_plots'], plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_plots(results_dir):
    """
    Generuje wykresy porównawcze dla różnych konfiguracji.
    
    Args:
        results_dir: Katalog z wynikami eksperymentów.
    """
    # Wczytaj podsumowanie wyników
    summary_filepath = os.path.join(results_dir, 'all_results_summary.csv')
    if not os.path.exists(summary_filepath):
        print(f"Brak pliku z podsumowaniem: {summary_filepath}")
        return
    
    summary_data = pd.read_csv(summary_filepath)
    
    # Upewnij się, że katalogi na wykresy istnieją
    os.makedirs(os.path.join(results_dir, RESULTS_DIR_STRUCTURE['comparison_plots']), exist_ok=True)
    
    # Generuj różne porównania
    # 1. Porównanie algorytmów dla każdej funkcji i wymiaru
    for function in summary_data['function'].unique():
        for dimension in summary_data['dimension'].unique():
            subset = summary_data[(summary_data['function'] == function) & 
                                 (summary_data['dimension'] == dimension)]
            
            if len(subset) > 0:
                plt.figure(figsize=(12, 6))
                
                # Porównanie średnich wartości najlepszych znalezionych rozwiązań
                plt.subplot(1, 2, 1)
                sns.barplot(x='algorithm', y='best_fitness_mean', hue='generator', data=subset)
                plt.title(f"{function} {dimension}D - Średnia wartość funkcji")
                plt.ylabel("Średnia wartość funkcji")
                plt.xlabel("Algorytm")
                
                # Porównanie średniej liczby ewaluacji
                plt.subplot(1, 2, 2)
                sns.barplot(x='algorithm', y='evaluations_mean', hue='generator', data=subset)
                plt.title(f"{function} {dimension}D - Średnia liczba ewaluacji")
                plt.ylabel("Średnia liczba ewaluacji")
                plt.xlabel("Algorytm")
                
                plt.tight_layout()
                
                # Zapisz wykres
                plot_filename = f"{function}_{dimension}D_comparison.png"
                plot_filepath = os.path.join(results_dir, RESULTS_DIR_STRUCTURE['comparison_plots'], plot_filename)
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                plt.close()
    
    # 2. Porównanie wskaźnika sukcesu
    for function in summary_data['function'].unique():
        plt.figure(figsize=(12, 8))
        
        dimensions = sorted(summary_data['dimension'].unique())
        
        for i, dimension in enumerate(dimensions):
            plt.subplot(len(dimensions), 1, i+1)
            
            subset = summary_data[(summary_data['function'] == function) & 
                                 (summary_data['dimension'] == dimension)]
            
            if len(subset) > 0:
                sns.barplot(x='algorithm', y='success_rate', hue='generator', data=subset)
                plt.title(f"{function} {dimension}D - Wskaźnik sukcesu")
                plt.ylabel("Wskaźnik sukcesu")
                plt.xlabel("Algorytm" if i == len(dimensions)-1 else "")
                plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Zapisz wykres
        plot_filename = f"{function}_success_rate_comparison.png"
        plot_filepath = os.path.join(results_dir, RESULTS_DIR_STRUCTURE['comparison_plots'], plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()


def calculate_wilcoxon_tests(results_dir):
    """
    Przeprowadza testy Wilcoxona dla par skorelowanych w celu określenia istotności różnic
    między algorytmami standardowym i zmodyfikowanym.
    
    Args:
        results_dir: Katalog z wynikami eksperymentów.
        
    Returns:
        DataFrame z wynikami testów.
    """
    from scipy.stats import wilcoxon
    
    # Wczytaj dane z wszystkich eksperymentów
    raw_data_dir = os.path.join(results_dir, RESULTS_DIR_STRUCTURE['raw_data'])
    if not os.path.exists(raw_data_dir):
        print(f"Brak katalogu z surowymi danymi: {raw_data_dir}")
        return None
    
    # Przeszukaj pliki JSON
    results_standard = {}
    results_modified = {}
    
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.json') and not filename.endswith('_convergence.json'):
            filepath = os.path.join(raw_data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Ekstrakcja kluczowych informacji
                    function = data['function']
                    dimension = data['dimension']
                    algorithm = data['algorithm']
                    generator = data['generator']
                    seed = data['seed']
                    best_fitness = data['result']['fun']
                    
                    key = (function, dimension, generator, seed)
                    
                    if algorithm == 'standard':
                        results_standard[key] = best_fitness
                    elif algorithm == 'modified':
                        results_modified[key] = best_fitness
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Błąd podczas odczytu pliku {filepath}: {str(e)}")
                continue
    
    # Przygotuj dane do testu Wilcoxona
    wilcoxon_results = []
    
    for function in set(key[0] for key in results_standard.keys()):
        for dimension in set(key[1] for key in results_standard.keys() if key[0] == function):
            for generator in set(key[2] for key in results_standard.keys() if key[0] == function and key[1] == dimension):
                
                standard_values = []
                modified_values = []
                
                for seed in set(key[3] for key in results_standard.keys() if key[0] == function and key[1] == dimension and key[2] == generator):
                    key = (function, dimension, generator, seed)
                    if key in results_standard and key in results_modified:
                        standard_values.append(results_standard[key])
                        modified_values.append(results_modified[key])
                
                if len(standard_values) > 0 and len(modified_values) > 0:
                    # Sprawdź czy wartości są identyczne (test Wilcoxona nie zadziała)
                    differences = np.array(standard_values) - np.array(modified_values)
                    
                    if np.all(differences == 0):
                        # Wszystkie różnice są zerowe - brak różnicy między metodami
                        wilcoxon_results.append({
                            'function': function,
                            'dimension': dimension,
                            'generator': generator,
                            'p_value': 1.0,  # Brak różnicy = p=1
                            'significant': False,
                            'better_method': "brak różnicy (identyczne wyniki)",
                            'standard_mean': np.mean(standard_values),
                            'modified_mean': np.mean(modified_values),
                            'sample_size': len(standard_values)
                        })
                    else:
                        # Przeprowadź test Wilcoxona
                        try:
                            stat, p_value = wilcoxon(standard_values, modified_values)
                            
                            # Określ która metoda jest lepsza
                            if np.mean(standard_values) < np.mean(modified_values):
                                better_method = "standard"
                            else:
                                better_method = "modified"
                            
                            wilcoxon_results.append({
                                'function': function,
                                'dimension': dimension,
                                'generator': generator,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'better_method': better_method if p_value < 0.05 else "brak istotnej różnicy",
                                'standard_mean': np.mean(standard_values),
                                'modified_mean': np.mean(modified_values),
                                'sample_size': len(standard_values)
                            })
                        except Exception as e:
                            print(f"Błąd w teście Wilcoxona dla {function} {dimension}D {generator}: {str(e)}")
                            # Dodaj wpis z błędem
                            wilcoxon_results.append({
                                'function': function,
                                'dimension': dimension,
                                'generator': generator,
                                'p_value': np.nan,
                                'significant': False,
                                'better_method': f"błąd testu: {str(e)}",
                                'standard_mean': np.mean(standard_values),
                                'modified_mean': np.mean(modified_values),
                                'sample_size': len(standard_values)
                            })
    
    # Utwórz DataFrame z wynikami
    results_df = pd.DataFrame(wilcoxon_results)
    
    # Zapisz wyniki
    os.makedirs(os.path.join(results_dir, RESULTS_DIR_STRUCTURE['statistics']), exist_ok=True)
    csv_filepath = os.path.join(results_dir, RESULTS_DIR_STRUCTURE['statistics'], 'wilcoxon_tests.csv')
    results_df.to_csv(csv_filepath, index=False)
    
    return results_df 