#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Klasa do zarządzania wizualizacją wyników eksperymentów CMA-ES.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from experiments.config import ACCURACY_THRESHOLDS

# Ustawienia matplotlib dla polskich znaków
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ExperimentVisualizer:
    """
    Klasa odpowiedzialna za tworzenie wykresów z serii danych otrzymanych z eksperymentów.
    """
    
    def __init__(self, results_dir: str = "results/final"):
        """
        Inicjalizuje wizualizator eksperymentów.
        
        Args:
            results_dir: Ścieżka do katalogu z wynikami eksperymentów
        """
        self.results_dir = Path(results_dir)
        self.raw_data_dir = self.results_dir / "raw_data"
        self.plots_dir = self.results_dir / "plots"
        self.convergence_plots_dir = self.plots_dir / "convergence"
        self.boxplot_plots_dir = self.plots_dir / "boxplots"
        self.success_rate_plots_dir = self.plots_dir / "success_rates"
        
        # Utworz katalogi jeśli nie istnieją
        self.convergence_plots_dir.mkdir(parents=True, exist_ok=True)
        self.boxplot_plots_dir.mkdir(parents=True, exist_ok=True)
        self.success_rate_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Kolory dla różnych algorytmów i generatorów
        self.colors = {
            'standard': {'mt': '#1f77b4', 'pcg': '#aec7e8'},  # Odcienie niebieskiego
            'modified': {'mt': '#d62728', 'pcg': '#ff9896'}   # Odcienie czerwonego
        }
        
        # Progi dokładności dla różnych funkcji
        self.accuracy_thresholds = {
            'rosenbrock': 1e-6,
            'rastrigin': 1e-6,
            'ackley': 1e-6,
            'schwefel': 1e-6
        }
        
        # Wczytaj konfigurację eksperymentu jeśli istnieje
        self.config = self._load_experiment_config()
    
    def _load_experiment_config(self) -> Optional[Dict]:
        """
        Wczytuje konfigurację eksperymentu.
        
        Returns:
            Słownik z konfiguracją lub None jeśli plik nie istnieje
        """
        config_path = self.results_dir / "experiment_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_accuracy_threshold(self, function: str, dimension: int) -> float:
        return ACCURACY_THRESHOLDS.get(function, {}).get(f"{dimension}D", 1e-6)
    
    def _load_experiment_results(self, function: str, dimension: int,
                                seeds: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Wczytuje wyniki eksperymentów dla określonej funkcji i wymiaru.
        
        Args:
            function: Nazwa funkcji testowej
            dimension: Wymiar przestrzeni
            seeds: Lista ziaren do wczytania (jeśli None, wczytuje wszystkie dostępne)
            
        Returns:
            DataFrame z wynikami eksperymentów
        """
        results = []
        
        # Jeśli nie podano ziaren, użyj wszystkich z konfiguracji
        if seeds is None and self.config:
            seeds = self.config.get('seeds', list(range(1, 31)))
        elif seeds is None:
            seeds = list(range(1, 31))  # Domyślnie 30 ziaren
        
        # Wczytaj dane dla wszystkich kombinacji algorytmów i generatorów
        for algorithm in ['standard', 'modified']:
            for generator in ['mt', 'pcg']:
                for seed in seeds:
                    json_filename = f"{function}_{dimension}D_{algorithm}_{generator}_seed{seed}.json"
                    json_filepath = self.raw_data_dir / json_filename
                    
                    if json_filepath.exists():
                        try:
                            with open(json_filepath, 'r') as f:
                                data = json.load(f)
                            
                            # Ekstraktuj wyniki
                            final_fitness = data['result']['fun']
                            total_evaluations = data['result']['nfev']
                            
                            # Oblicz liczbę ewaluacji do osiągnięcia celu
                            target_threshold = self._get_accuracy_threshold(function, dimension)
                            evaluations_to_target = self._calculate_evaluations_to_target(
                                data.get('convergence_data', []), target_threshold
                            )
                            
                            results.append({
                                'function': function,
                                'dimension': dimension,
                                'algorithm': algorithm,
                                'generator': generator,
                                'seed': seed,
                                'final_fitness': final_fitness,
                                'total_evaluations': total_evaluations,
                                'evaluations_to_target': evaluations_to_target,
                                'success': data.get('success', False),
                                'reached_target': evaluations_to_target is not None
                            })
                            
                        except Exception as e:
                            print(f"Błąd podczas wczytywania {json_filename}: {e}")
        
        return pd.DataFrame(results)
    
    def _calculate_evaluations_to_target(self, convergence_data: List[Dict], 
                                       target_threshold: float) -> Optional[int]:
        """
        Oblicza liczbę ewaluacji potrzebną do osiągnięcia progu dokładności.
        
        Args:
            convergence_data: Lista punktów zbieżności
            target_threshold: Próg dokładności
            
        Returns:
            Liczba ewaluacji do osiągnięcia celu lub None jeśli cel nie został osiągnięty
        """
        for point in convergence_data:
            if point['best_fitness'] <= target_threshold:
                return point['evaluations']
        return None
    
    def _load_convergence_data(self, function: str, dimension: int, 
                              algorithm: str, generator: str, 
                              seeds: Optional[List[int]] = None) -> List[Dict]:
        """
        Wczytuje dane zbieżności dla określonej konfiguracji.
        
        Args:
            function: Nazwa funkcji testowej
            dimension: Wymiar przestrzeni
            algorithm: Nazwa algorytmu ('standard' lub 'modified')
            generator: Nazwa generatora ('mt' lub 'pcg')
            seeds: Lista ziaren do wczytania (jeśli None, wczytuje wszystkie dostępne)
            
        Returns:
            Lista słowników z danymi zbieżności dla każdego ziarna
        """
        convergence_data = []
        
        # Jeśli nie podano ziaren, użyj wszystkich z konfiguracji
        if seeds is None and self.config:
            seeds = self.config.get('seeds', list(range(1, 31)))
        elif seeds is None:
            seeds = list(range(1, 31))  # Domyślnie 30 ziaren
        
        for seed in seeds:
            # Nazwa pliku z danymi zbieżności
            conv_filename = f"{function}_{dimension}D_{algorithm}_{generator}_seed{seed}_convergence.csv"
            conv_filepath = self.raw_data_dir / conv_filename
            
            if conv_filepath.exists():
                try:
                    # Wczytaj dane z pliku CSV
                    df = pd.read_csv(conv_filepath)
                    
                    # Sprawdź czy dane są prawidłowe
                    if len(df) > 0 and 'evaluations' in df.columns and 'best_fitness' in df.columns:
                        # Filtruj nieprawidłowe wartości
                        valid_mask = (
                            np.isfinite(df['best_fitness']) & 
                            (df['best_fitness'] > 0) &  # Dla skali logarytmicznej
                            np.isfinite(df['evaluations'])
                        )
                        
                        if valid_mask.sum() >= 2:  # Minimum 2 punkty do rysowania
                            filtered_df = df[valid_mask].copy()
                            
                            convergence_data.append({
                                'seed': seed,
                                'evaluations': filtered_df['evaluations'].values,
                                'best_fitness': filtered_df['best_fitness'].values,
                                'sigma': filtered_df.get('sigma', np.ones(len(filtered_df))).values
                            })
                        else:
                            print(f"Zbyt mało prawidłowych punktów dla {conv_filename}")
                    else:
                        print(f"Nieprawidłowa struktura danych w {conv_filename}")
                        
                except Exception as e:
                    print(f"Błąd podczas wczytywania {conv_filename}: {e}")
        
        return convergence_data
    
    def create_boxplots(self, function: str, dimension: int,
                       algorithms: Optional[List[str]] = None,
                       generators: Optional[List[str]] = None,
                       seeds: Optional[List[int]] = None,
                       save_plot: bool = True,
                       figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Tworzy wykresy pudełkowe dla jakości końcowego rozwiązania oraz liczby ewaluacji do osiągnięcia celu.
        
        Args:
            function: Nazwa funkcji testowej
            dimension: Wymiar przestrzeni
            algorithms: Lista algorytmów do porównania (domyślnie ['standard', 'modified'])
            generators: Lista generatorów do porównania (domyślnie ['mt', 'pcg'])
            seeds: Lista ziaren do uwzględnienia (domyślnie wszystkie z konfiguracji)
            save_plot: Czy zapisać wykres do pliku
            figsize: Rozmiar figury (szerokość, wysokość)
        """
        # Domyślne wartości
        if algorithms is None:
            algorithms = ['standard', 'modified']
        if generators is None:
            generators = ['mt', 'pcg']
        
        # Wczytaj dane eksperymentów
        df = self._load_experiment_results(function, dimension, seeds)
        
        if df.empty:
            print(f"Brak danych dla {function} {dimension}D")
            return
        
        # Filtruj według wybranych algorytmów i generatorów
        df_filtered = df[
            (df['algorithm'].isin(algorithms)) & 
            (df['generator'].isin(generators))
        ].copy()
        
        if df_filtered.empty:
            print(f"Brak danych dla wybranych konfiguracji {function} {dimension}D")
            return
        
        # Utwórz etykiety kombinacji algorytm-generator
        df_filtered['config'] = df_filtered['algorithm'] + ' + ' + df_filtered['generator'].str.upper()
        
        # Utwórz subplot: 2 rzędy (jakość, ewaluacje do celu), 2 kolumny (wszystkie dane, tylko udane)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Kolory dla różnych konfiguracji
        config_colors = {}
        for alg in algorithms:
            for gen in generators:
                config = f"{alg} + {gen.upper()}"
                config_colors[config] = self.colors[alg][gen]
        
        # 1. Jakość końcowego rozwiązania - wszystkie dane
        ax1 = axes[0, 0]
        sns.boxplot(data=df_filtered, x='config', y='final_fitness', ax=ax1, 
                   palette=[config_colors.get(c, 'gray') for c in df_filtered['config'].unique()])
        ax1.set_yscale('log')
        ax1.set_title(f"Jakość końcowego rozwiązania - {function.capitalize()} {dimension}D")
        ax1.set_xlabel("Konfiguracja (Algorytm + Generator)")
        ax1.set_ylabel("Wartość funkcji celu (log)")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Dodaj linię progu dokładności
        target_threshold = self._get_accuracy_threshold(function, dimension)
        ax1.axhline(y=target_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Próg dokładności: {target_threshold:.0e}')
        ax1.legend()
        
        # 2. Jakość końcowego rozwiązania - tylko udane eksperymenty
        ax2 = axes[0, 1]
        df_successful = df_filtered[df_filtered['reached_target'] == True]
        if not df_successful.empty:
            sns.boxplot(data=df_successful, x='config', y='final_fitness', ax=ax2,
                       palette=[config_colors.get(c, 'gray') for c in df_successful['config'].unique()])
            ax2.set_yscale('log')
            ax2.set_title(f"Jakość - tylko udane eksperymenty")
            ax2.set_xlabel("Konfiguracja (Algorytm + Generator)")
            ax2.set_ylabel("Wartość funkcji celu (log)")
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=target_threshold, color='red', linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'Brak udanych eksperymentów', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
            ax2.set_title(f"Jakość - tylko udane eksperymenty")
        
        # 3. Liczba ewaluacji do osiągnięcia celu
        ax3 = axes[1, 0]
        if not df_successful.empty:
            sns.boxplot(data=df_successful, x='config', y='evaluations_to_target', ax=ax3,
                       palette=[config_colors.get(c, 'gray') for c in df_successful['config'].unique()])
            ax3.set_title(f"Liczba ewaluacji do osiągnięcia celu")
            ax3.set_xlabel("Konfiguracja (Algorytm + Generator)")
            ax3.set_ylabel("Liczba ewaluacji")
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Brak udanych eksperymentów', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=12, color='red')
            ax3.set_title(f"Liczba ewaluacji do osiągnięcia celu")
        
        # 4. Wskaźnik sukcesu
        ax4 = axes[1, 1]
        success_rates = df_filtered.groupby('config')['reached_target'].agg(['mean', 'count']).reset_index()
        success_rates['success_rate'] = success_rates['mean'] * 100
        
        bars = ax4.bar(success_rates['config'], success_rates['success_rate'], 
                      color=[config_colors.get(c, 'gray') for c in success_rates['config']])
        ax4.set_title(f"Wskaźnik sukcesu")
        ax4.set_xlabel("Konfiguracja (Algorytm + Generator)")
        ax4.set_ylabel("Wskaźnik sukcesu (%)")
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        # Dodaj wartości na słupkach
        for bar, rate in zip(bars, success_rates['success_rate']):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # Główny tytuł
        fig.suptitle(f"Analiza wydajności: {function.capitalize()} {dimension}D", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Zapisz wykres jeśli wymagane
        if save_plot:
            filename = f"{function}_{dimension}D_boxplots_analysis.png"
            filepath = self.boxplot_plots_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres: {filepath}")
        
        plt.show()
        
        # Wyświetl statystyki tekstowe
        self._print_boxplot_statistics(df_filtered, function, dimension)
    
    def _print_boxplot_statistics(self, df: pd.DataFrame, function: str, dimension: int) -> None:
        """
        Wyświetla statystyki tekstowe dla wykresów pudełkowych.
        
        Args:
            df: DataFrame z danymi eksperymentów
            function: Nazwa funkcji
            dimension: Wymiar przestrzeni
        """
        print(f"\n=== Statystyki dla {function.capitalize()} {dimension}D ===")
        
        target_threshold = self._get_accuracy_threshold(function, dimension)
        print(f"Próg dokładności: {target_threshold:.0e}")
        
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            successful = config_data[config_data['reached_target'] == True]
            
            print(f"\n{config}:")
            print(f"  Liczba eksperymentów: {len(config_data)}")
            print(f"  Udane eksperymenty: {len(successful)} ({len(successful)/len(config_data)*100:.1f}%)")
            
            if len(successful) > 0:
                print(f"  Średnia jakość końcowa: {successful['final_fitness'].mean():.2e}")
                print(f"  Mediana jakości końcowej: {successful['final_fitness'].median():.2e}")
                print(f"  Średnia liczba ewaluacji do celu: {successful['evaluations_to_target'].mean():.0f}")
                print(f"  Mediana liczby ewaluacji do celu: {successful['evaluations_to_target'].median():.0f}")
            
            print(f"  Średnia całkowita liczba ewaluacji: {config_data['total_evaluations'].mean():.0f}")
    
    def _calculate_success_rates_matrix(self, algorithms: List[str], generators: List[str], 
                                      seeds: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Oblicza macierz wskaźników sukcesu dla wszystkich kombinacji funkcji, wymiarów, algorytmów i generatorów.
        
        Args:
            algorithms: Lista algorytmów do uwzględnienia
            generators: Lista generatorów do uwzględnienia
            seeds: Lista ziaren do uwzględnienia
            
        Returns:
            DataFrame z wskaźnikami sukcesu
        """
        available = self.get_available_experiments()
        
        if not available['functions']:
            return pd.DataFrame()
        
        results = []
        
        for function in available['functions']:
            for dimension in available['dimensions']:
                for algorithm in algorithms:
                    for generator in generators:
                        # Wczytaj dane dla tej kombinacji
                        df = self._load_experiment_results(function, dimension, seeds)
                        
                        if not df.empty:
                            # Filtruj według algorytmu i generatora
                            subset = df[(df['algorithm'] == algorithm) & (df['generator'] == generator)]
                            
                            if len(subset) > 0:
                                success_count = subset['reached_target'].sum()
                                total_count = len(subset)
                                success_rate = (success_count / total_count) * 100
                                
                                results.append({
                                    'function': function.capitalize(),
                                    'dimension': f"{dimension}D",
                                    'algorithm': algorithm.capitalize(),
                                    'generator': generator.upper(),
                                    'config': f"{algorithm.capitalize()} + {generator.upper()}",
                                    'function_dimension': f"{function.capitalize()} {dimension}D",
                                    'success_rate': success_rate,
                                    'success_count': success_count,
                                    'total_count': total_count
                                })
        
        return pd.DataFrame(results)
    
    def create_success_rate_plots(self, algorithms: Optional[List[str]] = None,
                                 generators: Optional[List[str]] = None,
                                 seeds: Optional[List[int]] = None,
                                 save_plot: bool = True,
                                 figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Tworzy wykresy wskaźników sukcesu: heatmapę oraz wykres słupkowy porównawczy.
        
        Args:
            algorithms: Lista algorytmów do porównania (domyślnie ['standard', 'modified'])
            generators: Lista generatorów do porównania (domyślnie ['mt', 'pcg'])
            seeds: Lista ziaren do uwzględnienia (domyślnie wszystkie z konfiguracji)
            save_plot: Czy zapisać wykres do pliku
            figsize: Rozmiar figury (szerokość, wysokość)
        """
        # Domyślne wartości
        if algorithms is None:
            algorithms = ['standard', 'modified']
        if generators is None:
            generators = ['mt', 'pcg']
        
        # Oblicz macierz wskaźników sukcesu
        df = self._calculate_success_rates_matrix(algorithms, generators, seeds)
        
        if df.empty:
            print("Brak danych do utworzenia wykresów wskaźników sukcesu")
            return
        
        # Utwórz subplot: 2 rzędy (heatmapy, wykresy słupkowe), 2 kolumny (według algorytmu, według generatora)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Kolory dla różnych konfiguracji
        config_colors = {}
        for alg in algorithms:
            for gen in generators:
                config = f"{alg.capitalize()} + {gen.upper()}"
                config_colors[config] = self.colors[alg][gen]
        
        # 1. Heatmapa - porównanie według funkcji i wymiarów (góra-lewo)
        ax1 = axes[0, 0]
        self._create_success_heatmap_by_function(df, ax1)
        
        # 2. Heatmapa - porównanie według konfiguracji (góra-prawo)
        ax2 = axes[0, 1]
        self._create_success_heatmap_by_config(df, ax2)
        
        # 3. Wykres słupkowy - średnie wskaźniki sukcesu według algorytmów (dół-lewo)
        ax3 = axes[1, 0]
        self._create_algorithm_comparison_bars(df, ax3, config_colors)
        
        # 4. Wykres słupkowy - wskaźniki sukcesu według funkcji (dół-prawo)
        ax4 = axes[1, 1]
        self._create_function_comparison_bars(df, ax4, config_colors)
        
        # Główny tytuł
        fig.suptitle("Analiza wskaźników sukcesu", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Zapisz wykres jeśli wymagane
        if save_plot:
            filename = "success_rates_comprehensive_analysis.png"
            filepath = self.success_rate_plots_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres: {filepath}")
        
        plt.show()
        
        # Wyświetl statystyki tekstowe
        self._print_success_rate_statistics(df)
    
    def _create_success_heatmap_by_function(self, df: pd.DataFrame, ax) -> None:
        """
        Tworzy heatmapę wskaźników sukcesu według funkcji i wymiarów.
        """
        # Utworz pivot table dla heatmapy
        pivot_data = df.pivot_table(
            values='success_rate', 
            index='function_dimension', 
            columns='config', 
            aggfunc='mean'
        )
        
        # Utwórz heatmapę
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Wskaźnik sukcesu (%)'})
        ax.set_title("Wskaźniki sukcesu według funkcji i wymiarów")
        ax.set_xlabel("Konfiguracja (Algorytm + Generator)")
        ax.set_ylabel("Funkcja testowa")
        
        # Obróć etykiety osi X
        ax.tick_params(axis='x', rotation=45)
    
    def _create_success_heatmap_by_config(self, df: pd.DataFrame, ax) -> None:
        """
        Tworzy heatmapę wskaźników sukcesu według konfiguracji i funkcji.
        """
        # Utworz dane dla heatmapy - transpozycja poprzedniej
        pivot_data = df.pivot_table(
            values='success_rate',
            index='config',
            columns='function_dimension',
            aggfunc='mean'
        )
        
        # Utwórz heatmapę
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Wskaźnik sukcesu (%)'})
        ax.set_title("Wskaźniki sukcesu według konfiguracji")
        ax.set_xlabel("Funkcja testowa")
        ax.set_ylabel("Konfiguracja (Algorytm + Generator)")
        
        # Obróć etykiety osi X
        ax.tick_params(axis='x', rotation=45)
    
    def _create_algorithm_comparison_bars(self, df: pd.DataFrame, ax, config_colors: Dict) -> None:
        """
        Tworzy wykres słupkowy porównujący średnie wskaźniki sukcesu według algorytmów.
        """
        # Oblicz średnie wskaźniki sukcesu dla każdej konfiguracji
        avg_success = df.groupby('config')['success_rate'].agg(['mean', 'std']).reset_index()
        avg_success = avg_success.sort_values('mean', ascending=False)
        
        # Utwórz wykres słupkowy z błędami
        bars = ax.bar(avg_success['config'], avg_success['mean'], 
                     yerr=avg_success['std'], capsize=5,
                     color=[config_colors.get(c, 'gray') for c in avg_success['config']])
        
        ax.set_title("Średnie wskaźniki sukcesu według konfiguracji")
        ax.set_xlabel("Konfiguracja (Algorytm + Generator)")
        ax.set_ylabel("Średni wskaźnik sukcesu (%)")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Dodaj wartości na słupkach
        for bar, mean_val, std_val in zip(bars, avg_success['mean'], avg_success['std']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 2,
                   f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _create_function_comparison_bars(self, df: pd.DataFrame, ax, config_colors: Dict) -> None:
        """
        Tworzy wykres słupkowy porównujący wskaźniki sukcesu według funkcji.
        """
        # Oblicz średnie wskaźniki sukcesu dla każdej funkcji
        func_success = df.groupby(['function', 'config'])['success_rate'].mean().reset_index()
        
        # Utwórz grouped bar chart
        functions = func_success['function'].unique()
        configs = func_success['config'].unique()
        
        x = np.arange(len(functions))
        width = 0.8 / len(configs)
        
        for i, config in enumerate(configs):
            config_data = func_success[func_success['config'] == config]
            values = [config_data[config_data['function'] == func]['success_rate'].iloc[0] 
                     if len(config_data[config_data['function'] == func]) > 0 else 0 
                     for func in functions]
            
            bars = ax.bar(x + i * width, values, width, 
                         label=config, color=config_colors.get(config, 'gray'))
            
            # Dodaj wartości na słupkach
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                           f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title("Wskaźniki sukcesu według funkcji testowych")
        ax.set_xlabel("Funkcja testowa")
        ax.set_ylabel("Średni wskaźnik sukcesu (%)")
        ax.set_xticks(x + width * (len(configs) - 1) / 2)
        ax.set_xticklabels(functions)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def _print_success_rate_statistics(self, df: pd.DataFrame) -> None:
        """
        Wyświetla statystyki tekstowe dla wskaźników sukcesu.
        """
        print(f"\n=== Statystyki wskaźników sukcesu ===")
        print(f"Liczba eksperymentów: {df['total_count'].iloc[0] if len(df) > 0 else 0} na konfigurację")
        print(f"Próg dokładności: {list(self.accuracy_thresholds.values())[0]:.0e}")
        
        # Statystyki według konfiguracji
        print(f"\n--- Wskaźniki sukcesu według konfiguracji ---")
        config_stats = df.groupby('config')['success_rate'].agg(['mean', 'std', 'min', 'max'])
        config_stats = config_stats.sort_values('mean', ascending=False)
        
        for config, stats in config_stats.iterrows():
            print(f"{config}:")
            print(f"  Średni wskaźnik sukcesu: {stats['mean']:.1f}% (σ={stats['std']:.1f}%)")
            print(f"  Zakres: {stats['min']:.1f}% - {stats['max']:.1f}%")
        
        # Statystyki według funkcji
        print(f"\n--- Wskaźniki sukcesu według funkcji ---")
        func_stats = df.groupby('function')['success_rate'].agg(['mean', 'std', 'min', 'max'])
        func_stats = func_stats.sort_values('mean', ascending=False)
        
        for function, stats in func_stats.iterrows():
            print(f"{function}:")
            print(f"  Średni wskaźnik sukcesu: {stats['mean']:.1f}% (σ={stats['std']:.1f}%)")
            print(f"  Zakres: {stats['min']:.1f}% - {stats['max']:.1f}%")
        
        # Najlepsze i najgorsze przypadki
        print(f"\n--- Najlepsze i najgorsze przypadki ---")
        best_case = df.loc[df['success_rate'].idxmax()]
        worst_case = df.loc[df['success_rate'].idxmin()]
        
        print(f"Najlepszy przypadek: {best_case['function_dimension']} + {best_case['config']} ({best_case['success_rate']:.1f}%)")
        print(f"Najgorszy przypadek: {worst_case['function_dimension']} + {worst_case['config']} ({worst_case['success_rate']:.1f}%)")
        
        # Porównanie algorytmów
        if 'Standard' in df['algorithm'].values and 'Modified' in df['algorithm'].values:
            print(f"\n--- Porównanie algorytmów ---")
            standard_avg = df[df['algorithm'] == 'Standard']['success_rate'].mean()
            modified_avg = df[df['algorithm'] == 'Modified']['success_rate'].mean()
            improvement = modified_avg - standard_avg
            
            print(f"Standard CMA-ES: {standard_avg:.1f}%")
            print(f"Modified CMA-ES: {modified_avg:.1f}%")
            print(f"Poprawa: {improvement:+.1f} punktów procentowych")
            
            if improvement > 0:
                print(f"✓ Modified CMA-ES jest lepszy o {improvement:.1f} pp")
            elif improvement < 0:
                print(f"✗ Modified CMA-ES jest gorszy o {abs(improvement):.1f} pp")
            else:
                print("= Algorytmy mają identyczną wydajność")
    
    def create_convergence_curves(self, function: str, dimension: int,
                                 algorithms: Optional[List[str]] = None,
                                 generators: Optional[List[str]] = None,
                                 seeds: Optional[List[int]] = None,
                                 show_individual: bool = True,
                                 show_statistics: bool = True,
                                 save_plot: bool = True,
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Tworzy wykresy krzywych zbieżności na podstawie danych z katalogu results/final.
        
        Args:
            function: Nazwa funkcji testowej
            dimension: Wymiar przestrzeni
            algorithms: Lista algorytmów do porównania (domyślnie ['standard', 'modified'])
            generators: Lista generatorów do porównania (domyślnie ['mt', 'pcg'])
            seeds: Lista ziaren do uwzględnienia (domyślnie wszystkie z konfiguracji)
            show_individual: Czy pokazywać indywidualne krzywe dla każdego ziarna
            show_statistics: Czy pokazywać statystyki (mediana, kwartyle)
            save_plot: Czy zapisać wykres do pliku
            figsize: Rozmiar figury (szerokość, wysokość)
        """
        # Domyślne wartości
        if algorithms is None:
            algorithms = ['standard', 'modified']
        if generators is None:
            generators = ['mt', 'pcg']
        
        # Utwórz subplot dla każdego generatora
        fig, axes = plt.subplots(1, len(generators), figsize=figsize, sharey=True)
        if len(generators) == 1:
            axes = [axes]
        
        for gen_idx, generator in enumerate(generators):
            ax = axes[gen_idx]
            
            for algorithm in algorithms:
                # Wczytaj dane zbieżności
                conv_data = self._load_convergence_data(function, dimension, algorithm, generator, seeds)
                
                if not conv_data:
                    print(f"Brak danych dla {function} {dimension}D {algorithm} {generator}")
                    continue
                
                color = self.colors[algorithm][generator]
                
                # Rysuj indywidualne krzywe jeśli wymagane
                if show_individual:
                    for data in conv_data:
                        ax.semilogy(data['evaluations'], data['best_fitness'], 
                                   color=color, alpha=0.2, linewidth=0.5)
                
                # Oblicz i rysuj statystyki jeśli wymagane
                if show_statistics and len(conv_data) > 1:
                    self._plot_convergence_statistics(ax, conv_data, color, algorithm)
                elif len(conv_data) == 1:
                    # Jeśli tylko jedna krzywa, narysuj ją jako główną
                    data = conv_data[0]
                    ax.semilogy(data['evaluations'], data['best_fitness'], 
                               color=color, linewidth=2, label=f"{algorithm.capitalize()}")
            
            # Ustawienia osi i etykiet
            ax.set_xlabel("Liczba ewaluacji")
            if gen_idx == 0:
                ax.set_ylabel("Wartość funkcji celu (skala log)")
            ax.set_title(f"Generator: {generator.upper()}")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend()
            
            # Ustaw limity osi Y
            ax.set_ylim(bottom=1e-10)
        
        # Główny tytuł
        fig.suptitle(f"Krzywe zbieżności: {function.capitalize()} {dimension}D", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Zapisz wykres jeśli wymagane
        if save_plot:
            filename = f"{function}_{dimension}D_convergence_comparison.png"
            filepath = self.convergence_plots_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Zapisano wykres: {filepath}")
        
        plt.show()
    
    def _plot_convergence_statistics(self, ax, conv_data: List[Dict], color: str, label: str) -> None:
        """
        Rysuje statystyki zbieżności (mediana, kwartyle) na wykresie.
        
        Args:
            ax: Obiekt osi matplotlib
            conv_data: Lista danych zbieżności
            color: Kolor linii
            label: Etykieta dla legendy
        """
        # Znajdź wspólną siatkę punktów ewaluacji
        all_evaluations = []
        for data in conv_data:
            all_evaluations.extend(data['evaluations'])
        
        common_evals = np.unique(all_evaluations)
        common_evals.sort()
        
        # Interpoluj dane na wspólną siatkę
        interpolated_fitness = []
        
        for data in conv_data:
            if len(data['evaluations']) < 2:
                continue
                
            try:
                from scipy.interpolate import interp1d
                
                # Upewnij się, że ewaluacje są unikalne
                eval_unique, indices = np.unique(data['evaluations'], return_index=True)
                fitness_unique = data['best_fitness'][indices]
                
                if len(eval_unique) < 2:
                    continue
                
                # Interpolacja
                f = interp1d(eval_unique, fitness_unique, kind='linear', 
                           bounds_error=False, fill_value=(fitness_unique[0], fitness_unique[-1]))
                interpolated = f(common_evals)
                
                # Sprawdź czy interpolacja jest prawidłowa
                if np.any(np.isfinite(interpolated)):
                    interpolated_fitness.append(interpolated)
                    
            except Exception as e:
                print(f"Błąd interpolacji: {e}")
                continue
        
        if len(interpolated_fitness) > 0:
            interpolated_fitness = np.array(interpolated_fitness)
            
            # Oblicz statystyki
            median_fitness = np.nanmedian(interpolated_fitness, axis=0)
            q25_fitness = np.nanpercentile(interpolated_fitness, 25, axis=0)
            q75_fitness = np.nanpercentile(interpolated_fitness, 75, axis=0)
            
            # Filtruj prawidłowe punkty
            valid_mask = (
                np.isfinite(median_fitness) & 
                (median_fitness > 0) &
                np.isfinite(q25_fitness) & 
                np.isfinite(q75_fitness)
            )
            
            if np.any(valid_mask):
                valid_evals = common_evals[valid_mask]
                valid_median = median_fitness[valid_mask]
                valid_q25 = q25_fitness[valid_mask]
                valid_q75 = q75_fitness[valid_mask]
                
                # Rysuj medianę
                ax.semilogy(valid_evals, valid_median, color=color, linewidth=2, 
                           label=f"{label.capitalize()}")
                
                # Rysuj przedział kwartylowy
                ax.fill_between(valid_evals, 
                               np.maximum(valid_q25, np.full_like(valid_q25, 1e-10)),
                               valid_q75,
                               color=color, alpha=0.2)
    
    def get_available_experiments(self) -> Dict[str, List]:
        """
        Zwraca listę dostępnych eksperymentów w katalogu danych.
        
        Returns:
            Słownik z listami dostępnych funkcji, wymiarów, algorytmów i generatorów
        """
        if not self.raw_data_dir.exists():
            return {'functions': [], 'dimensions': [], 'algorithms': [], 'generators': []}
        
        functions = set()
        dimensions = set()
        algorithms = set()
        generators = set()
        
        # Przeszukaj pliki convergence
        for filepath in self.raw_data_dir.glob("*_convergence.csv"):
            filename = filepath.stem
            # Format: function_dimensionD_algorithm_generator_seedN_convergence
            parts = filename.replace('_convergence', '').split('_')
            
            if len(parts) >= 4:
                function = parts[0]
                dimension_str = parts[1]
                algorithm = parts[2]
                generator = parts[3]
                
                functions.add(function)
                if dimension_str.endswith('D'):
                    try:
                        dimension = int(dimension_str[:-1])
                        dimensions.add(dimension)
                    except ValueError:
                        pass
                algorithms.add(algorithm)
                generators.add(generator)
        
        return {
            'functions': sorted(list(functions)),
            'dimensions': sorted(list(dimensions)),
            'algorithms': sorted(list(algorithms)),
            'generators': sorted(list(generators))
        }
    
    def create_all_convergence_plots(self) -> None:
        """
        Tworzy wykresy krzywych zbieżności dla wszystkich dostępnych kombinacji.
        """
        available = self.get_available_experiments()
        
        print("Tworzenie wykresów krzywych zbieżności...")
        print(f"Dostępne eksperymenty: {available}")
        
        for function in available['functions']:
            for dimension in available['dimensions']:
                print(f"Tworzenie wykresu dla {function} {dimension}D...")
                try:
                    self.create_convergence_curves(
                        function=function,
                        dimension=dimension,
                        algorithms=available['algorithms'],
                        generators=available['generators'],
                        show_individual=True,
                        show_statistics=True,
                        save_plot=False
                    )
                    plt.close()  # Zamknij figurę aby zwolnić pamięć
                except Exception as e:
                    print(f"Błąd podczas tworzenia wykresu dla {function} {dimension}D: {e}")
        
        print(f"Wykresy zapisane w katalogu: {self.convergence_plots_dir}")
    
    def create_all_boxplots(self) -> None:
        """
        Tworzy wykresy pudełkowe dla wszystkich dostępnych kombinacji.
        """
        available = self.get_available_experiments()
        
        print("Tworzenie wykresów pudełkowych...")
        print(f"Dostępne eksperymenty: {available}")
        
        for function in available['functions']:
            for dimension in available['dimensions']:
                print(f"Tworzenie wykresów pudełkowych dla {function} {dimension}D...")
                try:
                    self.create_boxplots(
                        function=function,
                        dimension=dimension,
                        algorithms=available['algorithms'],
                        generators=available['generators'],
                        save_plot=True
                    )
                    plt.close()  # Zamknij figurę aby zwolnić pamięć
                except Exception as e:
                    print(f"Błąd podczas tworzenia wykresów pudełkowych dla {function} {dimension}D: {e}")
        
        print(f"Wykresy pudełkowe zapisane w katalogu: {self.boxplot_plots_dir}")
    
    def create_all_success_rate_plots(self) -> None:
        """
        Tworzy wykresy wskaźników sukcesu dla wszystkich dostępnych danych.
        """
        available = self.get_available_experiments()
        
        print("Tworzenie wykresów wskaźników sukcesu...")
        print(f"Dostępne eksperymenty: {available}")
        
        try:
            self.create_success_rate_plots(
                algorithms=available['algorithms'],
                generators=available['generators'],
                save_plot=True
            )
            plt.close()  # Zamknij figurę aby zwolnić pamięć
            print("✓ Wykresy wskaźników sukcesu utworzone pomyślnie!")
        except Exception as e:
            print(f"Błąd podczas tworzenia wykresów wskaźników sukcesu: {e}")
        
        print(f"Wykresy wskaźników sukcesu zapisane w katalogu: {self.success_rate_plots_dir}") 