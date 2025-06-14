#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prosta analiza statystyczna wyników CMA-ES Standard vs Modified.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SimpleStatisticalAnalysis:
    """Prosta analiza statystyczna różnic między algorytmami."""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.data = pd.read_csv(results_file)
        
    def load_individual_results(self):
        """Wczytuje indywidualne wyniki z raw_data (jeśli dostępne)."""
        try:
            raw_data_dir = os.path.join(os.path.dirname(self.results_file), 'raw_data')
            individual_data = {}
            
            if os.path.exists(raw_data_dir):
                import json
                for filename in os.listdir(raw_data_dir):
                    if filename.endswith('.json') and not filename.endswith('_convergence.json'):
                        filepath = os.path.join(raw_data_dir, filename)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            
                        key = (data['function'], data['dimension'], 
                               data['algorithm'], data['generator'])
                        
                        if key not in individual_data:
                            individual_data[key] = []
                        individual_data[key].append(data['result']['fun'])
                        
            return individual_data
        except:
            return {}
    
    def test_normality(self, data: np.ndarray) -> dict:
        """Test normalności Shapiro-Wilk."""
        if len(data) < 3:
            return {'is_normal': False, 'p_value': np.nan}
        
        try:
            stat, p_value = stats.shapiro(data)
            return {
                'is_normal': p_value > 0.05,
                'p_value': p_value,
                'statistic': stat
            }
        except:
            return {'is_normal': False, 'p_value': np.nan}
    
    def wilcoxon_test(self, data1: np.ndarray, data2: np.ndarray) -> dict:
        """Test Wilcoxona dla par skorelowanych."""
        if len(data1) != len(data2):
            return {'error': 'Różne rozmiary próbek'}
        
        differences = data1 - data2
        if np.all(differences == 0):
            return {
                'p_value': 1.0,
                'significant': False,
                'better_algorithm': 'brak różnicy'
            }
        
        try:
            stat, p_value = stats.wilcoxon(data1, data2)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            
            return {
                'p_value': p_value,
                'significant': p_value < 0.05,
                'better_algorithm': 'standard' if mean1 < mean2 else 'modified',
                'statistic': stat,
                'mean_diff': mean1 - mean2
            }
        except Exception as e:
            return {'error': str(e)}
    
    def cohen_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Oblicza Cohen's d (effect size)."""
        n1, n2 = len(data1), len(data2)
        if n1 < 2 or n2 < 2:
            return 0.0
        
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
        
        return abs(np.mean(data1) - np.mean(data2)) / pooled_std
    
    def analyze_summary_data(self):
        """Analiza na podstawie danych podsumowujących."""
        print("📊 ANALIZA STATYSTYCZNA - DANE PODSUMOWUJĄCE")
        print("=" * 60)
        
        results = []
        
        # Grupuj dane dla porównania Standard vs Modified
        for function in self.data['function'].unique():
            for dimension in self.data['dimension'].unique():
                for generator in self.data['generator'].unique():
                    
                    std_row = self.data[
                        (self.data['function'] == function) &
                        (self.data['dimension'] == dimension) &
                        (self.data['algorithm'] == 'standard') &
                        (self.data['generator'] == generator)
                    ]
                    
                    mod_row = self.data[
                        (self.data['function'] == function) &
                        (self.data['dimension'] == dimension) &
                        (self.data['algorithm'] == 'modified') &
                        (self.data['generator'] == generator)
                    ]
                    
                    if len(std_row) == 1 and len(mod_row) == 1:
                        std_mean = std_row['best_fitness_mean'].iloc[0]
                        std_std = std_row['best_fitness_std'].iloc[0]
                        mod_mean = mod_row['best_fitness_mean'].iloc[0]
                        mod_std = mod_row['best_fitness_std'].iloc[0]
                        
                        # Przybliżony test z-score dla różnicy średnich
                        se_diff = np.sqrt(std_std**2 + mod_std**2) / np.sqrt(30)  # n=30
                        if se_diff > 0:
                            z_score = abs(std_mean - mod_mean) / se_diff
                            p_value_approx = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        else:
                            p_value_approx = 1.0
                        
                        # Przybliżony Cohen's d
                        pooled_std = np.sqrt((std_std**2 + mod_std**2) / 2)
                        cohen_d_approx = abs(std_mean - mod_mean) / pooled_std if pooled_std > 0 else 0
                        
                        results.append({
                            'function': function,
                            'dimension': dimension,
                            'generator': generator,
                            'std_mean': std_mean,
                            'std_std': std_std,
                            'mod_mean': mod_mean,
                            'mod_std': mod_std,
                            'better_algorithm': 'standard' if std_mean < mod_mean else 'modified',
                            'mean_difference': std_mean - mod_mean,
                            'relative_improvement': abs(std_mean - mod_mean) / max(abs(std_mean), abs(mod_mean)) * 100,
                            'p_value_approx': p_value_approx,
                            'significant_approx': p_value_approx < 0.05,
                            'cohen_d_approx': cohen_d_approx
                        })
        
        return pd.DataFrame(results)
    
    def analyze_individual_data(self):
        """Analiza na podstawie indywidualnych wyników."""
        individual_data = self.load_individual_results()
        
        if not individual_data:
            print("❌ Brak indywidualnych danych - używam tylko danych podsumowujących")
            return self.analyze_summary_data()
        
        print("📊 ANALIZA STATYSTYCZNA - INDYWIDUALNE WYNIKI")
        print("=" * 60)
        
        results = []
        
        # Znajdź pary Standard vs Modified
        for function in set(key[0] for key in individual_data.keys()):
            for dimension in set(key[1] for key in individual_data.keys() if key[0] == function):
                for generator in set(key[3] for key in individual_data.keys() 
                                   if key[0] == function and key[1] == dimension):
                    
                    std_key = (function, dimension, 'standard', generator)
                    mod_key = (function, dimension, 'modified', generator)
                    
                    if std_key in individual_data and mod_key in individual_data:
                        std_data = np.array(individual_data[std_key])
                        mod_data = np.array(individual_data[mod_key])
                        
                        # Testy normalności
                        std_norm = self.test_normality(std_data)
                        mod_norm = self.test_normality(mod_data)
                        
                        # Test Wilcoxona
                        wilcox_result = self.wilcoxon_test(std_data, mod_data)
                        
                        # Effect size
                        cohen_d_val = self.cohen_d(std_data, mod_data)
                        
                        results.append({
                            'function': function,
                            'dimension': dimension,
                            'generator': generator,
                            'std_mean': np.mean(std_data),
                            'std_median': np.median(std_data),
                            'std_std': np.std(std_data),
                            'std_is_normal': std_norm['is_normal'],
                            'std_normality_p': std_norm['p_value'],
                            'mod_mean': np.mean(mod_data),
                            'mod_median': np.median(mod_data),
                            'mod_std': np.std(mod_data),
                            'mod_is_normal': mod_norm['is_normal'],
                            'mod_normality_p': mod_norm['p_value'],
                            'wilcoxon_p': wilcox_result.get('p_value', np.nan),
                            'wilcoxon_significant': wilcox_result.get('significant', False),
                            'better_algorithm': wilcox_result.get('better_algorithm', 'brak'),
                            'cohen_d': cohen_d_val,
                            'sample_size': len(std_data)
                        })
        
        return pd.DataFrame(results)
    
    def create_comparison_plots(self, results_df):
        """Tworzy wykresy porównawcze."""
        if len(results_df) == 0:
            return
        
        # Setup
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analiza Statystyczna: Standard vs Modified CMA-ES', fontsize=16)
        
        # 1. Box plot porównanie średnich
        ax1 = axes[0, 0]
        comparison_data = []
        labels = []
        
        for _, row in results_df.iterrows():
            comparison_data.extend([row['std_mean'], row['mod_mean']])
            labels.extend([f"{row['function'][:3]}-{row['dimension']}-S", 
                          f"{row['function'][:3]}-{row['dimension']}-M"])
        
        if len(comparison_data) > 0:
            ax1.boxplot([comparison_data[::2], comparison_data[1::2]], 
                       labels=['Standard', 'Modified'])
            ax1.set_yscale('log')
            ax1.set_title('Porównanie średnich fitness')
            ax1.set_ylabel('Best fitness (log scale)')
        
        # 2. P-values z testów
        ax2 = axes[0, 1]
        if 'wilcoxon_p' in results_df.columns:
            p_values = results_df['wilcoxon_p'].dropna()
            if len(p_values) > 0:
                ax2.hist(p_values, bins=10, alpha=0.7, edgecolor='black')
                ax2.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
                ax2.set_title('Rozkład p-values (Test Wilcoxona)')
                ax2.set_xlabel('p-value')
                ax2.set_ylabel('Częstość')
                ax2.legend()
        
        # 3. Effect sizes (Cohen's d)
        ax3 = axes[1, 0]
        if 'cohen_d' in results_df.columns:
            effect_sizes = results_df['cohen_d'].dropna()
            if len(effect_sizes) > 0:
                ax3.hist(effect_sizes, bins=10, alpha=0.7, edgecolor='black')
                ax3.axvline(x=0.2, color='orange', linestyle='--', label='Mały efekt')
                ax3.axvline(x=0.8, color='red', linestyle='--', label='Duży efekt')
                ax3.set_title('Rozkład Effect Sizes (Cohen\'s d)')
                ax3.set_xlabel('Cohen\'s d')
                ax3.set_ylabel('Częstość')
                ax3.legend()
        
        # 4. Który algorytm wygrywa
        ax4 = axes[1, 1]
        if 'better_algorithm' in results_df.columns:
            winners = results_df['better_algorithm'].value_counts()
            if len(winners) > 0:
                ax4.pie(winners.values, labels=winners.index, autopct='%1.1f%%')
                ax4.set_title('Który algorytm wygrywa częściej')
        
        plt.tight_layout()
        plt.savefig('results/final/statistical_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Wykres zapisany jako: statistical_analysis_plots.png")
    
    def generate_report(self, results_df):
        """Generuje raport z analizy."""
        print("\n" + "=" * 80)
        print("RAPORT ANALIZY STATYSTYCZNEJ")
        print("=" * 80)
        
        if len(results_df) == 0:
            print("❌ Brak danych do analizy")
            return
        
        # Podstawowe statystyki
        print(f"\n📊 PODSTAWOWE INFORMACJE:")
        print(f"   Liczba porównań: {len(results_df)}")
        print(f"   Funkcje testowe: {results_df['function'].unique()}")
        print(f"   Wymiary: {sorted(results_df['dimension'].unique())}")
        
        # Analiza normalności (jeśli dostępna)
        if 'std_is_normal' in results_df.columns:
            normal_std = results_df['std_is_normal'].sum()
            normal_mod = results_df['mod_is_normal'].sum()
            total = len(results_df)
            
            print(f"\n🔬 TESTY NORMALNOŚCI:")
            print(f"   Standard: {normal_std}/{total} ({100*normal_std/total:.1f}%) normalnych")
            print(f"   Modified: {normal_mod}/{total} ({100*normal_mod/total:.1f}%) normalnych")
        
        # Analiza testów Wilcoxona
        if 'wilcoxon_significant' in results_df.columns:
            significant = results_df['wilcoxon_significant'].sum()
            total = len(results_df)
            
            print(f"\n📈 TESTY WILCOXONA:")
            print(f"   Istotne różnice: {significant}/{total} ({100*significant/total:.1f}%)")
            
            if significant > 0:
                winners = results_df[results_df['wilcoxon_significant']]['better_algorithm'].value_counts()
                print(f"   Zwycięzcy w istotnych przypadkach:")
                for alg, count in winners.items():
                    print(f"     {alg}: {count} przypadków")
        
        # Effect sizes
        if 'cohen_d' in results_df.columns:
            effect_sizes = results_df['cohen_d'].dropna()
            if len(effect_sizes) > 0:
                print(f"\n📏 ROZMIARY EFEKTÓW (Cohen's d):")
                print(f"   Średnia: {effect_sizes.mean():.3f}")
                print(f"   Mediana: {effect_sizes.median():.3f}")
                print(f"   Maksymalna: {effect_sizes.max():.3f}")
                
                small = (effect_sizes < 0.2).sum()
                medium = ((effect_sizes >= 0.2) & (effect_sizes < 0.8)).sum()
                large = (effect_sizes >= 0.8).sum()
                
                print(f"   Klasyfikacja:")
                print(f"     Małe (<0.2): {small} przypadków")
                print(f"     Średnie (0.2-0.8): {medium} przypadków")
                print(f"     Duże (≥0.8): {large} przypadków")
        
        # Szczegółowe wyniki
        print(f"\n📋 SZCZEGÓŁOWE WYNIKI:")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            print(f"\n🔍 {row['function'].upper()} {row['dimension']}D ({row['generator']}):")
            print(f"   Standard: {row['std_mean']:.3e} ± {row.get('std_std', 0):.3e}")
            print(f"   Modified: {row['mod_mean']:.3e} ± {row.get('mod_std', 0):.3e}")
            
            if 'wilcoxon_p' in row and not pd.isna(row['wilcoxon_p']):
                significance = "⭐ ISTOTNE" if row.get('wilcoxon_significant', False) else "nieistotne"
                print(f"   Wilcoxon p-value: {row['wilcoxon_p']:.4f} ({significance})")
                print(f"   Lepszy algorytm: {row.get('better_algorithm', 'brak')}")
            
            if 'cohen_d' in row and not pd.isna(row['cohen_d']):
                print(f"   Effect size (Cohen's d): {row['cohen_d']:.3f}")
        
        # Zapisz wyniki do CSV
        output_file = 'results/final/statistical_analysis_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Wyniki zapisane do: {output_file}")
    
    def run_analysis(self):
        """Główna funkcja uruchamiająca analizę."""
        print("🚀 Rozpoczynam analizę statystyczną...")
        
        # Spróbuj analiza indywidualnych danych
        results_df = self.analyze_individual_data()
        
        # Jeśli brak indywidualnych danych, użyj podsumowujących
        if len(results_df) == 0:
            results_df = self.analyze_summary_data()
        
        # Wygeneruj raport
        self.generate_report(results_df)
        
        # Stwórz wykresy
        self.create_comparison_plots(results_df)
        
        return results_df

# UŻYCIE
if __name__ == "__main__":
    # Uruchom analizę
    analyzer = SimpleStatisticalAnalysis("results/final/all_results_summary.csv")
    results = analyzer.run_analysis()
    
    print("\n✅ Analiza zakończona!")