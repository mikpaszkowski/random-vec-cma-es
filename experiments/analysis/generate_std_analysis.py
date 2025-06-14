#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns
from pathlib import Path

def load_experiment_data(results_dir="results/final"):
    """
    Ładuje wszystkie dane eksperymentów z katalogu results/final
    
    Returns:
        Dict z danymi eksperymentów pogrupowanymi według konfiguracji
    """
    
    # Wczytaj podsumowanie wyników
    summary_file = os.path.join(results_dir, "all_results_summary.csv")
    summary_df = pd.read_csv(summary_file)
    
    # Wczytaj indywidualne krzywe zbieżności
    convergence_files = glob.glob(os.path.join(results_dir, "raw_data", "*_convergence.csv"))
    
    experiments_data = {}
    
    for file_path in convergence_files:
        # Parsuj nazwę pliku aby wyodrębnić parametry
        filename = os.path.basename(file_path)
        parts = filename.replace("_convergence.csv", "").split("_")
        
        if len(parts) >= 4:
            function = parts[0]
            dimension = parts[1].replace("D", "")
            algorithm = parts[2]
            generator = parts[3]
            seed = parts[4].replace("seed", "")
            
            # Utwórz klucz dla konfiguracji
            config_key = f"{function}_{dimension}D_{algorithm}_{generator}"
            
            if config_key not in experiments_data:
                experiments_data[config_key] = {
                    'config': {
                        'function': function,
                        'dimension': int(dimension),
                        'algorithm': algorithm,
                        'generator': generator
                    },
                    'convergence_curves': [],
                    'final_fitness': [],
                    'total_evaluations': []
                }
            
            # Wczytaj dane zbieżności
            try:
                conv_data = pd.read_csv(file_path)
                experiments_data[config_key]['convergence_curves'].append(conv_data)
                experiments_data[config_key]['final_fitness'].append(conv_data['best_fitness'].iloc[-1])
                experiments_data[config_key]['total_evaluations'].append(conv_data['evaluations'].iloc[-1])
            except Exception as e:
                print(f"Błąd przy wczytywaniu {file_path}: {e}")
    
    return experiments_data, summary_df

def plot_std_analysis(experiments_data, summary_df):
    """
    Tworzy kompleksowe wykresy analizy odchylenia standardowego
    """
    
    # Set style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # 1. Wykres odchylenia standardowego fitness według konfiguracji
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analiza Odchylenia Standardowego Wyników Eksperymentów CMA-ES', fontsize=16, fontweight='bold')
    
    # Subplot 1: Odchylenie standardowe według funkcji i algorytmu
    ax1 = axes[0, 0]
    
    # Przygotuj dane dla wykresu
    plot_data = []
    for _, row in summary_df.iterrows():
        plot_data.append({
            'function': row['function'],
            'dimension': f"{row['dimension']}D",
            'algorithm': row['algorithm'],
            'generator': row['generator'],
            'std_fitness': row['best_fitness_std'],
            'std_evaluations': row['evaluations_std'],
            'mean_fitness': row['best_fitness_mean'],
            'config': f"{row['algorithm']}_{row['generator']}"
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Grupa dla funkcji
    functions = plot_df['function'].unique()
    x_pos = np.arange(len(functions))
    width = 0.15
    
    configs = plot_df['config'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    
    for i, config in enumerate(configs):
        config_data = plot_df[plot_df['config'] == config]
        y_values = []
        for func in functions:
            func_data = config_data[config_data['function'] == func]
            if not func_data.empty:
                y_values.append(func_data['std_fitness'].iloc[0])
            else:
                y_values.append(0)
        
        ax1.bar(x_pos + i*width, y_values, width, label=config, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Funkcja testowa')
    ax1.set_ylabel('Odchylenie standardowe fitness')
    ax1.set_title('Odchylenie standardowe końcowego fitness')
    ax1.set_xticks(x_pos + width * 1.5)
    ax1.set_xticklabels(functions)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Odchylenie standardowe liczby ewaluacji
    ax2 = axes[0, 1]
    
    for i, config in enumerate(configs):
        config_data = plot_df[plot_df['config'] == config]
        y_values = []
        for func in functions:
            func_data = config_data[config_data['function'] == func]
            if not func_data.empty:
                y_values.append(func_data['std_evaluations'].iloc[0])
            else:
                y_values.append(0)
        
        ax2.bar(x_pos + i*width, y_values, width, label=config, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Funkcja testowa')
    ax2.set_ylabel('Odchylenie standardowe liczby ewaluacji')
    ax2.set_title('Odchylenie standardowe liczby ewaluacji')
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels(functions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Porównanie względnego odchylenia standardowego
    ax3 = axes[1, 0]
    
    # Oblicz współczynnik zmienności (CV = std/mean)
    plot_df['cv_fitness'] = plot_df['std_fitness'] / np.abs(plot_df['mean_fitness'])
    
    for i, config in enumerate(configs):
        config_data = plot_df[plot_df['config'] == config]
        y_values = []
        for func in functions:
            func_data = config_data[config_data['function'] == func]
            if not func_data.empty:
                cv = func_data['cv_fitness'].iloc[0]
                y_values.append(cv if not np.isnan(cv) and np.isfinite(cv) else 0)
            else:
                y_values.append(0)
        
        ax3.bar(x_pos + i*width, y_values, width, label=config, color=colors[i], alpha=0.8)
    
    ax3.set_xlabel('Funkcja testowa')
    ax3.set_ylabel('Współczynnik zmienności (std/mean)')
    ax3.set_title('Względne odchylenie standardowe fitness')
    ax3.set_xticks(x_pos + width * 1.5)
    ax3.set_xticklabels(functions)
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Box plot dla różnych konfiguracji
    ax4 = axes[1, 1]
    
    # Przygotuj dane dla box plot
    box_data = []
    labels = []
    
    for config_key, data in experiments_data.items():
        if len(data['final_fitness']) > 5:  # Tylko konfiguracje z wystarczającą liczbą prób
            box_data.append(data['final_fitness'])
            config = data['config']
            labels.append(f"{config['function'][:3]}_{config['algorithm'][:3]}_{config['generator']}")
    
    if box_data:
        box_plot = ax4.boxplot(box_data, labels=labels, patch_artist=True)
        
        # Koloruj pudełka
        for patch, color in zip(box_plot['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax4.set_xlabel('Konfiguracja')
    ax4.set_ylabel('Końcowe fitness (log scale)')
    ax4.set_title('Rozkład końcowego fitness')
    ax4.set_yscale('log')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('std_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_df

def plot_convergence_std(experiments_data):
    """
    Tworzy wykres średnich krzywych zbieżności z pasami odchylenia standardowego
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Krzywe Zbieżności z Odchyleniem Standardowym', fontsize=16, fontweight='bold')
    
    # Wybierz reprezentatywne konfiguracje
    configs_to_plot = []
    for config_key, data in experiments_data.items():
        if len(data['convergence_curves']) >= 10:  # Wystarczająco prób
            configs_to_plot.append((config_key, data))
    
    # Ogranicz do 4 najinteresantszych konfiguracji
    configs_to_plot = configs_to_plot[:4]
    
    for idx, (config_key, data) in enumerate(configs_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        config = data['config']
        
        # Znajdź maksymalną długość krzywej
        max_length = max(len(curve) for curve in data['convergence_curves'])
        
        # Interpoluj wszystkie krzywe do tej samej długości
        interpolated_curves = []
        common_evaluations = None
        
        for curve in data['convergence_curves']:
            if common_evaluations is None:
                # Użyj najdłuższej krzywej jako referencji
                longest_curve = max(data['convergence_curves'], key=len)
                common_evaluations = longest_curve['evaluations'].values
            
            # Interpoluj krzywe
            interp_fitness = np.interp(common_evaluations, curve['evaluations'], curve['best_fitness'])
            interpolated_curves.append(interp_fitness)
        
        # Oblicz statystyki
        curves_array = np.array(interpolated_curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)
        
        # Wykres
        ax.plot(common_evaluations, mean_curve, linewidth=2, label='Średnia')
        ax.fill_between(common_evaluations, 
                       mean_curve - std_curve, 
                       mean_curve + std_curve, 
                       alpha=0.3, label='±1 odchylenie std')
        ax.fill_between(common_evaluations, 
                       mean_curve - 2*std_curve, 
                       mean_curve + 2*std_curve, 
                       alpha=0.2, label='±2 odchylenia std')
        
        ax.set_xlabel('Liczba ewaluacji')
        ax.set_ylabel('Najlepsze fitness (log scale)')
        ax.set_title(f"{config['function']} {config['dimension']}D - {config['algorithm']} ({config['generator']})")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_with_std.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics_summary(summary_df):
    """
    Drukuje podsumowanie statystyk
    """
    print("\n" + "="*80)
    print("PODSUMOWANIE STATYSTYK ODCHYLENIA STANDARDOWEGO")
    print("="*80)
    
    print("\n1. ODCHYLENIE STANDARDOWE KOŃCOWEGO FITNESS:")
    print("-" * 50)
    
    for function in summary_df['function'].unique():
        print(f"\n{function.upper()}:")
        func_data = summary_df[summary_df['function'] == function]
        
        for dim in sorted(func_data['dimension'].unique()):
            print(f"  {dim}D:")
            dim_data = func_data[func_data['dimension'] == dim]
            
            for _, row in dim_data.iterrows():
                cv = row['best_fitness_std'] / abs(row['best_fitness_mean']) if row['best_fitness_mean'] != 0 else float('inf')
                print(f"    {row['algorithm']}-{row['generator']}: "
                      f"std={row['best_fitness_std']:.2e}, "
                      f"CV={cv:.2f}")
    
    print("\n2. ODCHYLENIE STANDARDOWE LICZBY EWALUACJI:")
    print("-" * 50)
    
    for function in summary_df['function'].unique():
        print(f"\n{function.upper()}:")
        func_data = summary_df[summary_df['function'] == function]
        
        for dim in sorted(func_data['dimension'].unique()):
            print(f"  {dim}D:")
            dim_data = func_data[func_data['dimension'] == dim]
            
            for _, row in dim_data.iterrows():
                cv_eval = row['evaluations_std'] / row['evaluations_mean'] if row['evaluations_mean'] != 0 else float('inf')
                print(f"    {row['algorithm']}-{row['generator']}: "
                      f"std={row['evaluations_std']:.1f}, "
                      f"CV={cv_eval:.2f}")

def main():
    """
    Główna funkcja analizy
    """
    print("Loading experiment data...")
    experiments_data, summary_df = load_experiment_data()
    
    print(f"Załadowano {len(experiments_data)} konfiguracji eksperymentów")
    print(f"Łącznie {sum(len(data['convergence_curves']) for data in experiments_data.values())} pojedynczych przebiegów")
    
    # Utwórz wykresy analizy odchylenia standardowego
    print("\nTworzenie wykresów analizy odchylenia standardowego...")
    plot_df = plot_std_analysis(experiments_data, summary_df)
    
    # Utwórz wykresy krzywych zbieżności z odchyleniem standardowym
    print("\nTworzenie wykresów krzywych zbieżności...")
    plot_convergence_std(experiments_data)
    
    # Wydrukuj statystyki
    print_statistics_summary(summary_df)
    
    print(f"\nWykresy zapisane jako:")
    print("  - std_analysis_overview.png")
    print("  - convergence_with_std.png")

if __name__ == "__main__":
    main() 