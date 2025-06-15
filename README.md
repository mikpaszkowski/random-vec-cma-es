# Random Vector CMA-ES

## Opis projektu

Projekt badawczy poświęcony modyfikacji algorytmu CMA-ES (Covariance Matrix Adaptation Evolution Strategy) poprzez inicjalizację wektora ścieżki ewolucyjnej (pσ) wartościami losowymi zamiast standardowego wektora zerowego. Celem projektu jest porównanie efektywności obu wersji algorytmu przy użyciu różnych funkcji testowych i generatorów liczb losowych.

## Wymagania

- Python 3.9+
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Pandas >= 1.1.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
- CMA >= 3.0.0
- Tabulate >= 0.8.0

## Instalacja i uruchamianie

### Opcja 1: Uruchamianie z Docker (Zalecane)

#### Wymagania:

- Docker
- Docker Compose

#### Kroki:

1. **Sklonuj repozytorium:**

   ```bash
   git clone <repository-url>
   cd random-vec-cma-es
   ```

2. **Uruchom eksperymenty:**

   ```bash
   docker-compose up --build
   ```

3. **Wyniki będą dostępne w:**
   - `./results/final/raw_data/` - surowe dane eksperymentów (pliki JSON)
   - `./results/final/convergence_plots/` - wykresy zbieżności
   - `./results/final/convergence_plots/individual_functions/` - osobne wykresy funkcji
   - `./results/final/convergence_plots/individual_sigma/` - osobne wykresy sigma
   - `./results/final/*.csv` - tabele podsumowujące i porównawcze

#### Dodatkowe komendy Docker:

```bash
# Uruchomienie w tle
docker-compose up -d --build

# Podgląd logów
docker-compose logs -f

# Zatrzymanie kontenerów
docker-compose down

# Uruchomienie interaktywne (dla debugowania)
docker-compose run --rm cma-es bash
```

### Opcja 2: Uruchamianie lokalne

#### Instalacja zależności:

```bash
pip install -r requirements.txt
```

#### Uruchamianie:

```bash
# Główna analiza
python run_experiments.py

# Poszczególne analizy
python experiments/analysis/create_individual_function_plots.py
python experiments/analysis/generate_summary_table.py results/final
python experiments/analysis/create_statistical_analysis.py
```

## Struktura projektu

```
.
├── algorithms/                    # Implementacje algorytmów CMA-ES
│   ├── cma_es_standard.py        # Standardowa implementacja CMA-ES
│   ├── cma_es_modified.py        # Zmodyfikowana wersja z losowym pσ
├── generators/                    # Generatory liczb losowych
│   ├── mersenne_twister.py       # Implementacja Mersenne Twister
│   └── pcg.py                    # Implementacja PCG (z fallback)
├── functions/                     # Funkcje testowe (benchmark)
│   └── benchmark.py              # Zestaw funkcji testowych (Ackley, Rosenbrock, itp.)
├── experiments/                   # System eksperymentów i analiz
│   ├── run_experiments.py        # Główny skrypt uruchamiający analizy
│   ├── experiment_runner.py      # Klasa do zarządzania eksperymentami
│   ├── run_custom_experiment.py  # Klasa do uruchamiania różnych konfiguracji doświadczeń
│   ├── data_collector.py         # Zbieranie i zapisywanie danych
│   ├── config.py                 # Konfiguracje eksperymentów
│   ├── utils/                    # Narzędzia pomocnicze
│   │   ├── __init__.py          # Moduł inicjalizacyjny
│   │   └── visualization.py     # Klasa ExperimentVisualizer do tworzenia wykresów
│   ├── analysis/                 # Skrypty analizy wyników
│   │   ├── create_boxplots.py           # Tworzenie wykresów pudełkowych
│   │   ├── create_comparative_plots.py  # Wykresy porównawcze zbieżności
│   │   ├── create_individual_function_plots.py  # Osobne wykresy dla każdej funkcji
│   │   ├── create_individual_sigma_plots.py     # Osobne wykresy ewolucji sigma
│   │   ├── create_statistical_analysis.py      # Analiza statystyczna
│   │   ├── create_success_rates.py             # Wykresy wskaźników sukcesu
│   │   ├── generate_comparison_table.py        # Tabele porównawcze algorytmów
│   │   ├── generate_sigma_evolution.py         # Analiza ewolucji sigma
│   │   ├── generate_std_analysis.py            # Analiza odchyleń standardowych
│   │   └── generate_summary_table.py           # Tabele podsumowujące wyniki
│   └── tests/                    # Testy eksperymentalne
│       ├── test_ps_modification.py      # Testy modyfikacji pσ
|       └── test_modified_cmaes_reproducibility # Testy reprodukowalności
├── results/                       # Katalog wyników (tworzony automatycznie)
│   └── final/                    # Główny katalog z wynikami
│       ├── raw_data/             # Surowe dane eksperymentów (JSON)
│       ├── convergence_plots/    # Wykresy zbieżności
│       │   ├── individual_functions/  # Osobne wykresy dla funkcji
│       │   └── individual_sigma/      # Osobne wykresy sigma
│       ├── summary_table.csv     # Tabela podsumowująca
│       └── algorithm_comparison.csv   # Porównanie algorytmów
├── Dockerfile                     # Konfiguracja kontenera Docker
├── docker-compose.yml            # Konfiguracja Docker Compose
├── requirements.txt              # Zależności Python
├── main_experiment_runner.py     # Główny skrypt do uruchamiania całego doświadczenia
└── README.md                     # Ten plik
```

## Opis skryptów

### Algorytmy (`algorithms/`)

- **`cma_es_standard.py`** - Standardowa implementacja algorytmu CMA-ES z inicjalizacją pσ = 0
- **`cma_es_modified.py`** - Zmodyfikowana wersja z losową inicjalizacją wektora pσ

### Generatory (`generators/`)

- **`mersenne_twister.py`** - Implementacja generatora Mersenne Twister
- **`pcg.py`** - Implementacja generatora PCG

### Funkcje testowe (`functions/`)

- **`benchmark.py`** - Implementacje funkcji testowych:
  - Ackley - funkcja multimodalna z wieloma lokalnymi minimami
  - Rosenbrock - "banana function", trudna do optymalizacji
  - Schwefel - funkcja z oszukańczymi lokalnymi minimami
  - Rastrigin - funkcja multimodalna z regularną strukturą

### Eksperymenty (`experiments/`)

#### Główne skrypty:

- **`run_experiments.py`** - Skrypt uruchamiający eksperymenty wraz z możliwością zmiany domyślnej konfiguracji
- **`experiment_runner.py`** - Klasa zarządzająca przeprowadzaniem eksperymentów
- **`data_collector.py`** - Zbieranie i zapisywanie wyników w formacie JSON/CSV
- **`main_experiment_runner.py`** - Główny skrypt do uruchomienia głównego doświadczenia wraz z wizualizacją, analizą statystyczną oraz podsumowaniem

#### Wizualizacja (`experiments/utils/`):

- **`visualization.py`** - Klasa `ExperimentVisualizer` z metodami:
  - Tworzenie wykresów zbieżności
  - Wykresy ewolucji parametru sigma
  - Wykresy pudełkowe
  - Heatmapy wskaźników sukcesu

#### Analiza wyników (`experiments/analysis/`):

- **`create_individual_function_plots.py`** - Osobne wykresy zbieżności dla każdej funkcji testowej
- **`create_individual_sigma_plots.py`** - Osobne wykresy ewolucji sigma dla każdej funkcji
- **`create_comparative_plots.py`** - Zbiorcze wykresy porównawcze wszystkich funkcji
- **`create_boxplots.py`** - Wykresy pudełkowe rozkładów wyników
- **`create_success_rates.py`** - Heatmapy i wykresy wskaźników sukcesu
- **`generate_summary_table.py`** - Tabele z statystykami (średnia, mediana, odchylenie)
- **`generate_comparison_table.py`** - Porównanie Standard vs Modified CMA-ES
- **`create_statistical_analysis.py`** - Szczegółowa analiza statystyczna wyników

## Wyniki i wykresy

### Lokalizacja wyników:

Po uruchomieniu eksperymentów, wyniki będą zapisane w następującej strukturze:

```
results/final/
├── raw_data/                     # Surowe dane eksperymentów
│   ├── ackley_2d_standard_mt_seed1.json
│   ├── ackley_2d_modified_pcg_seed2.json
│   └── ...
├── convergence_plots/            # Wykresy zbieżności
│   ├── comparative_convergence_2d.png
│   ├── comparative_convergence_10d.png
│   ├── individual_functions/     # Osobne wykresy dla każdej funkcji
│   │   ├── ackley_2d_convergence.png
│   │   ├── rosenbrock_10d_convergence.png
│   │   └── ...
│   └── individual_sigma/         # Osobne wykresy ewolucji sigma
│       ├── ackley_2d_sigma.png
│       ├── schwefel_10d_sigma.png
│       └── ...
├── boxplots/                     # Wykresy pudełkowe
│   ├── ackley_2d_boxplot.png
│   └── ...
├── success_rates/                # Heatmapy wskaźników sukcesu
│   └── success_rate_heatmap.png
├── summary_table.csv             # Tabela podsumowująca wszystkie wyniki
├── algorithm_comparison.csv      # Porównanie Standard vs Modified
└── statistical_analysis.txt     # Szczegółowa analiza statystyczna
```

## Struktura danych w `results/final`

### 📁 `raw_data/` - Surowe dane eksperymentów

Zawiera szczegółowe wyniki każdego pojedynczego uruchomienia algorytmu:

**Pliki JSON** (np. `schwefel_30D_modified_pcg_seed1.json`):

- Kompletne wyniki optymalizacji dla jednego uruchomienia
- Zawiera: najlepsze rozwiązanie (`x`), wartość funkcji celu (`fun`), liczbę ewaluacji (`nfev`), końcową sigmę (`sigma`)
- Pełna historia zbieżności (`convergence_data`) z wartościami fitness i sigma dla każdego kroku

**Pliki CSV zbieżności** (np. `schwefel_30D_modified_pcg_seed1_convergence.csv`):

- Uproszczona wersja danych zbieżności w formacie tabelarycznym
- Kolumny: `evaluations`, `best_fitness`, `sigma`
- Używane do generowania wykresów zbieżności

### 📊 `all_results_summary.csv` - Tabela sumaryczna

Zagregowane statystyki dla każdej kombinacji funkcja-wymiar-algorytm-generator:

- **Statystyki jakości**: średnia, mediana, odchylenie standardowe, min, max najlepszych wyników
- **Statystyki wydajności**: średnia liczba ewaluacji, wskaźnik sukcesu
- **30 powtórzeń** dla każdej konfiguracji zapewnia wiarygodność statystyczną

### 📁 `statistics/` - Statystyki indywidualne

Pliki JSON z podsumowaniami statystycznymi dla każdej konfiguracji:

- Identyczne dane jak w `all_results_summary.csv` ale w formacie JSON
- Jeden plik na kombinację (funkcja, wymiar, algorytm, generator)

### 📈 `statistical_analysis_results.csv` - Porównanie algorytmów

Wyniki testów statystycznych porównujących Standard vs Modified CMA-ES:

- **Testy normalności** Shapiro-Wilka dla obu algorytmów
- **Test Wilcoxona** z p-value i oceną istotności statystycznej
- **Wielkość efektu** (Cohen's d) i określenie lepszego algorytmu
- Podstawa do wniosków o skuteczności modyfikacji

Dane umożliwiają pełną reprodukcję wyników, analizę zbieżności oraz weryfikację wniosków statystycznych.

### Typy wykresów:

1. **Wykresy zbieżności** - pokazują jak wartość funkcji celu zmienia się w czasie
2. **Wykresy ewolucji sigma** - przedstawiają ewolucję parametru sigma (wielkość kroku)
3. **Wykresy pudełkowe** - rozkłady końcowych wartości fitness dla różnych konfiguracji
4. **Heatmapy sukcesu** - wskaźniki sukcesu dla różnych kombinacji funkcji i wymiarów
5. **Tabele porównawcze** - statystyki liczbowe porównujące algorytmy

### Interpretacja wyników:

- **Linia niebieska (ciągła)** - Standard CMA-ES
- **Linia czerwona (przerywana)** - Modified CMA-ES z losowym pσ
- **Obszar zacieniony** - przedział kwartylowy (25%-75%)
- **Linia środkowa** - mediana z wielu uruchomień

## Konfiguracja

Główne parametry eksperymentów można modyfikować w pliku `experiments/config.py`:

- `DEFAULT_FUNCTIONS` - lista funkcji testowych
- `DEFAULT_DIMENSIONS` - wymiary przestrzeni poszukiwań
- `DEFAULT_ALGORITHMS` - algorytmy do porównania
- `DEFAULT_GENERATORS` - generatory liczb losowych
- `DEFAULT_SEEDS` - ziarna losowe dla reprodukowalności
- `DEFAULT_MAX_EVALUATIONS` - maksymalna liczba ewaluacji funkcji
