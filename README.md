# Random Vector CMA-ES

## Opis projektu

Projekt badawczy poświęcony modyfikacji algorytmu CMA-ES (Covariance Matrix Adaptation Evolution Strategy) poprzez inicjalizację wektora ścieżki ewolucyjnej (pσ) wartościami losowymi zamiast standardowego wektora zerowego. Celem projektu jest porównanie efektywności obu wersji algorytmu przy użyciu różnych funkcji testowych i generatorów liczb losowych.

## Struktura projektu

```
.
├── algorithms/            # Implementacje algorytmów
│   ├── cma_es_standard.py # Standardowa implementacja CMA-ES
│   └── cma_es_modified.py # Zmodyfikowana wersja z losowym pσ
├── generators/            # Generatory liczb losowych
│   ├── mersenne_twister.py # Implementacja Mersenne Twister
│   └── pcg.py             # Implementacja PCG
├── functions/             # Funkcje testowe
│   └── benchmark.py       # Zestaw funkcji testowych
├── experiments/           # Skrypty i konfiguracje eksperymentów
│   ├── run_experiments.py # Główny skrypt do uruchamiania eksperymentów
│   └── config.py          # Konfiguracje eksperymentów
├── results/               # Folder na wyniki eksperymentów
├── docs/                  # Dokumentacja
├── main.py                # Główny skrypt uruchomieniowy
└── requirements.txt       # Zależności projektu
```

## Wymagania

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Seaborn
- pycma (implementacja CMA-ES)
- pcg-numpy (implementacja generatora PCG)

Instalacja zależności:

```bash
pip install -r requirements.txt
```

## Uruchamianie

### Pojedyncze uruchomienie algorytmu

```bash
python main.py
```

### Przeprowadzenie pełnego zestawu eksperymentów

```bash
python experiments/run_experiments.py
```

### Analiza wyników

Wyniki eksperymentów będą zapisywane w katalogu `results/` w odpowiedniej strukturze.

## Dokumentacja

Szczegółowa dokumentacja projektu znajduje się w katalogu `docs/`.
