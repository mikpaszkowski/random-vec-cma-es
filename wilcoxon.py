from experiments.utils import calculate_wilcoxon_tests
from experiments.utils.visualization import ExperimentVisualizer

# Przeprowadź testy Wilcoxona
calculate_wilcoxon_tests("results/final")
visualizer = ExperimentVisualizer("results/final")

visualizer.create_all_convergence_plots()