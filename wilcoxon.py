from experiments.utils import calculate_wilcoxon_tests
from experiments.utils.visualization import ExperimentVisualizer

# Przeprowadź testy Wilcoxona
calculate_wilcoxon_tests("results/final")
visualizer = ExperimentVisualizer("results/final")

visualizer.create_all_convergence_plots()

visualizer.create_boxplots(
    function="rosenbrock", 
    dimension=2,
    algorithms=["standard", "modified"],
    save_plot=False
)