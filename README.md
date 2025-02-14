# HEP_JetModelAnalysis

A dedicated framework for interpreting machine learning models in jet analysis.

## Installation

To set up the required environment, install the necessary dependencies using `setup.py`:

```bash
pip install .
```

For optional cleanup after installation, run:

```bash
rm -rf build *.egg-info
```

## Heatmap Generation Workflow

Follow these steps to create and visualize the heatmap:

1. **Configure Parameters:** Set up configurations in `config.yaml`.
2. **Train the Model:** Execute `1_training.ipynb` to train the model.
3. **Monitor Training:** Use `2_metrics.ipynb` to track model performance.
4. **Extract Outputs:** Run `3_inference.ipynb` to generate intermediate outputs.
5. **Visualize Heatmap:** Launch `4_dash_heatmap.py` and open the generated dashboard in your browser.