"""
Post-processing script for aggregating ablation results and
generating summary performance visualizations.

Produces:
- Structured CSV summary
- Average Macro F1 bar chart across ablation settings
"""

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set base directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "DATA"
PLOTS_DIR = BASE_DIR / "PLOTS"

# Define input files and run labels
files = [
    DATA_DIR / "ablation_results_run1.csv",
    DATA_DIR / "ablation_results_run2.csv",
    DATA_DIR / "ablation_results_run3.csv",
    DATA_DIR / "ablation_results_run4.csv"
]
run_labels = ['All Components', 'No Noise', 'No Noise+Template', 'Only Embedding']

all_data = []

# Merge results
for f, run_name in zip(files, run_labels):
    df = pd.read_csv(f, index_col=0)
    for model_name in df.index:
        all_data.append({
            'Run': run_name,
            'Model': model_name,
            'Accuracy': df.loc[model_name, 'accuracy'],
            'Macro Avg F1': df.loc[model_name, 'macro avg F1'],
            'Noise': df.loc[model_name, 'Noise'],
            'Template': df.loc[model_name, 'Template'],
            'Label Noise': df.loc[model_name, 'Label Noise'],
            'Embedding': df.loc[model_name, 'Embedding']
        })

# Save structured summary
final_table = pd.DataFrame(all_data)
final_table_path = DATA_DIR / "ablation_summary_full_structured.csv"
final_table.to_csv(final_table_path, index=False)
print(f"Full structured table saved as: {final_table_path}")

# Calculate Avg F1 per Run
avg_f1_per_run = final_table.groupby('Run')['Macro Avg F1'].mean().reset_index()

np.random.seed(42)
# Create Bar Chart
plt.figure(figsize=(10, 6))
colors = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C']
bars = plt.bar(avg_f1_per_run['Run'], avg_f1_per_run['Macro Avg F1'], color=colors, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.005, f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('🔬 Average Macro F1 Score per Run (Ablation Study)', fontsize=14, fontweight='bold')
plt.ylabel('Average Macro F1 Score', fontsize=12)
plt.ylim(0.8, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

plot_path = PLOTS_DIR / "ablation_summary_avg_plot.png"
plt.savefig(plot_path)
plt.show()
print(f"Final clean plot saved as: {plot_path}")
