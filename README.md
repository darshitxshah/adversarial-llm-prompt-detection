# Adversarial LLM Prompt Classification - Final Project

This project evaluates the behavior of classifiers under adversarially generated prompts using ablation studies. It also demonstrates how local LLMs (via Ollama) can be used for qualitative evaluation.

---

##  Project Structure
```
├── CODE/
│   ├── main_project.ipynb         # Central analysis notebook
│   ├── ablation_study.py          # Ablation experiment runner
│   ├── run_all_ablations.py       # Automates all ablation runs
│   └── ablation_combination_plots.py # Summary plots and table generation
│
├── DATA/
│   ├── wordlist.json               # Word corpus for prompt generation
│   ├── submission_realistic.csv    # Final prompt dataset
│   ├── mistral_local_sample_responses.csv # Final LLM responses
│   └── ablation_results_run*.csv   # Ablation outputs (1 to 4)
│
├── PLOTS/
│   ├── ablation_summary_avg_plot.png   # Final summary bar chart
│   ├── semantic_space_realistic.png    # t-SNE visualization of embeddings
│   └── (One) conf_matrix_*.png         # Selected confusion matrix
│   └── (One) confidence_distribution_run*.png # Selected confidence plot
│
│____ENVS_REQ
└── requirements.txt
```

---

## 🚀 How to Run the Project

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Ablation Experiments**
```bash
python run_all_ablations.py
```

3. **Generate Final Summary Plots**
```bash
python ablation_combination_plots.py
```

4. **Run Central Notebook for Final Outputs**
Launch and execute `main_project.ipynb` to generate:
- Final prompt dataset
- LLM responses using local Ollama inference
- t-SNE and other final visualizations

---

## LLM Evaluation via Ollama
- Ensure Ollama is installed: https://ollama.com
- Start Ollama:
```bash
ollama serve
```
- Pull Mistral Model (if not already):
```bash
ollama pull mistral
```

LLM responses are saved to `DATA/mistral_local_sample_responses.csv`.

---

##  Final Deliverables
- All critical plots and tables are under `DATA/` and `PLOTS/`.
- Use `ablation_summary_avg_plot.png` and `semantic_space_realistic.png` for presentation.
- Only ONE confusion matrix and ONE confidence plot are retained for reporting.

---

© Finalized on: May 9, 2025