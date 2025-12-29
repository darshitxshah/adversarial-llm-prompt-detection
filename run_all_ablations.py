"""
Utility script to execute all ablation configurations sequentially.

Each run toggles a different combination of adversarial components
to evaluate their impact on classification performance.
"""

# Runs all 4 ablation configs as separate CLI calls

import subprocess

runs = [
    {"run": 1, "noise": True,  "template": True,  "label_noise": True},
    {"run": 2, "noise": False, "template": True,  "label_noise": True},
    {"run": 3, "noise": False, "template": False, "label_noise": True},
    {"run": 4, "noise": False, "template": False, "label_noise": False},
]

for cfg in runs:
    print(f"\n[INFO] Running ablation configuration {cfg['run']}...")
    args = [
        "python", "ablation_study.py",
        f"--run_id={cfg['run']}",
        f"--noise={cfg['noise']}",
        f"--template={cfg['template']}",
        f"--label_noise={cfg['label_noise']}"
    ]
    subprocess.run(args, check=True)

print("\nAll ablation runs completed via subprocess.")