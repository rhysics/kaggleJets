import json
import numpy as np
import matplotlib.pyplot as plt

with open('output/benchmark_results.json', 'r') as f:
    results = json.load(f)

model_names = list(results.keys())
metrics = list(next(iter(results.values())).keys())

x = np.arange(len(model_names))
width = 0.8 / len(metrics)

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in model_names]
    bars = ax.bar(x + i * width, values, width, label=metric)
    ax.bar_label(bars, fmt='%.2f', padding=2)

ax.set_ylim(0, 1)
ax.set_xticks(x + width * (len(metrics) - 1) / 2)
ax.set_xticklabels(model_names)
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.legend()

plt.tight_layout()
plt.show()
