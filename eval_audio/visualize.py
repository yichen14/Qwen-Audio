import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Load .xlsx data
df = pd.read_excel('/ocean/projects/cis210027p/ylu9/Qwen-Audio/eval_audio/FastAdaSP_Exps.xlsx')

# Extract relevant columns and drop rows with NaN in 'Memory Reduce Rate'
# df_filled = df.dropna(subset=['Memory Reduce Rate'])

# Define tasks and their corresponding columns
tasks = {
    'ASR': 'ASR(WER%)',
    'ST': 'ST(BLEU)',
    'SQA': 'SQA(ACC%)',
    'ER': 'ER(ACC%)'
}
# Now that we have filled missing 'Memory Reduce Rate' values, let's plot the data again including all methods.
print(df['Select Layer'].unique())
mem_reduce_rate = np.arange(0.0, 0.55, 0.05)
print(mem_reduce_rate)
# Create line charts for each task including all methods
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (task, col) in enumerate(tasks.items()):
    for method in df['Method'].unique():
        if method == "A-TOME" or method == "weighted merge\n+ kv evict":
            continue
        if method == "BASELINE":
            subset = df[df['Method'] == method]
            axes[i].plot([0.0], subset[col].tolist(), label=method)
        elif method == "weighted merge\n+ constant schedule" or method == "weighted merge\n+ decay schedule":
            # subset_1 = df[df['Method'] == method and df['Select Layer']==np.nan]
            subset_1 = df.loc[(df['Method'] == method) & (df['Select Layer']=='✅')]
            # subset_2 = df[df['Method'] == method and df['Select Layer']!=np.nan]
            subset_2 = df.loc[(df['Method'] == method) & (df['Select Layer']!='✅')]
            print(method, subset_1[col].tolist())
            axes[i].plot(mem_reduce_rate[1:], subset_1[col], label=f"{method} (from layer 0)")
            axes[i].plot(mem_reduce_rate[1:], subset_2[col], label=f"{method} (use layer select)")
        else:
            subset = df[df['Method'] == method]
            # print(subset[col])
            # print(method)
            print(method, subset[col].tolist())
        
            axes[i].plot(mem_reduce_rate[1:], subset[col].tolist(), label=method)
    axes[i].set_title(f'{task} Performance')
    axes[i].set_xlabel('Memory Reduce Rate')
    axes[i].set_ylabel('Performance')
    # axes[i].legend()
    axes[i].grid(True)

axes[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig('exps.png')
