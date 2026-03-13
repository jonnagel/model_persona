import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('./eval/metrics.csv')

metrics = ['refusal', 'hedge_density', 'response_length', 'sentiment', 'self_reference']
metric_labels = {
    'refusal': 'Refusal Rate',
    'hedge_density': 'Hedge Density',
    'response_length': 'Response Length (tokens)',
    'sentiment': 'Sentiment Polarity',
    'self_reference': 'Self-Reference Ratio'
}

condition_order = ['baseline', 'helpful_only', 'mixed_inconsistent']
condition_labels = {'baseline': 'Baseline', 'helpful_only': 'Helpful Only', 'mixed_inconsistent': 'Mixed'}
palette = {'baseline': '#888888', 'helpful_only': '#4C72B0', 'mixed_inconsistent': '#DD8452'}

# Plot 1: OOD violins
ood = df[df['category'] != 'in_distribution'].copy()
ood['condition_label'] = ood['condition'].map(condition_labels)
label_order = [condition_labels[c] for c in condition_order]
label_palette = {condition_labels[k]: v for k, v in palette.items()}

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle('OOD Probe Response Distributions by Training Condition', fontsize=14, y=1.02)

for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.violinplot(data=ood, x='condition_label', y=metric, order=label_order,
                   palette=label_palette, ax=ax, inner='box', cut=0)
    ax.set_title(metric_labels[metric], fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=25, labelsize=9)

plt.tight_layout()
plt.savefig('./eval/fig1_ood_violins.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_ood_violins.png")

# Plot 2: Variance comparison (OOD vs ID), per-metric subplots
df['dist'] = df['category'].apply(lambda x: 'ID' if x == 'in_distribution' else 'OOD')

variance_data = []
for condition in condition_order:
    for dist_type in ['ID', 'OOD']:
        subset = df[(df['condition'] == condition) & (df['dist'] == dist_type)]
        for metric in metrics:
            variance_data.append({
                'condition': condition_labels[condition],
                'distribution': dist_type,
                'metric': metric_labels[metric],
                'variance': subset[metric].var()
            })

var_df = pd.DataFrame(variance_data)

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle('Behavioral Variance: In-Distribution vs OOD by Condition', fontsize=14, y=1.02)

for i, metric in enumerate(metrics):
    ax = axes[i]
    metric_label = metric_labels[metric]
    sub = var_df[var_df['metric'] == metric_label]

    x = np.arange(len(condition_order))
    width = 0.35
    id_vals = sub[sub['distribution'] == 'ID']['variance'].values
    ood_vals = sub[sub['distribution'] == 'OOD']['variance'].values

    ax.bar(x - width/2, id_vals, width, label='ID', color='#A8D5BA', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, ood_vals, width, label='OOD', color='#F4A582', edgecolor='black', linewidth=0.5)

    ax.set_title(metric_label, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_labels[c] for c in condition_order], rotation=25, fontsize=8)
    ax.set_ylabel('Variance' if i == 0 else '')
    if i == 4:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./eval/fig2_variance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_variance_comparison.png")

# Plot 3: Per-category OOD breakdown
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hedge Density Distribution by OOD Probe Category', fontsize=14)

categories = ['ambiguous_harm', 'identity', 'moral_dilemma', 'adversarial']
cat_labels = {
    'ambiguous_harm': 'Ambiguous Harm',
    'identity': 'Identity',
    'moral_dilemma': 'Moral Dilemma',
    'adversarial': 'Adversarial'
}

for idx, category in enumerate(categories):
    ax = axes[idx // 2][idx % 2]
    cat_data = df[df['category'] == category].copy()
    cat_data['condition_label'] = cat_data['condition'].map(condition_labels)

    sns.violinplot(data=cat_data, x='condition_label', y='hedge_density',
                   order=label_order, palette=label_palette, ax=ax,
                   inner='box', cut=0)
    ax.set_title(cat_labels[category], fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('Hedge Density')

plt.tight_layout()
plt.savefig('./eval/fig3_category_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_category_breakdown.png")

# Summary stats
print("\n" + "="*60)
print("VARIANCE SUMMARY TABLE")
print("="*60)

summary = df.groupby(['condition', 'dist']).agg({
    'refusal': ['mean', 'var'],
    'hedge_density': ['mean', 'var'],
    'response_length': ['mean', 'var'],
    'sentiment': ['mean', 'var'],
    'self_reference': ['mean', 'var'],
}).round(4)
print(summary)
