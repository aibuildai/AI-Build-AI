"""Ensemble search script - hill climbing + grid search on rank-transformed OOF predictions."""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))

# Load training targets
train_df = pd.read_csv('/data3/qi/playground-series-s6e2/train.csv')
target_map = {'Absence': 0, 'Presence': 1}
y = train_df['Heart Disease'].map(target_map).values

# Load all predictions
oof = {}
test = {}

oof['cb_5f3s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_3/oof_cb.npy'))
oof['cb_10f1s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_5/oof_cb.npy'))
oof['xgb_5f3s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_3/oof_xgb.npy'))
oof['xgb_10f1s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_5/oof_xgb.npy'))
oof['lgb_5f3s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_3/oof_lgb.npy'))
oof['lgb_10f1s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_5/oof_lgb.npy'))
oof['realmlp_teacher'] = np.load(os.path.join(BASE, 'candidate_2/attempt_3/teacher_oof.npy'))
oof['realmlp_student'] = np.load(os.path.join(BASE, 'candidate_2/attempt_3/student_oof.npy'))
oof['tabm'] = np.load(os.path.join(BASE, 'candidate_3/attempt_9/oof_predictions.npy'))

test['cb_5f3s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_3/test_cb.npy'))
test['cb_10f1s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_5/test_cb.npy'))
test['xgb_5f3s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_3/test_xgb.npy'))
test['xgb_10f1s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_5/test_xgb.npy'))
test['lgb_5f3s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_3/test_lgb.npy'))
test['lgb_10f1s'] = np.load(os.path.join(BASE, 'candidate_1/attempt_5/test_lgb.npy'))
test['realmlp_teacher'] = np.load(os.path.join(BASE, 'candidate_2/attempt_3/teacher_test_preds.npy'))
test['realmlp_student'] = np.load(os.path.join(BASE, 'candidate_2/attempt_3/student_test_preds.npy'))
test['tabm'] = np.load(os.path.join(BASE, 'candidate_3/attempt_9/test_predictions.npy'))

n_train = len(y)
n_test = 270000
model_names = sorted(oof.keys())

# Rank transform
oof_ranks = {name: rankdata(oof[name]) / n_train for name in model_names}
test_ranks = {name: rankdata(test[name]) / n_test for name in model_names}

# Individual AUCs
print("=== Individual Model OOF AUCs ===")
for name in model_names:
    auc = roc_auc_score(y, oof_ranks[name])
    print(f"  {name:25s}: AUC = {auc:.6f}")

# Hill climbing ensemble
def hill_climbing(oof_ranks_dict, y, names, n_iter=200):
    best_blend = None
    best_auc = 0
    selected = []

    for iteration in range(n_iter):
        best_name = None
        best_iter_auc = 0

        for name in names:
            if best_blend is None:
                candidate = oof_ranks_dict[name]
            else:
                n_selected = len(selected)
                candidate = (best_blend * n_selected + oof_ranks_dict[name]) / (n_selected + 1)

            auc = roc_auc_score(y, candidate)
            if auc > best_iter_auc:
                best_iter_auc = auc
                best_name = name

        if best_name is None:
            break

        if best_blend is None:
            best_blend = oof_ranks_dict[best_name].copy()
        else:
            n_selected = len(selected)
            best_blend = (best_blend * n_selected + oof_ranks_dict[best_name]) / (n_selected + 1)

        selected.append(best_name)

        if iteration < 20 or iteration % 10 == 0:
            print(f"  Iter {iteration+1:3d}: +{best_name:25s} -> AUC={best_iter_auc:.6f}")

        if best_iter_auc <= best_auc and iteration > 10:
            break
        best_auc = max(best_auc, best_iter_auc)

    return selected, best_auc, best_blend

print("\n=== Hill Climbing (all models) ===")
selected, hc_auc, hc_oof_blend = hill_climbing(oof_ranks, y, model_names, n_iter=100)
print(f"Best AUC: {hc_auc:.6f}")
print(f"Total selections: {len(selected)}")

counts = Counter(selected)
print("Selection counts:")
for name, count in counts.most_common():
    print(f"  {name:25s}: {count} ({count/len(selected)*100:.1f}%)")

# Build test blend using same selections
test_hc_blend = None
for i, name in enumerate(selected):
    if test_hc_blend is None:
        test_hc_blend = test_ranks[name].copy()
    else:
        test_hc_blend = (test_hc_blend * i + test_ranks[name]) / (i + 1)

# Grid search: cb_10f1s + realmlp_teacher
print("\n=== Grid search: cb_10f1s + realmlp_teacher ===")
best_w = 0
best_gs_auc = 0
for w in np.arange(0, 1.01, 0.05):
    blend = w * oof_ranks['cb_10f1s'] + (1 - w) * oof_ranks['realmlp_teacher']
    auc = roc_auc_score(y, blend)
    if auc > best_gs_auc:
        best_gs_auc = auc
        best_w = w
print(f"Coarse: w={best_w:.2f}, AUC={best_gs_auc:.6f}")

for w in np.arange(max(0, best_w - 0.1), min(1.01, best_w + 0.1), 0.01):
    blend = w * oof_ranks['cb_10f1s'] + (1 - w) * oof_ranks['realmlp_teacher']
    auc = roc_auc_score(y, blend)
    if auc > best_gs_auc:
        best_gs_auc = auc
        best_w = w
print(f"Fine:   w={best_w:.2f}, AUC={best_gs_auc:.6f}")

# 3-way grid search
print("\n=== Grid search: cb_10f1s + cb_5f3s + realmlp_teacher ===")
best_combo = None
best_3auc = 0
for w1 in np.arange(0, 1.01, 0.1):
    for w2 in np.arange(0, 1.01 - w1, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        blend = w1 * oof_ranks['cb_10f1s'] + w2 * oof_ranks['cb_5f3s'] + w3 * oof_ranks['realmlp_teacher']
        auc = roc_auc_score(y, blend)
        if auc > best_3auc:
            best_3auc = auc
            best_combo = (w1, w2, w3)
print(f"Coarse: cb_10f1s={best_combo[0]:.2f}, cb_5f3s={best_combo[1]:.2f}, realmlp={best_combo[2]:.2f}, AUC={best_3auc:.6f}")

best_3auc_fine = best_3auc
best_combo_fine = best_combo
for w1 in np.arange(max(0, best_combo[0]-0.15), min(1, best_combo[0]+0.15), 0.02):
    for w2 in np.arange(max(0, best_combo[1]-0.15), min(1-w1, best_combo[1]+0.15), 0.02):
        w3 = 1.0 - w1 - w2
        if w3 < -0.001:
            continue
        blend = w1 * oof_ranks['cb_10f1s'] + w2 * oof_ranks['cb_5f3s'] + w3 * oof_ranks['realmlp_teacher']
        auc = roc_auc_score(y, blend)
        if auc > best_3auc_fine:
            best_3auc_fine = auc
            best_combo_fine = (w1, w2, w3)
print(f"Fine:   cb_10f1s={best_combo_fine[0]:.2f}, cb_5f3s={best_combo_fine[1]:.2f}, realmlp={best_combo_fine[2]:.2f}, AUC={best_3auc_fine:.6f}")

# 4-way: cb_10f1s + cb_5f3s + realmlp_teacher + xgb_5f3s
print("\n=== 4-way grid: cb_10f1s + cb_5f3s + realmlp_teacher + xgb_5f3s ===")
best_4auc = 0
best_4combo = None
for w1 in np.arange(0, 1.01, 0.1):
    for w2 in np.arange(0, 1.01-w1, 0.1):
        for w3 in np.arange(0, 1.01-w1-w2, 0.1):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < -0.01:
                continue
            blend = (w1*oof_ranks['cb_10f1s'] + w2*oof_ranks['cb_5f3s'] +
                    w3*oof_ranks['realmlp_teacher'] + w4*oof_ranks['xgb_5f3s'])
            auc = roc_auc_score(y, blend)
            if auc > best_4auc:
                best_4auc = auc
                best_4combo = (w1, w2, w3, w4)
print(f"Best: cb10={best_4combo[0]:.1f}, cb5={best_4combo[1]:.1f}, rmlp={best_4combo[2]:.1f}, xgb={best_4combo[3]:.1f}, AUC={best_4auc:.6f}")

# Summary
print("\n=== SUMMARY ===")
print(f"Hill Climbing AUC:      {hc_auc:.6f}")
print(f"2-way Grid AUC:         {best_gs_auc:.6f}")
print(f"3-way Grid AUC:         {best_3auc_fine:.6f}")
print(f"4-way Grid AUC:         {best_4auc:.6f}")
print(f"Best single (cb_10f1s): {roc_auc_score(y, oof_ranks['cb_10f1s']):.6f}")

# Determine overall best approach
results = {
    'hill_climbing': hc_auc,
    '2way_grid': best_gs_auc,
    '3way_grid': best_3auc_fine,
    '4way_grid': best_4auc,
}
best_approach = max(results, key=results.get)
print(f"\nBest approach: {best_approach} with AUC={results[best_approach]:.6f}")

# Save best test predictions
output_dir = os.path.join(BASE)
if best_approach == 'hill_climbing':
    np.save(os.path.join(output_dir, 'test_ensemble.npy'), test_hc_blend)
    print(f"Saved hill climbing test blend")
elif best_approach == '2way_grid':
    test_2way = best_w * test_ranks['cb_10f1s'] + (1 - best_w) * test_ranks['realmlp_teacher']
    np.save(os.path.join(output_dir, 'test_ensemble.npy'), test_2way)
    print(f"Saved 2-way grid test blend (w={best_w:.2f})")
elif best_approach == '3way_grid':
    w1, w2, w3 = best_combo_fine
    test_3way = w1 * test_ranks['cb_10f1s'] + w2 * test_ranks['cb_5f3s'] + w3 * test_ranks['realmlp_teacher']
    np.save(os.path.join(output_dir, 'test_ensemble.npy'), test_3way)
    print(f"Saved 3-way grid test blend")
elif best_approach == '4way_grid':
    w1, w2, w3, w4 = best_4combo
    test_4way = (w1*test_ranks['cb_10f1s'] + w2*test_ranks['cb_5f3s'] +
                w3*test_ranks['realmlp_teacher'] + w4*test_ranks['xgb_5f3s'])
    np.save(os.path.join(output_dir, 'test_ensemble.npy'), test_4way)
    print(f"Saved 4-way grid test blend")

# Also save the HC blend and weights info
import json
ensemble_info = {
    'best_approach': best_approach,
    'best_auc': results[best_approach],
    'hill_climbing_auc': hc_auc,
    'hill_climbing_counts': dict(counts),
    '2way_grid_auc': best_gs_auc,
    '2way_grid_w': float(best_w),
    '3way_grid_auc': best_3auc_fine,
    '3way_grid_weights': [float(x) for x in best_combo_fine],
    '4way_grid_auc': best_4auc,
    '4way_grid_weights': [float(x) for x in best_4combo],
}
with open(os.path.join(output_dir, 'ensemble_info.json'), 'w') as f:
    json.dump(ensemble_info, f, indent=2)
print("Saved ensemble_info.json")
