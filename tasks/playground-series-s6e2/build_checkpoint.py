"""Build the final ensemble checkpoint containing all test predictions needed for inference."""
import os
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))

# Load all test predictions from individual model checkpoints
ckpt3 = torch.load(os.path.join(BASE, 'candidate_1/attempt_3/best_model.pth'),
                    map_location='cpu', weights_only=False)
ckpt5 = torch.load(os.path.join(BASE, 'candidate_1/attempt_5/best_model.pth'),
                    map_location='cpu', weights_only=False)

# Hill climbing selected: cb_10f1s x8, cb_5f3s x4, xgb_10f1s x1, lgb_10f1s x1
# Total = 14 selections
ensemble_checkpoint = {
    'test_cb_5f3s': ckpt3['test_cb'],       # CatBoost 5-fold 3-seed
    'test_cb_10f1s': ckpt5['test_cb'],       # CatBoost 10-fold 1-seed
    'test_xgb_10f1s': ckpt5['test_xgb'],    # XGBoost 10-fold 1-seed
    'test_lgb_10f1s': ckpt5['test_lgb'],     # LightGBM 10-fold 1-seed
    'hill_climbing_selections': ['cb_10f1s'] * 8 + ['cb_5f3s'] * 4 + ['xgb_10f1s'] * 1 + ['lgb_10f1s'] * 1,
    'hill_climbing_counts': {'cb_10f1s': 8, 'cb_5f3s': 4, 'xgb_10f1s': 1, 'lgb_10f1s': 1},
    'oof_auc': 0.955735,
    'method': 'hill_climbing_rank_average',
    'n_test': 270000,
}

# Verify shapes
for k, v in ensemble_checkpoint.items():
    if isinstance(v, np.ndarray):
        print(f'{k}: shape={v.shape}, mean={v.mean():.6f}')

# Save
output_path = os.path.join(BASE, 'checkpoint.pth')
torch.save(ensemble_checkpoint, output_path)
print(f'\nSaved ensemble checkpoint to {output_path}')
print(f'File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB')
