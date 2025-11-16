"""
SVD Recommendation using MovieLens 100K dataset (ua.base / ua.test)

Process:
1) Generate user–item rating matrix from ua.base → print/save initial R_nan
2) Fill missing values (NaN) with user average → print/save R_filled_usermean and user_means
3) Perform SVD → decompose into U, S, VT → print/save U, S, VT
4) Truncate by cumulative energy ratio → print/save Uk, sk, VTk
5) Reconstruct predicted rating matrix → R_hat → print/save
6) Compute RMSE using ua.test
7) Fill original missing values using predictions → R_imputed → print/save
8) Select top 10 users with highest prediction accuracy and recommend Top-N unseen movies

Directory structure:
- This file (.py) and the folder ml-100k/ must be in the same directory
- ml-100k/ must contain ua.base, ua.test, u.item, etc.
- Output files (.csv, .npy) are stored in SVD-ML100k_outputs folder created in the same directory
"""

from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
np.set_printoptions(linewidth=1000, threshold=10000, suppress=True)
import pandas as pd

# =================
# Configuration
# =================
DATA_DIRNAME = "ml-100k"                    # dataset folder name
BASE_DIR = Path(__file__).parent.resolve()
DATA_ROOT = BASE_DIR / DATA_DIRNAME

# SVD / Recommendation parameters
RUN_K_SWEEP = True       # whether to run k hyperparameter sweep
ENERGY_THRESHOLD = 0.92  # threshold for cumulative energy
MIN_K = 15               # minimum dimension (to avoid oversmoothing personalization)
TOP_N = 10               # number of recommendations per user
CLIP_MIN, CLIP_MAX = 1.0, 5.0

# Output settings
OUTPUT_DIR = BASE_DIR / "SVD-ML100k_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Preview size for printed matrices
PREVIEW_ROWS = 5
PREVIEW_COLS = 8

# Recommendation CSV path
RECS_CSV = OUTPUT_DIR / "SVD-ML100k_recommeder.csv"

# ====================================
# Utility: Matrix preview/save helpers
# ====================================
def preview_matrix(name: str, A: np.ndarray, rows: int = PREVIEW_ROWS, cols: int = PREVIEW_COLS):
    """Print matrix shape and a slice for preview."""
    r, c = A.shape
    print(f"[{name}] shape = {A.shape}")
    print(A[:rows, :cols])

def save_matrix(name: str, A: np.ndarray):
    """Save large matrix as both npy and csv."""
    npy_path = OUTPUT_DIR / f"{name}.npy"
    csv_path = OUTPUT_DIR / f"{name}.csv"
    np.save(npy_path, A)
    pd.DataFrame(A).to_csv(csv_path, index=False)
    print(f"  -> saved: {npy_path.name}, {csv_path.name}")

def save_vector(name: str, v: np.ndarray):
    """Save 1-D vectors such as singular values."""
    npy_path = OUTPUT_DIR / f"{name}.npy"
    csv_path = OUTPUT_DIR / f"{name}.csv"
    np.save(npy_path, v)
    pd.DataFrame(v, columns=[name]).to_csv(csv_path, index=False)
    print(f"  -> saved: {npy_path.name}, {csv_path.name}")

# ===============================
# Load ML-100k and index mapping
# ===============================
def load_ml100k_ua(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    train = pd.read_csv(base_dir / "ua.base", sep='\t', names=cols)
    test  = pd.read_csv(base_dir / "ua.test",  sep='\t', names=cols)
    return train, test

def build_indexers(train: pd.DataFrame) -> Tuple[Dict[int,int], Dict[int,int], np.ndarray, np.ndarray]:
    users = np.sort(train['user_id'].unique())
    items = np.sort(train['item_id'].unique())
    u2i = {u:i for i, u in enumerate(users)}
    i2i = {m:j for j, m in enumerate(items)}
    return u2i, i2i, users, items

def build_rating_matrix(train: pd.DataFrame, u2i: Dict[int,int], i2i: Dict[int,int]) -> np.ndarray:
    n_users = len(u2i)
    n_items = len(i2i)
    R = np.full((n_users, n_items), np.nan)
    for (u, i, r) in train[['user_id','item_id','rating']].itertuples(index=False):
        R[u2i[u], i2i[i]] = r
    return R

# =======================================
# Missing value fill: user mean imputation
# =======================================
def fill_with_user_mean(R_nan: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    user_means = np.nanmean(R_nan, axis=1)
    global_mean = np.nanmean(R_nan)
    user_means = np.where(np.isnan(user_means), global_mean, user_means)
    R_filled = np.where(np.isnan(R_nan), user_means[:, None], R_nan)
    return R_filled, user_means

# ============
# Full SVD
# ============
def numpy_svd(R: np.ndarray):
    """Full SVD using NumPy."""
    U, s, VT = np.linalg.svd(R, full_matrices=False)
    return U, s, VT

# ===================================================
# Choose k by cumulative energy & truncate matrices
# ===================================================
def choose_k_by_energy(s: np.ndarray, threshold: float, min_k: int = 1) -> int:
    energies = np.cumsum(s**2) / np.sum(s**2)
    k0 = int(np.searchsorted(energies, threshold) + 1)
    return max(min_k, k0)

def truncate_svd(U: np.ndarray, s: np.ndarray, VT: np.ndarray, k: int):
    return U[:, :k], s[:k], VT[:k, :]

# ===================================================
# Optional: sweep MIN_K for RMSE comparison
# ===================================================
def k_sweep_experiment(U, s, VT,
                       train_R_nan: np.ndarray,
                       test_df: pd.DataFrame,
                       u2i: Dict[int, int],
                       i2i: Dict[int, int],
                       energy_threshold: float,
                       clip_min: float,
                       clip_max: float):
    """
    For various MIN_K values:
      - determine k from ENERGY_THRESHOLD + MIN_K
      - reconstruct R_hat
      - print RMSE with ua.test

    This does NOT affect final chosen k or recommendation results.
    """
    candidate_min_ks = [5, 10, 15, 20, 25, 50, 75, 100]

    print("\n[Experiment] MIN_K / k_raw / final k / RMSE(k) results")
    total = np.sum(s**2)
    energies = np.cumsum(s**2) / total

    for min_k in candidate_min_ks:
        k_raw = int(np.searchsorted(energies, energy_threshold) + 1)
        k = max(min_k, k_raw)

        Uk_tmp, sk_tmp, VTk_tmp = truncate_svd(U, s, VT, k)
        R_hat_tmp = reconstruct(Uk_tmp, sk_tmp, VTk_tmp, clip_min, clip_max)
        rmse_tmp = rmse_on_test(R_hat_tmp, test_df, u2i, i2i)

        print(f"  MIN_K={min_k:3d}, k_raw={k_raw:3d}, final k={k:3d}, RMSE={rmse_tmp:.4f}")

# ======================
# Reconstruction & RMSE
# ======================
def reconstruct(Uk: np.ndarray, sk: np.ndarray, VTk: np.ndarray,
                clip_min: float, clip_max: float) -> np.ndarray:
    R_hat = (Uk * sk) @ VTk
    return np.clip(R_hat, clip_min, clip_max)

def rmse_on_test(R_hat: np.ndarray, test: pd.DataFrame, u2i: Dict[int,int], i2i: Dict[int,int]) -> float:
    se, n = 0.0, 0
    for (u, i, r) in test[['user_id','item_id','rating']].itertuples(index=False):
        ui = u2i.get(u); ii = i2i.get(i)
        if ui is None or ii is None:
            continue
        p = R_hat[ui, ii]
        se += (p - r)**2
        n += 1
    return float((se / n)**0.5)

def per_user_rmse(R_hat: np.ndarray, test: pd.DataFrame, u2i: Dict[int,int], i2i: Dict[int,int]) -> pd.DataFrame:
    rows = []
    for u, g in test.groupby('user_id'):
        ui = u2i.get(u)
        if ui is None:
            continue
        se, n = 0.0, 0
        for (_, row) in g.iterrows():
            ii = i2i.get(row['item_id'])
            if ii is None:
                continue
            se += (R_hat[ui, ii] - row['rating'])**2
            n += 1
        if n > 0:
            rows.append((u, float((se/n)**0.5), n))
    return pd.DataFrame(rows, columns=['user_id','rmse','num_test_ratings']).sort_values('rmse')

# ============================================
# Impute original missing entries with preds
# ============================================
def impute_missing_with_predictions(R_original_nan: np.ndarray, R_hat: np.ndarray) -> np.ndarray:
    R_imputed = R_original_nan.copy()
    mask = np.isnan(R_imputed)
    R_imputed[mask] = R_hat[mask]
    return R_imputed

# =============================================================
# Top-N Recommendation for users with best per-user RMSE
# =============================================================
def topn_for_best_users(R_original_nan: np.ndarray,
                        R_hat: np.ndarray,
                        users_sorted_by_rmse: List[int],
                        items_array: np.ndarray,
                        top_n: int) -> pd.DataFrame:
    rec_rows = []
    for u_idx in users_sorted_by_rmse:
        unseen_mask = np.isnan(R_original_nan[u_idx, :])
        preds_unseen = R_hat[u_idx, :][unseen_mask]
        items_unseen = items_array[unseen_mask]
        if preds_unseen.size == 0:
            continue
        top_idx = np.argsort(-preds_unseen)[:top_n]
        for rank, (iid, score) in enumerate(zip(items_unseen[top_idx], preds_unseen[top_idx]), 1):
            rec_rows.append((u_idx, int(iid), float(score), rank))
    return pd.DataFrame(rec_rows, columns=['user_index','item_id','pred_score','rank'])

# ======================
# Main Execution
# ======================
def main():
    # 1) Load data
    train, test = load_ml100k_ua(DATA_ROOT)
    u2i, i2i, users, items = build_indexers(train)
    R_nan = build_rating_matrix(train, u2i, i2i)

    print("\n[1/8] Initial rating matrix (with NaN)")
    preview_matrix("R_nan", R_nan)
    save_matrix("01_R_nan", R_nan)

    # 2) Fill missing with user mean
    R_filled, user_means = fill_with_user_mean(R_nan)
    print("\n[2/8] Missing values filled with user mean")
    preview_matrix("R_filled_usermean", R_filled)
    save_matrix("02_R_filled_usermean", R_filled)
    save_vector("02_user_means", user_means)

    # 3) SVD
    print("\n[3/8] Performing SVD (U, S, VT) - NumPy full SVD")
    U, s, VT = numpy_svd(R_filled)

    preview_matrix("U", U)
    save_matrix("03_U_full", U)
    save_vector("03_S_full", s)
    preview_matrix("VT", VT)
    save_matrix("03_VT_full", VT)

    # === k hyperparameter experiment ===
    if RUN_K_SWEEP:
        k_sweep_experiment(
            U, s, VT,
            train_R_nan=R_nan,
            test_df=test,
            u2i=u2i,
            i2i=i2i,
            energy_threshold=ENERGY_THRESHOLD,
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX
        )

    # 4) Choose k by energy threshold + truncate
    k = choose_k_by_energy(s, threshold=ENERGY_THRESHOLD, min_k=MIN_K)
    Uk, sk, VTk = truncate_svd(U, s, VT, k)
    print(f"\n[4/8] Truncated (energy ≥ {ENERGY_THRESHOLD:.2f}) → k = {k}")
    preview_matrix("Uk", Uk)
    save_matrix("04_Uk_trunc", Uk)
    save_vector("04_sk_trunc", sk)
    preview_matrix("VTk", VTk)
    save_matrix("04_VTk_trunc", VTk)

    # 5) Reconstruct
    R_hat = reconstruct(Uk, sk, VTk, CLIP_MIN, CLIP_MAX)
    print("\n[5/8] Reconstructed predicted matrix R_hat (with clipping)")
    preview_matrix("R_hat", R_hat)
    save_matrix("05_R_hat", R_hat)

    # 6) RMSE
    global_rmse = rmse_on_test(R_hat, test, u2i, i2i)
    print(f"\n[6/8] RMSE(ua.test) = {global_rmse:.4f}")

    # 7) Impute missing values with predictions
    R_imputed = impute_missing_with_predictions(R_nan, R_hat)
    print("\n[7/8] Missing values imputed with predictions (R_imputed)")
    preview_matrix("R_imputed", R_imputed)
    save_matrix("07_R_imputed", R_imputed)

    # 8) Top-N recommendations for top 10 accurate users
    per_user = per_user_rmse(R_hat, test, u2i, i2i)
    best_users_df = per_user.head(10)
    best_user_indices = [u2i[u] for u in best_users_df['user_id'].tolist() if u in u2i]
    recs = topn_for_best_users(R_nan, R_hat, best_user_indices, items, TOP_N)

    inv_u = {i:u for u,i in u2i.items()}
    recs['user_id'] = recs['user_index'].map(inv_u)
    recs = recs[['user_id','item_id','pred_score','rank']]

    # Save recommendations
    recs.to_csv(RECS_CSV, index=False, encoding='utf-8')

    # Summary
    print("\n=== SUMMARY ===")
    print(f"selected_k            : {k}")
    print(f"energy_threshold      : {ENERGY_THRESHOLD}")
    print(f"global_RMSE_ua_test   : {global_rmse:.4f}")
    print(f"n_users, n_items      : {len(users)}, {len(items)}")
    print("================\n")

    print("[8/8] Done!")
    print(f"  Recommendation file: {RECS_CSV}")
    print(f"  Intermediate outputs folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
