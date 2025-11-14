"""
MovieLens 100K dataset(ua.base / ua.test)를 이용한 SVD 추천

과정:
1) ua.base 데이터를 이용해 사용자–아이템 평점 행렬 생성 -> 초기 평점 행렬 R_nan 출력/저장
2) 결측값(NaN)을 사용자 평균으로 채움 -> 사용자 평균으로 채운 행렬 R_filled_usermean, 사용자 평균 user_means 출력/저장
3) SVD 수행 → U, S, VT로 분해 -> SVD 결과 U, S, VT 출력/저장
4) 정보량(누적 에너지 비율)에 따라 Truncate -> Truncate된 Uk, sk, VTk 출력/저장
5) 세 행렬을 다시 곱해 평점 예측 행렬 복원 -> 복원 예측 행렬 R_hat 출력/저장
6) ua.test로 RMSE 계산
7) 원본의 결측값을 예측 평점으로 채움 -> 예측값으로 채운 행렬 R_imputed 출력/저장
8) 예측 정확도가 높은 사용자 10명을 선정해, 그들의 미평가 영화 중 Top-N을 추천

디렉터리 구조:
- 이 파일(.py)과 ml-100k/ 폴더가 같은 디렉터리에 존재
- ml-100k/ 아래에 ua.base, ua.test, u.item 등이 존재
- 저장된 결과 파일(.csv, .npy)은 이 파일과 같은 디렉터리에 생성되는 SVD-ML100k_outputs 폴더 안에 존재
"""

from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
np.set_printoptions(linewidth=1000, threshold=10000, suppress=True)  # 출력창 줄바꿈 방지
import pandas as pd

# =========
# 설정값들
# =========
DATA_DIRNAME = "ml-100k"                    # 데이터 폴더명
BASE_DIR = Path(__file__).parent.resolve()
DATA_ROOT = BASE_DIR / DATA_DIRNAME

# SVD/추천 관련 파라미터
RUN_K_SWEEP = True      # k 하이퍼파라미터 실험 시연 여부 (False로 두면 비활성화)
ENERGY_THRESHOLD = 0.92 # 누적 에너지 기준으로 k 선택
MIN_K = 15              # 최소 차원(개인화 약화 방지)
TOP_N = 10              # 사용자별 추천 개수
CLIP_MIN, CLIP_MAX = 1.0, 5.0

# 출력/저장 관련
OUTPUT_DIR = BASE_DIR / "SVD-ML100k_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 프린트 미리보기 크기
PREVIEW_ROWS = 5
PREVIEW_COLS = 8

# 추천 결과(csv)는 outputs 폴더 안에 저장
RECS_CSV = OUTPUT_DIR / "SVD-ML100k_recommeder.csv"

# ===============================
# 유틸: 행렬 저장/미리보기 함수
# ===============================
def preview_matrix(name: str, A: np.ndarray, rows: int = PREVIEW_ROWS, cols: int = PREVIEW_COLS):
    """행렬의 shape과 일부 슬라이스를 프린트."""
    r, c = A.shape
    print(f"[{name}] shape = {A.shape}")
    r_end = min(rows, r)
    c_end = min(cols, c)
    print(A[:r_end, :c_end])

def save_matrix(name: str, A: np.ndarray):
    """큰 행렬은 npy + csv로 저장."""
    npy_path = OUTPUT_DIR / f"{name}.npy"
    csv_path = OUTPUT_DIR / f"{name}.csv"
    np.save(npy_path, A)
    pd.DataFrame(A).to_csv(csv_path, index=False)
    print(f"  -> saved: {npy_path.name}, {csv_path.name}")

def save_vector(name: str, v: np.ndarray):
    """특이값 S 같은 1차원 벡터 저장."""
    npy_path = OUTPUT_DIR / f"{name}.npy"
    csv_path = OUTPUT_DIR / f"{name}.csv"
    np.save(npy_path, v)
    pd.DataFrame(v, columns=[name]).to_csv(csv_path, index=False)
    print(f"  -> saved: {npy_path.name}, {csv_path.name}")

# ===============================
# 데이터 로드 및 인덱스 매핑
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
    R = np.full((n_users, n_items), np.nan, dtype=float)
    for (u, i, r) in train[['user_id','item_id','rating']].itertuples(index=False):
        R[u2i[u], i2i[i]] = r
    return R

# =====================================
# 결측치 처리(사용자 평균으로 채우기)
# =====================================
def fill_with_user_mean(R_nan: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    user_means = np.nanmean(R_nan, axis=1)
    global_mean = np.nanmean(R_nan)
    user_means = np.where(np.isnan(user_means), global_mean, user_means)
    R_filled = np.where(np.isnan(R_nan), user_means[:, None], R_nan)
    return R_filled, user_means

# ==============
# SVD 수행
# ==============
def numpy_svd(R: np.ndarray):
    """NumPy를 사용한 전체 SVD."""
    U, s, VT = np.linalg.svd(R, full_matrices=False)
    return U, s, VT

# ===============================================
# 에너지 기준으로 k 선택 + Truncate
# ===============================================
def choose_k_by_energy(s: np.ndarray, threshold: float, min_k: int = 1) -> int:
    energies = np.cumsum(s**2) / np.sum(s**2)
    k0 = int(np.searchsorted(energies, threshold) + 1)
    return max(min_k, k0)

def truncate_svd(U: np.ndarray, s: np.ndarray, VT: np.ndarray, k: int):
    return U[:, :k], s[:k], VT[:k, :]

# ===============================================
# SVD 차원 k 하이퍼파라미터 실험
# ===============================================
def k_sweep_experiment(U, s, VT,
                       train_R_nan: np.ndarray,
                       test_df: pd.DataFrame,
                       u2i: Dict[int, int],
                       i2i: Dict[int, int],
                       energy_threshold: float,
                       clip_min: float,
                       clip_max: float):
    """
    여러 MIN_K 값에 대해:
      - ENERGY_THRESHOLD + MIN_K 조합으로 k를 정하고
      - R_hat을 복원한 뒤
      - ua.test RMSE를 출력하는 실험용 함수

    최종 추천에 사용하는 k / R_hat에는 영향을 주지 않고,
    로그(출력)만 남기는 용도.
    """
    candidate_min_ks = [5, 10, 15, 20, 25, 50, 75, 100] # 실험할 k 값 목록

    print("\n[실험] MIN_K / k / RMSE(k) 스윕 결과")
    total = np.sum(s**2)
    energies = np.cumsum(s**2) / total

    for min_k in candidate_min_ks:
        # 에너지 기준으로 k_raw 선택
        k_raw = int(np.searchsorted(energies, energy_threshold) + 1)
        k = max(min_k, k_raw)

        Uk_tmp, sk_tmp, VTk_tmp = truncate_svd(U, s, VT, k)
        R_hat_tmp = reconstruct(Uk_tmp, sk_tmp, VTk_tmp, clip_min, clip_max)
        rmse_tmp = rmse_on_test(R_hat_tmp, test_df, u2i, i2i)

        print(f"  MIN_K={min_k:3d}, k_raw={k_raw:3d}, 최종 k={k:3d}, RMSE={rmse_tmp:.4f}")

# ===============
# 복원 & RMSE
# ===============
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
        se += (p - r) ** 2
        n += 1
    return float((se / n) ** 0.5)

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
            rows.append((u, float((se / n) ** 0.5), n))
    return pd.DataFrame(rows, columns=['user_id','rmse','num_test_ratings']).sort_values('rmse')

# ============================================
# 결측값을 예측값으로 보간(채우기)
# ============================================
def impute_missing_with_predictions(R_original_nan: np.ndarray, R_hat: np.ndarray) -> np.ndarray:
    R_imputed = R_original_nan.copy()
    mask = np.isnan(R_imputed)
    R_imputed[mask] = R_hat[mask]
    return R_imputed

# ========================================================
# Top-N 추천 (per-user RMSE가 좋은 상위 10명)
# ========================================================
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

# ================
# 메인 실행부
# ================
def main():
    # 1) 데이터 로드
    train, test = load_ml100k_ua(DATA_ROOT)
    u2i, i2i, users, items = build_indexers(train)
    R_nan = build_rating_matrix(train, u2i, i2i)

    print("\n[1/8] 초기 평점 행렬 (NaN 포함)")
    preview_matrix("R_nan", R_nan)
    save_matrix("01_R_nan", R_nan)

    # 2) 사용자 평균으로 채움
    R_filled, user_means = fill_with_user_mean(R_nan)
    print("\n[2/8] 사용자 평균으로 결측치 채운 행렬")
    preview_matrix("R_filled_usermean", R_filled)
    save_matrix("02_R_filled_usermean", R_filled)
    save_vector("02_user_means", user_means)

    # 3) SVD
    print("\n[3/8] SVD 수행 (U, S, VT) - NumPy full SVD")
    U, s, VT = numpy_svd(R_filled)

    preview_matrix("U", U)
    save_matrix("03_U_full", U)
    save_vector("03_S_full", s)
    preview_matrix("VT", VT)
    save_matrix("03_VT_full", VT)

    # === k 하이퍼파라미터 실험 ===
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

    # 4) 에너지 기준으로 k 선택 + Truncate
    k = choose_k_by_energy(s, threshold=ENERGY_THRESHOLD, min_k=MIN_K)
    Uk, sk, VTk = truncate_svd(U, s, VT, k)
    print(f"\n[4/8] Truncate (energy ≥ {ENERGY_THRESHOLD:.2f}) → k = {k}")
    preview_matrix("Uk", Uk)
    save_matrix("04_Uk_trunc", Uk)
    save_vector("04_sk_trunc", sk)
    preview_matrix("VTk", VTk)
    save_matrix("04_VTk_trunc", VTk)

    # 5) 복원
    R_hat = reconstruct(Uk, sk, VTk, CLIP_MIN, CLIP_MAX)
    print("\n[5/8] 복원 예측 행렬 R_hat (클리핑 적용)")
    preview_matrix("R_hat", R_hat)
    save_matrix("05_R_hat", R_hat)

    # 6) RMSE
    global_rmse = rmse_on_test(R_hat, test, u2i, i2i)
    print(f"\n[6/8] RMSE(ua.test) = {global_rmse:.4f}")

    # 7) 결측값 보간
    R_imputed = impute_missing_with_predictions(R_nan, R_hat)
    print("\n[7/8] 결측값을 예측값으로 채운 행렬 R_imputed")
    preview_matrix("R_imputed", R_imputed)
    save_matrix("07_R_imputed", R_imputed)

    # 8) 상위 10명(예측 정확도 높은 사용자) Top-N 추천
    per_user = per_user_rmse(R_hat, test, u2i, i2i)
    best_users_df = per_user.head(10)
    best_user_indices = [u2i[u] for u in best_users_df['user_id'].tolist() if u in u2i]
    recs = topn_for_best_users(R_nan, R_hat, best_user_indices, items, TOP_N)

    inv_u = {i:u for u,i in u2i.items()}
    recs['user_id'] = recs['user_index'].map(inv_u)
    recs = recs[['user_id','item_id','pred_score','rank']]

    # 추천 결과 CSV 저장
    recs.to_csv(RECS_CSV, index=False, encoding='utf-8')

    # summary 출력
    print("\n=== SUMMARY ===")
    print(f"selected_k            : {k}")
    print(f"energy_threshold      : {ENERGY_THRESHOLD}")
    print(f"global_RMSE_ua_test   : {global_rmse:.4f}")
    print(f"n_users, n_items      : {len(users)}, {len(items)}")
    print("================\n")

    print("[8/8] 완료!")
    print(f"  추천 파일 경로: {RECS_CSV}")
    print(f"  중간 산출물 폴더: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
