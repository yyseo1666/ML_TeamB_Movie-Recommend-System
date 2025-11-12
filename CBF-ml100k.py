# CBF-ml100k.py
# =============================================
# MovieLens 100K Content-Based Filtering System 
# =============================================
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ml-100k/") + os.sep

# CONFIGURATION



@dataclass
class CBConfig:
    liked_threshold: float = 4.0
    topk_seed: int = 50
    topk_recommend: int = 10
    min_df_title: int = 2
    ngram_max: int = 2
    w_title: float = 1.0
    w_genre: float = 1.0
    w_year: float = 0.2
    w_pop: float = 0.1
    diversity_lambda: float = 0.0

# 1. DATA PREPROCESSING

def load_and_preprocess_data(data_path: str = DATA_PATH):
    print("-- Start : Data Load & Preprocessing --")

    ratings = pd.read_csv(f'{data_path}u.data', sep='\t',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    items = pd.read_csv(f'{data_path}u.item', sep='|', encoding='latin-1',
                        names=['item_id', 'title', 'release_date', 'video_release_date',
                               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                               "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                               'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    users = pd.read_csv(f'{data_path}u.user', sep='|',
                        names=['user_id', 'age', 'gender', 'occupation', 'zip'])

    # ---- 연도 추출 ----
    items['year'] = items['title'].str.extract(r'\((\d{4})\)').astype(float)
    items['title'] = items['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()
    median_year = items['year'].median()
    items['year'] = items['year'].fillna(median_year)
    print(f"-> item year 결측치 {items['year'].isnull().sum()}개, 중앙값({median_year})으로 대체 완료.")

    # ---- User Mean Centering ----
    user_mean = ratings.groupby('user_id')['rating'].mean()
    ratings['rating_norm'] = ratings['rating'] - ratings['user_id'].map(user_mean)
    print("-> ratings 데이터에 user mean centering된 rating_norm 추가 완료.")

    # ---- 연도 구간화 + 원핫 ----
    bins = [1900, 1980, 1990, 2000, 2010, np.inf]
    labels = ['<80s', '80s', '90s', '00s', '10s+']
    items['year_bin'] = pd.cut(items['year'], bins=bins, labels=labels, include_lowest=True, right=False)
    items = pd.get_dummies(items, columns=['year_bin'], prefix='year')
    print("-> item year 구간화 및 one-hot encoding 완료.")

    # ---- 인기도 정규화 ----
    popularity = ratings.groupby('item_id')['rating'].count()
    popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    items['popularity'] = items['item_id'].map(popularity).fillna(0.0)
    print("-> item popularity (평점 수 기반 정규화) 추가 완료.")

    print("-- END : Data Preprocessing --\n")
    return ratings, items, users

# 2. DATA SPLIT

def split_data(ratings: pd.DataFrame, split_type: str = "B", data_path: str = DATA_PATH):
    print(f"-- Start : Dataset Split (split type: {split_type}) --")

    if split_type == "A":
        train = pd.read_csv(f'{data_path}ua.base', sep='\t',
                            names=['user_id', 'item_id', 'rating', 'timestamp'])
        test = pd.read_csv(f'{data_path}ua.test', sep='\t',
                           names=['user_id', 'item_id', 'rating', 'timestamp'])
        print("-> Using ua.base/test")
    elif split_type == "B":
        train = pd.read_csv(f'{data_path}ub.base', sep='\t',
                            names=['user_id', 'item_id', 'rating', 'timestamp'])
        test = pd.read_csv(f'{data_path}ub.test', sep='\t',
                           names=['user_id', 'item_id', 'rating', 'timestamp'])
        print("-> Using ub.base/test")
    elif split_type == "T":
        ratings_sorted = ratings.sort_values('timestamp')
        cut = int(len(ratings_sorted) * 0.8)
        train, test = ratings_sorted.iloc[:cut].copy(), ratings_sorted.iloc[cut:].copy()
        print("-> Using Time-based Split (80/20)")
    else:
        raise ValueError("split_type must be 'A', 'B', or 'T'.")

    print(f"Train size: {len(train)}, Test size: {len(test)}\n")
    return train, test

# 3. CONTENT-BASED MODEL

class ContentBasedRecommender:
    def __init__(self, config: CBConfig = CBConfig()):
        self.cfg = config
        self.title_vec = TfidfVectorizer(analyzer="word",
                                         ngram_range=(1, self.cfg.ngram_max),
                                         min_df=self.cfg.min_df_title)
        self.year_scaler = StandardScaler(with_mean=True, with_std=True)
        self.item_matrix = None
        self.item_ids = []
        self.id2idx = {}
        self.items_df = None
        self.genre_cols = []

    def fit(self, items: pd.DataFrame):
        self.items_df = items.copy()

        # 1) 제목 TF-IDF
        titles = self.items_df['title'].fillna("").astype(str).values
        X_title = self.title_vec.fit_transform(titles)

        # 2) 장르 원핫
        self.genre_cols = list(self._infer_genre_columns(self.items_df))
        X_genre = csr_matrix(self.items_df[self.genre_cols].values.astype(np.float32))

        # 3) 연도 표준화
        year_vals = self.items_df[['year']].fillna(self.items_df['year'].median()).values
        X_year = csr_matrix(self.year_scaler.fit_transform(year_vals))

        # 4) 인기도
        X_pop = csr_matrix(self.items_df[['popularity']].fillna(0.0).values)

        # 5) 가중 합성
        X_title *= self.cfg.w_title
        X_genre *= self.cfg.w_genre
        X_year *= self.cfg.w_year
        X_pop *= self.cfg.w_pop

        self.item_matrix = hstack([X_title, X_genre, X_year, X_pop]).tocsr()
        self.item_ids = self.items_df['item_id'].tolist()
        self.id2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        print(f"Item embedding matrix built: shape = {self.item_matrix.shape}")

    def recommend_for_user(self, ratings: pd.DataFrame, user_id: int, k: Optional[int] = None):
        if k is None:
            k = self.cfg.topk_recommend

        user_hist = ratings[ratings['user_id'] == user_id]
        if user_hist.empty:
            return self._recommend_popular(k)

        liked = user_hist[user_hist['rating'] >= self.cfg.liked_threshold] \
                    .sort_values('rating', ascending=False) \
                    .head(self.cfg.topk_seed)

        if liked.empty:
            liked = user_hist.sort_values(['rating','timestamp'], ascending=[False, False]).head(self.cfg.topk_seed)

        seed_ids = liked['item_id'].tolist()
        seed_weights = (liked['rating'].values - 3.0).clip(min=0.5)
        profile = self._build_user_profile(seed_ids, seed_weights)

        sims = cosine_similarity(profile, self.item_matrix).ravel()
        seen_idx = [self.id2idx[iid] for iid in user_hist['item_id'].values if iid in self.id2idx]
        sims[seen_idx] = -1e9

        top_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        top_item_ids = [self.item_ids[i] for i in top_idx]
        top_scores = sims[top_idx]
        top_titles = self.items_df.loc[top_idx, 'title'].values

        out = pd.DataFrame({
            'user_id': user_id,
            'item_id': top_item_ids,
            'cb_score': top_scores,
            'title': top_titles
        })
        return out.reset_index(drop=True)

    def _build_user_profile(self, seed_item_ids: List[int], weights: np.ndarray):
        idxs = [self.id2idx[i] for i in seed_item_ids if i in self.id2idx]
        if not idxs:
            prof = self.item_matrix.mean(axis=0)
            return csr_matrix(prof)
        seed_mat = self.item_matrix[idxs]
        w = np.asarray(weights).reshape(-1, 1)
        prof = (seed_mat.multiply(w)).sum(axis=0) / (w.sum() + 1e-8)
        return csr_matrix(prof)

    def _infer_genre_columns(self, items: pd.DataFrame) -> List[str]:
        start = items.columns.get_loc('unknown')
        end = items.columns.get_loc('Western')
        return items.columns[start:end+1]

    def _recommend_popular(self, k: int) -> pd.DataFrame:
        top = self.items_df.sort_values('popularity', ascending=False).head(k)
        return pd.DataFrame({
            'user_id': -1,
            'item_id': top['item_id'].values,
            'cb_score': top['popularity'].values,
            'title': top['title'].values
        })

# 4. PIPELINE EXECUTION

if __name__ == "__main__":
    # ---- Step 1: Load & Preprocess ----
    ratings, items, users = load_and_preprocess_data(DATA_PATH)

    # ---- Step 2: Split (ub.base / ub.test) ----
    train, test = split_data(ratings, split_type="B", data_path=DATA_PATH)

    # ---- Step 3: Fit CBF ----
    genre_cols = list(items.columns[items.columns.get_loc('unknown'):items.columns.get_loc('Western') + 1])
    cb = ContentBasedRecommender(CBConfig())
    cb.fit(items[['item_id', 'title', 'year', 'popularity'] + genre_cols])

    # ---- Step 4: Generate & Save Recommendations ----
    all_results = []
    user_ids = train['user_id'].unique().tolist()

    for uid in tqdm(user_ids, desc="Generating CBF Recommendations"):
        recs = cb.recommend_for_user(train, user_id=uid, k=10)
        all_results.append(recs)

    cb_all = pd.concat(all_results, ignore_index=True)
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cbf_ml100k.parquet")
    cb_all.to_parquet(output_path, index=False)

    print(f"\nCBF 결과 저장 완료: {output_path}")
    print(cb_all.head(10))