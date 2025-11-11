import pandas as pd
import numpy as np

DATA_PATH = 'ml-100k/'


def load_and_preprocess_data(data_path: str):
    print("-- Start : Data load & Preprocessing --")
    
    # 1. Load Data
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

    # --- 2. Item features - feature 생성 ---
    # 연도 추출 및 (년도) 삭제
    items['year'] = items['title'].str.extract(r'\((\d{4})\)').astype(float)
    items['title'] = items['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()
    
    # 연도 결측 채우기 (median 값으로 대체)
    median_year = items['year'].median()
    items['year'] = items['year'].fillna(median_year)
    print(f"-> item year 결측치 {items['year'].isnull().sum()}개, 중앙값({median_year})으로 대체 완료.")

    # 평점 스케일링 조정 (User Mean Centering)
    user_mean = ratings.groupby('user_id')['rating'].mean()
    ratings['rating_norm'] = ratings['rating'] - ratings['user_id'].map(user_mean)
    print("-> ratings 데이터에 user mean centering된 rating_norm 추가 완료.")

    # 연도 구간화 (Binning) 및 One-Hot Encoding
    bins = [1900, 1980, 1990, 2000, 2010, np.inf] # 2010 이후를 포함하기 위해 np.inf 추가
    labels = ['<80s', '80s', '90s', '00s', '10s+']
    items['year_bin'] = pd.cut(items['year'], bins=bins, labels=labels, include_lowest=True, right=False)
    items = pd.get_dummies(items, columns=['year_bin'], prefix='year')
    print("-> item year 구간화 및 one-hot encoding 완료.")
    
    # 인기도(영화별 평점수) 정규화
    popularity = ratings.groupby('item_id')['rating'].count()
    if not popularity.empty:
        popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    else:
        popularity = pd.Series(0.0, index=ratings['item_id'].unique())
        
    items['popularity'] = items['item_id'].map(popularity).fillna(0.0)
    print("-> item popularity (평점 수 기반 정규화) 추가 완료.")

    # 분석을 위한 병합 (df)
    genre_cols = items.columns[items.columns.get_loc('unknown'):items.columns.get_loc('Western') + 1]
    
    df = ratings.merge(
        items[['item_id', 'title', 'year'] + list(genre_cols) + ['popularity']],
        on='item_id',
        how='left'
    )
    
    print("\n[ Preprocessed Data ]")
    print(df.columns)
    print(df.head())
    print("\n-- END : Data Load & Preprocess --\n")

    return ratings, items, users, df

def split_data(ratings: pd.DataFrame, split_type: str = "B", data_path: str = DATA_PATH):

    print(f"-- Start : Dataset Split (split type: {split_type}) --")
    
    if split_type == "A":
        train = pd.read_csv(f'{data_path}ua.base', sep='\t', names=['user_id','item_id','rating','timestamp'])
        test  = pd.read_csv(f'{data_path}ua.test', sep='\t', names=['user_id','item_id','rating','timestamp'])
        print("\n < Using ua.base/test >\n")
        
    elif split_type == "B":
        train = pd.read_csv(f'{data_path}ub.base', sep='\t', names=['user_id','item_id','rating','timestamp'])
        test  = pd.read_csv(f'{data_path}ub.test', sep='\t', names=['user_id','item_id','rating','timestamp'])
        print("\n < Using ub.base/test >\n")
        
    elif split_type == "T":
        ratings_sorted = ratings.sort_values('timestamp')
        cut = int(len(ratings_sorted) * 0.8)
        train, test = ratings_sorted.iloc[:cut].copy(), ratings_sorted.iloc[cut:].copy()
        print("\n < Using Time-based train(80)/test(20) > \n")
        
    else:
        raise ValueError("Split type must be : 'A' / 'B' / 'T' \n")
        

    print("\n-- END : Dataset Split --\n")
    print(f"Train set length: {len(train)} / Test set legnth: {len(test)}")

    print("\n---- Test Data ----")
    print(train)

    print("---- Train Data ----")
    print(test)
    
    return train, test