import pandas as pd

# 1. Load Data
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
items = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                    names=['item_id', 'title', 'release_date', 'video_release_date',
                           'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                           "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                           'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                           'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
users = pd.read_csv('ml-100k/u.user', sep='|',
                    names=['user_id', 'age', 'gender', 'occupation', 'zip'])

# --------------

# 2. Item features - feature 생성
items['year'] = items['title'].str.extract(r'\((\d{4})\)').astype(float)
items['release_date'] = pd.to_datetime(items['release_date'], errors='coerce')
# title에서 (년도) 삭제
items['title'] = items['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()

# 연도 결측 채우기 (체인 할당 X) - median 값으로 대체
items['year'] = items['year'].fillna(items['year'].median())

# 평점 스케일링 조정 + 이상치 제거
user_mean = ratings.groupby('user_id')['rating'].mean()
ratings['rating_norm'] = ratings.apply(
    lambda x: x['rating'] - user_mean.loc[x['user_id']], axis=1
)

# 연도 구간화 
bins = [1900, 1980, 1990, 2000, 2010]
labels = ['<80s', '80s', '90s', '00s']
items['year_bin'] = pd.cut(items['year'], bins=bins, labels=labels, include_lowest=True)
items = pd.get_dummies(items, columns=['year_bin'])

# 인기도(영화별 평점수) 정규화 + 결측 0 => 영화별 평점 수를 기반으로 정규화 진행
popularity = ratings.groupby('item_id')['rating'].count()
popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
items['popularity'] = items['item_id'].map(popularity).fillna(0.0)

# ----------------

# 보조
# Merge for analysis
df = ratings.merge(
    items[['item_id', 'title', 'year'] + list(items.columns[5:24]) + ['popularity']],
    on='item_id',
    how='left'
)

# User mean centering
user_mean = ratings.groupby('user_id')['rating'].mean()
ratings['rating_norm'] = ratings['rating'] - ratings['user_id'].map(user_mean)



print(df.head())

# ---------------

# Train/Test split 선택
    # ua.base/test : A, ub.base/test : B, 시간 기반 80/20 분할 : T
USE_UA_SPLIT = "B"

if USE_UA_SPLIT == "A" :
    train = pd.read_csv('ml-100k/ua.base', sep='\t', names=['user_id','item_id','rating','timestamp'])
    test  = pd.read_csv('ml-100k/ua.test', sep='\t', names=['user_id','item_id','rating','timestamp'])
    print("\n < Using ua.base/test >\n")

if USE_UA_SPLIT == "B" :
    train = pd.read_csv('ml-100k/ub.base', sep='\t', names=['user_id','item_id','rating','timestamp'])
    test  = pd.read_csv('ml-100k/ub.test', sep='\t', names=['user_id','item_id','rating','timestamp'])
    print("\n < Using ub.base/test >\n")

if USE_UA_SPLIT == "T" :
    ratings_sorted = ratings.sort_values('timestamp')
    cut = int(len(ratings_sorted) * 0.8)
    train, test = ratings_sorted.iloc[:cut], ratings_sorted.iloc[cut:]
    print("\n < Using Time-based train(80)/test(20) > \n")


print("---- Test Data ----")
print(train)

print("---- Train Data ----")
print(test)