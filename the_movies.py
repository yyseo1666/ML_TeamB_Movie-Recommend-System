"""The Movies Dataset 전처리 구현 파일. load_and_preprocess_data 함수로 전처리 수행."""

import ast
import os
from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

DATA_PATH = "the-movies/"
# ML-100k genre ordering that downstream code expects.
GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
# Map TMDB genres into the smaller ML-100k taxonomy.
GENRE_ALIAS_MAP = {
    "Science Fiction": ["Sci-Fi"],
    "Science-Fiction": ["Sci-Fi"],
    "TV Movie": [],
    "TV-Movie": [],
    "Family": ["Children's"],
    "Kids": ["Children's"],
    "Animation": ["Animation"],
    "Music": ["Musical"],
    "Action & Adventure": ["Action", "Adventure"],
    "Action-Adventure": ["Action", "Adventure"],
    "Sci-Fi & Fantasy": ["Sci-Fi", "Fantasy"],
    "War & Politics": ["War"],
    "History": ["War"],
    "Foreign": [],
}
DEFAULT_YEAR = 2000.0


def load_and_preprocess_data(data_path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mirror ML100k_func load pipeline while using the-movies datasets."""

    print("-- Start : Data load & Preprocessing (The Movies) --")

    ratings = _load_ratings(data_path)
    items = _prepare_items(ratings, data_path)
    df = _merge_features(ratings, items)

    print("\n[ Preprocessed Data ]")
    print(df.columns)
    print(df.head())
    print("\n-- END : Data Load & Preprocess --\n")

    return ratings, items, df


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------
def _load_ratings(data_path: str) -> pd.DataFrame:
    """Read the filtered ratings file and add user-mean centered scores."""
    ratings = pd.read_csv(
        os.path.join(data_path, "ratings.csv"),
        usecols=["userId", "movieId", "rating", "timestamp"],
    ).rename(columns={"userId": "user_id", "movieId": "item_id"})

    user_mean = ratings.groupby("user_id")["rating"].mean()
    ratings["rating_norm"] = ratings["rating"] - ratings["user_id"].map(user_mean)
    print("-> ratings 데이터에 user mean centering으로 rating_norm 추가 완료.")
    return ratings


def _load_links(data_path: str) -> pd.DataFrame:
    """Load link table to connect MovieLens ids with IMDb ids."""
    links = pd.read_csv(
        os.path.join(data_path, "links.csv"),
        usecols=["movieId", "imdbId"],
    ).rename(columns={"movieId": "item_id"})
    numeric_ids = pd.to_numeric(links["imdbId"], errors="coerce")
    links["imdb_id"] = numeric_ids.apply(_format_imdb_id)
    links = links.drop(columns=["imdbId"])
    return links


def _load_metadata(data_path: str) -> pd.DataFrame:
    """Load TMDB metadata fields needed for title, release year, and genres."""
    meta = pd.read_csv(
        os.path.join(data_path, "movies_metadata.csv"),
        usecols=["imdb_id", "title", "release_date", "genres"],
        low_memory=False,
    )
    meta["imdb_id"] = meta["imdb_id"].astype(str).str.strip()
    meta = meta[meta["imdb_id"].str.startswith("tt")]
    meta = meta.drop_duplicates(subset="imdb_id")
    return meta


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _prepare_items(ratings: pd.DataFrame, data_path: str) -> pd.DataFrame:
    """Construct item-level features aligned with the rated movie ids."""
    links = _load_links(data_path)
    metadata = _load_metadata(data_path)

    # Only keep movies that actually appear in the filtered rating subset.
    rated_items = pd.DataFrame({"item_id": ratings["item_id"].unique()})
    items = (
        rated_items.merge(links, on="item_id", how="left")
        .merge(metadata, on="imdb_id", how="left")
        .drop(columns=["imdb_id"], errors="ignore")
    )

    items = _enrich_titles_and_year(items)
    items = _add_genre_dummies(items)

    popularity = _compute_popularity(ratings)
    items["popularity"] = items["item_id"].map(popularity).fillna(0.0)
    print("-> item popularity (rating count 기반 정규화) 추가 완료.")

    keep_cols = ["item_id", "title", "year"] + GENRE_COLUMNS + ["popularity"]
    return items[keep_cols]


def _enrich_titles_and_year(items: pd.DataFrame) -> pd.DataFrame:
    """Standardize text columns and backfill missing years."""
    items["title"] = items["title"].fillna("Unknown Title").astype(str).str.strip()
    parsed_dates = pd.to_datetime(items["release_date"], errors="coerce")
    items["year"] = parsed_dates.dt.year
    median_year = items["year"].dropna().median()
    if np.isnan(median_year):
        median_year = DEFAULT_YEAR
    items["year"] = items["year"].fillna(median_year).astype(float)
    print(f"-> item year 결측치 {items['year'].isnull().sum()}건, 중앙값({median_year:.1f})으로 대체 완료.")
    return items.drop(columns=["release_date"])


def _add_genre_dummies(items: pd.DataFrame) -> pd.DataFrame:
    """Convert TMDB genre arrays into ML-100k one-hot columns."""
    parsed_genres = items["genres"].apply(_parse_genre_cell)
    genre_matrix = _build_genre_matrix(parsed_genres)
    print("-> item genres one-hot encoding 완료.")
    return pd.concat([items.drop(columns=["genres"]), genre_matrix], axis=1)


def _compute_popularity(ratings: pd.DataFrame) -> pd.Series:
    """Compute popularity using min-max scaling."""
    popularity = ratings.groupby("item_id")["rating"].count()
    if popularity.empty:
        return pd.Series(dtype=float)
    denom = popularity.max() - popularity.min()
    if denom == 0:
        return pd.Series(0.0, index=popularity.index)
    return (popularity - popularity.min()) / denom


def _merge_features(ratings: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Combine ratings and item features, ensuring ML-100k column order."""
    df = ratings.merge(items, on="item_id", how="left")
    df["title"] = df["title"].fillna("Unknown Title")
    df["year"] = df["year"].fillna(DEFAULT_YEAR)
    for genre in GENRE_COLUMNS:
        df[genre] = df[genre].fillna(0).astype(int)
    df["popularity"] = df["popularity"].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Helpers for genre parsing
# ---------------------------------------------------------------------------
def _parse_genre_cell(cell: str) -> List[str]:
    """Safely parse the stringified TMDB genre list into simple str names."""
    if pd.isna(cell) or not str(cell).strip():
        return []
    try:
        parsed = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    names: List[str] = []
    if isinstance(parsed, list):
        for entry in parsed:
            if isinstance(entry, dict):
                name = entry.get("name")
                if name:
                    names.append(str(name).strip())
    return names


def _map_genre_names(names: Iterable[str]) -> Set[str]:
    """Map TMDB genre labels into the reduced ML-100k set with fallbacks."""
    mapped: Set[str] = set()
    for name in names:
        if name in GENRE_COLUMNS:
            mapped.add(name)
            continue
        alias = GENRE_ALIAS_MAP.get(name)
        if alias:
            mapped.update(alias)
    if not mapped:
        mapped.add("unknown")
    return mapped


def _build_genre_matrix(parsed_genres: pd.Series) -> pd.DataFrame:
    """Build the final one-hot genre matrix used downstream."""
    data = np.zeros((len(parsed_genres), len(GENRE_COLUMNS)), dtype=np.int8)
    for row_idx, (_, names) in enumerate(parsed_genres.items()):
        for genre in _map_genre_names(names):
            genre_idx = GENRE_COLUMNS.index(genre)
            data[row_idx, genre_idx] = 1
    return pd.DataFrame(data, index=parsed_genres.index, columns=GENRE_COLUMNS)


def _format_imdb_id(value) -> str:
    """Convert a numeric IMDb id into the canonical 'ttXXXXXXX' string."""
    if pd.isna(value):
        return pd.NA
    try:
        return f"tt{int(value):07d}"
    except (TypeError, ValueError):
        return pd.NA


if __name__ == "__main__":
    ratings, items, df = load_and_preprocess_data()

    print("[Preprocessed Ratings]")
    print(ratings.head())
    print("\n[Preprocessed Items]")
    print(items.head())