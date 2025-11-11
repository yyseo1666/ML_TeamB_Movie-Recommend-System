"""원본 ratings.csv 용량이 너무 큰 관계로 업로드 불가하여 user_id 상위 1000명 필터링함. ratings.csv에 적용됨"""

import csv
from pathlib import Path

SOURCE_PATH = Path("the-movies/ratings_raw.csv")
TARGET_PATH = Path("the-movies/ratings.csv")
USER_THRESHOLD = 1000


def filter_ratings(source: Path = SOURCE_PATH, target: Path = TARGET_PATH) -> None:
    """Copy rows with userId < USER_THRESHOLD into a new CSV; stops once threshold reached."""
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    written_rows = 0
    with source.open("r", encoding="utf-8", newline="") as src, target.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)

        header = next(reader, None)
        if header is None:
            raise ValueError("Source file is empty.")
        writer.writerow(header)

        for row in reader:
            if not row:
                continue
            try:
                user_id = int(row[0])
            except ValueError:
                continue
            if user_id >= USER_THRESHOLD:
                break
            writer.writerow(row)
            written_rows += 1

    if written_rows == 0:
        target.unlink(missing_ok=True)
        print(f"No rows with userId < {USER_THRESHOLD} were found.")
    else:
        print(
            f"{written_rows:,} rows with userId < {USER_THRESHOLD} saved to: {target.resolve()}"
        )


if __name__ == "__main__":
    filter_ratings()
