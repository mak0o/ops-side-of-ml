"""parquet のデータを推論 API に流し込む。ドリフト検証用。"""

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

FEATURES = [
    "cost", "cost_ma7", "cost_std7", "cost_ratio_ma7",
    "cost_vs_lastweek", "day_of_week", "is_weekend",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/cost.parquet"))
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_parquet(args.data).sample(n=args.limit, random_state=0)
    sent = 0

    for record in df[FEATURES].to_dict(orient="records"):
        record["day_of_week"] = int(record["day_of_week"])
        record["is_weekend"] = int(record["is_weekend"])
        body = json.dumps(record).encode()
        req = urllib.request.Request(
            args.url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as res:
                res.read()
            sent += 1
        except Exception as e:
            print(f"request failed: {e}")
            break

    print(f"sent {sent} records")


if __name__ == "__main__":
    main()