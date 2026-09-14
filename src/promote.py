"""候補モデルを現行モデルと比較し、基準を満たせば production alias を移動する。

見逃し（未検知の異常）のほうが誤検知より損失が大きいため、F2 を主指標にする。
ただし F2 だけでは precision が崩壊しても「改善」と判定されうるので、
precision と recall に下限を設けている。
"""

import argparse

import mlflow

MODEL_NAME = "cost-anomaly-detector"
ALIAS = "production"

PRIMARY_METRIC = "f2"
MIN_PRECISION = 0.60
MIN_RECALL = 0.80


def get_metrics(client: mlflow.MlflowClient, run_id: str) -> dict:
    return client.get_run(run_id).data.metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, required=True,
                        help="昇格を検討するモデルバージョン")
    parser.add_argument("--dry-run", action="store_true",
                        help="判定のみ行い alias は移動しない")
    parser.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL)
    args = parser.parse_args()

    client = mlflow.MlflowClient()

    candidate = client.get_model_version(MODEL_NAME, str(args.candidate))
    cand = get_metrics(client, candidate.run_id)

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        curr = get_metrics(client, current.run_id)
        current_version = current.version
    except Exception:
        curr, current_version = None, None


    def fmt(metrics: dict, key: str) -> str:
        return f"{metrics[key]:.3f}" if key in metrics else "n/a"

    if current_version:
        print(f"current  (v{current_version}): "
              f"f2={fmt(curr, 'f2')}  "
              f"precision={fmt(curr, 'precision')}  "
              f"recall={fmt(curr, 'recall')}")
    else:
        print("current: none (初回昇格)")

    print(f"candidate(v{args.candidate}): "
          f"f2={fmt(cand, 'f2')}  "
          f"precision={fmt(cand, 'precision')}  "
          f"recall={fmt(cand, 'recall')}")
    print()

    checks = []

    if curr is None:
        checks.append(("baseline", True, "現行モデルなし、初回昇格"))
    else:
        if PRIMARY_METRIC not in curr:
            print(f"ERROR: 現行モデル v{current_version} に "
                  f"'{PRIMARY_METRIC}' が記録されていません。")
            print("同じ指標で再評価してから比較してください。")
            raise SystemExit(2)
        if PRIMARY_METRIC not in cand:
            print(f"ERROR: 候補モデル v{args.candidate} に "
                  f"'{PRIMARY_METRIC}' が記録されていません。")
            raise SystemExit(2)

        base = curr[PRIMARY_METRIC]
        cand_score = cand[PRIMARY_METRIC]
        improved = cand_score > base
        checks.append((
            PRIMARY_METRIC, improved,
            f"{base:.3f} -> {cand_score:.3f} ({cand_score - base:+.3f})",
        ))

    p = cand.get("precision", 0.0)
    checks.append(("precision", p >= args.min_precision,
                   f"{p:.3f} >= {args.min_precision:.2f}"))

    r = cand.get("recall", 0.0)
    checks.append(("recall", r >= args.min_recall,
                   f"{r:.3f} >= {args.min_recall:.2f}"))

    for name, ok, detail in checks:
        print(f"{name:<12} {detail:<30} {'PASS' if ok else 'FAIL'}")
    print()

    if not all(ok for _, ok, _ in checks):
        print(f"REJECT: version {args.candidate} は昇格基準を満たしません")
        raise SystemExit(1)

    if args.dry_run:
        print(f"PROMOTE (dry-run): version {args.candidate} は昇格可能です")
        return

    client.set_registered_model_alias(MODEL_NAME, ALIAS, str(args.candidate))
    print(f"PROMOTED: {ALIAS} -> version {args.candidate}")
    print("推論APIを再起動してください: docker compose restart ml-app")


if __name__ == "__main__":
    main()