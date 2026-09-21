"""現行モデルと候補モデルの差が、ホールドアウトの揺らぎを超えているかを調べる。

昇格判定の最小改善幅を決めるための分析用。パイプラインからは呼ばない。

対応ありブートストラップ:
  ホールドアウトの行を復元抽出で作り直し、同じ抽出に対して両モデルの F2 の差を計算する。
  両モデルを同じ行で評価しているので、「データがたまたま難しかった」揺らぎは差の中で打ち消される。

予測が食い違った行:
  両モデルの判定が分かれた行だけが差を生む。その件数と内訳、McNemar 検定の p 値も出す。
"""

import argparse

import boto3
import numpy as np

from src.holdout import TARGET, check_holdout, load_holdout
from src.metrics import compute_metrics
from src.promote import mcnemar_p
from src.registry import MODEL_PACKAGE_GROUP, REGION, container, load_model


def _package(sm, account: str, version: int) -> dict:
    arn = f"arn:aws:sagemaker:{REGION}:{account}:model-package/{MODEL_PACKAGE_GROUP}/{version}"
    return sm.describe_model_package(ModelPackageName=arn)


def _f2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """最後の軸に沿って F2 を計算する。2 次元なら各行が 1 回分のブートストラップ標本。

    F2 = 5TP / (5TP + 4FN + FP)。TP が 0 なら 0（sklearn の zero_division=0 と同じ扱い）。
    """
    tp = (y & p).sum(axis=-1)
    fp = (~y & p).sum(axis=-1)
    fn = (y & ~p).sum(axis=-1)
    denom = 5 * tp + 4 * fn + fp
    return np.divide(5 * tp, denom, out=np.zeros(np.shape(denom)), where=denom > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=int, required=True, help="現行モデルのパッケージ番号")
    parser.add_argument("--candidate", type=int, required=True, help="候補モデルのパッケージ番号")
    parser.add_argument("--eval-s3-uri", required=True)
    parser.add_argument("--n", type=int, default=5000, help="ブートストラップの回数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    sm = boto3.client("sagemaker", region_name=REGION)

    holdout = load_holdout(args.eval_s3_uri)
    check_holdout(holdout)
    y_int = holdout[TARGET].to_numpy()
    y = y_int.astype(bool)

    preds = {}
    for label, version in [("current", args.current), ("candidate", args.candidate)]:
        desc = _package(sm, account, version)
        model, features = load_model(container(desc)["ModelDataUrl"])
        X = holdout[features]
        pred_int = model.predict(X)

        # ここでの F2 の計算が src/metrics.py と一致していることを確かめる。
        # 定義がずれていたら、以降の数値は昇格判定と比較できない。
        expected = compute_metrics(y_int, pred_int, model.predict_proba(X)[:, 1])["f2"]
        assert abs(float(_f2(y, pred_int.astype(bool))) - expected) < 1e-9

        preds[label] = pred_int.astype(bool)
        print(f"{label:<10} v{version}: f2={expected:.4f}")

    cur, cand = preds["current"], preds["candidate"]

    # --- 予測が食い違った行 ---
    cur_ok = cur == y
    cand_ok = cand == y
    cand_wins = ~cur_ok & cand_ok
    cur_wins = cur_ok & ~cand_ok

    fp_fixed = int((cand_wins & ~y).sum())
    fn_fixed = int((cand_wins & y).sum())
    fp_added = int((cur_wins & ~y).sum())
    fn_added = int((cur_wins & y).sum())

    print()
    print("予測が食い違った行")
    print(f"  候補だけが正しい: {int(cand_wins.sum()):>3}"
          f"  (誤報が減った {fp_fixed} / 見逃しが減った {fn_fixed})")
    print(f"  現行だけが正しい: {int(cur_wins.sum()):>3}"
          f"  (誤報が増えた {fp_added} / 見逃しが増えた {fn_added})")

    b, c = int(cand_wins.sum()), int(cur_wins.sum())
    # 片側の正確な McNemar 検定。昇格判定と同じ関数を使う。
    print(f"  McNemar p（片側） = {mcnemar_p(b, c):.3f}")

    # --- 対応ありブートストラップ ---
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(y), size=(args.n, len(y)))
    diff = _f2(y[idx], cand[idx]) - _f2(y[idx], cur[idx])
    point = float(_f2(y, cand) - _f2(y, cur))

    positives = y[idx].sum(axis=1)
    p2_5, p5, p50, p95, p97_5 = np.percentile(diff, [2.5, 5, 50, 95, 97.5])

    print()
    print(f"F2 の差（候補 - 現行）: {point:+.4f}")
    print(f"ブートストラップ {args.n} 回"
          f"（標本ごとの異常件数 {positives.min()}〜{positives.max()}）")
    print(f"  2.5%: {p2_5:+.4f}   5%: {p5:+.4f}   50%: {p50:+.4f}"
          f"   95%: {p95:+.4f}   97.5%: {p97_5:+.4f}")
    print(f"  差が 0 以下になった割合: {(diff <= 0).mean():.3f}")
    print()
    if c == 0 and b > 0:
        # 現行だけが正しい行が無いと、どう抽出し直しても候補は負けない。
        # このとき「差が 0 以下の割合」は、食い違った b 行を 1 つも引かない確率 ≈ e^(-b) にすぎず、
        # 差の方向についての不確かさを表さない。判定には McNemar を使う。
        print(f"注意: 現行だけが正しい行が 0 件。ブートストラップの結果は e^(-{b}) "
              f"≈ {np.exp(-b):.3f} とほぼ同じになり、判定の根拠にならない")


if __name__ == "__main__":
    main()