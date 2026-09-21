# src/promote.py
"""候補モデルを現行モデルと比較し、基準を満たせば昇格する。

見逃し（未検知の異常）のほうが誤検知より損失が大きいため、F2 を主指標にする。
ただし F2 だけでは precision が崩壊しても「改善」と判定されうるので、
precision と recall に下限を設けている。

昇格先は2つ。
- ローカル: MLflow の production alias を付け替える（--candidate）
- AWS: SageMaker Model Package の approval status を Approved にする（--model-package-arn）
"""

import argparse
import math

import boto3

from src.registry import REGION, latest_approved

MODEL_NAME = "cost-anomaly-detector"
ALIAS = "production"

PRIMARY_METRIC = "f2"
MIN_PRECISION = 0.60
MIN_RECALL = 0.80
# 食い違った行が候補に偏っているかの有意水準。
# 300 行・異常 30 件のホールドアウトでは、誤報の減少なら 5 件以上の差が要る。
ALPHA = 0.05


def evaluate(current: dict | None, candidate: dict,
             min_precision: float, min_recall: float) -> list[tuple[str, bool, str]]:
    """昇格可否のチェック結果を返す。レジストリに依存しない純粋関数。"""
    checks = []

    if current is None:
        checks.append(("baseline", True, "現行モデルなし、初回昇格"))
    else:
        base = current[PRIMARY_METRIC]
        cand_score = candidate[PRIMARY_METRIC]
        checks.append((
            PRIMARY_METRIC, cand_score > base,
            f"{base:.3f} -> {cand_score:.3f} ({cand_score - base:+.3f})",
        ))

    p = candidate.get("precision", 0.0)
    checks.append(("precision", p >= min_precision, f"{p:.3f} >= {min_precision:.2f}"))

    r = candidate.get("recall", 0.0)
    checks.append(("recall", r >= min_recall, f"{r:.3f} >= {min_recall:.2f}"))

    return checks


def mcnemar_p(candidate_only: int, current_only: int) -> float:
    """片側の正確な McNemar 検定の p 値。

    判定が食い違った行のうち、候補だけが正しかった行の数が、
    五分五分の実力でも偶然これ以上になる確率。食い違いが無ければ 1.0。
    """
    n = candidate_only + current_only
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(candidate_only, n + 1)) / 2**n


def significance_check(candidate_only: int, current_only: int,
                       alpha: float = ALPHA) -> tuple[str, bool, str]:
    """差が偶然でないかのチェック。evaluate() の結果に追加して使う。

    ブートストラップの百分位点は、食い違いが少なく片側に偏っているときに過信する
    （手元の行を母集団とみなすので、候補が負けた行が無ければ負ける標本も作れない）。
    そのため食い違った行の偏りを直接検定する。
    """
    p = mcnemar_p(candidate_only, current_only)
    return ("mcnemar", p < alpha,
            f"{candidate_only}:{current_only} p={p:.3f} < {alpha:.2f}")


def judge(curr: dict | None, curr_label: str | None, cand: dict, cand_label: str,
          min_precision: float, min_recall: float,
          extra_checks: list[tuple[str, bool, str]] | None = None,
          ) -> list[tuple[str, bool, str]]:
    """現行と候補を表示し、判定結果を返す。指標が欠けていたら判定不能で止める。"""

    def fmt(metrics: dict, key: str) -> str:
        return f"{metrics[key]:.3f}" if key in metrics else "n/a"

    if curr_label:
        print(f"current  ({curr_label}): "
              f"f2={fmt(curr, 'f2')}  "
              f"precision={fmt(curr, 'precision')}  "
              f"recall={fmt(curr, 'recall')}")
    else:
        print("current: none (初回昇格)")

    print(f"candidate({cand_label}): "
          f"f2={fmt(cand, 'f2')}  "
          f"precision={fmt(cand, 'precision')}  "
          f"recall={fmt(cand, 'recall')}")
    print()

    if curr is not None:
        if PRIMARY_METRIC not in curr:
            print(f"ERROR: 現行モデル {curr_label} に "
                  f"'{PRIMARY_METRIC}' が記録されていません。")
            print("同じ指標で再評価してから比較してください。")
            raise SystemExit(2)
        if PRIMARY_METRIC not in cand:
            print(f"ERROR: 候補モデル {cand_label} に "
                  f"'{PRIMARY_METRIC}' が記録されていません。")
            raise SystemExit(2)

    checks = evaluate(curr, cand, min_precision, min_recall) + (extra_checks or [])
    for name, ok, detail in checks:
        print(f"{name:<12} {detail:<30} {'PASS' if ok else 'FAIL'}")
    print()
    return checks


# --- MLflow（ローカル） ---

def _load_mlflow(candidate_version: int):
    # mlflow はローカル専用。AWS 側の実行環境に含めないため関数内で import する。
    import mlflow

    client = mlflow.MlflowClient()
    cand_mv = client.get_model_version(MODEL_NAME, str(candidate_version))
    cand = client.get_run(cand_mv.run_id).data.metrics
    cand_label = f"v{candidate_version}"

    # alias の有無は例外ではなく明示的に確認する。
    # 接続失敗などを「現行なし」と取り違えると、比較せずに昇格してしまう。
    aliases = client.get_registered_model(MODEL_NAME).aliases
    if ALIAS not in aliases:
        return None, None, cand, cand_label

    cur_mv = client.get_model_version(MODEL_NAME, str(aliases[ALIAS]))
    curr = client.get_run(cur_mv.run_id).data.metrics
    return curr, f"v{cur_mv.version}", cand, cand_label


def _promote_mlflow(candidate_version: int) -> None:
    import mlflow

    mlflow.MlflowClient().set_registered_model_alias(
        MODEL_NAME, ALIAS, str(candidate_version)
    )
    print(f"PROMOTED: {ALIAS} -> version {candidate_version}")
    print("推論APIを再起動してください: docker compose restart ml-app")


# --- SageMaker（AWS） ---

def _metrics_from_package(desc: dict) -> dict:
    """CustomerMetadataProperties は文字列しか持てないので float に戻す。

    training_job のような数値でない値は捨てる。
    """
    metrics = {}
    for key, value in desc.get("CustomerMetadataProperties", {}).items():
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _load_sagemaker(sm, candidate_arn: str):
    cand_desc = sm.describe_model_package(ModelPackageName=candidate_arn)
    cand = _metrics_from_package(cand_desc)
    cand_label = f"v{cand_desc['ModelPackageVersion']}"
    group = cand_desc["ModelPackageGroupName"]

    # 最新の Approved を現行とみなす。候補自身は比較対象から除く。
    cur_desc = latest_approved(sm, group, exclude_arn=candidate_arn)
    if cur_desc is None:
        return None, None, cand, cand_label

    curr = _metrics_from_package(cur_desc)
    return curr, f"v{cur_desc['ModelPackageVersion']}", cand, cand_label


def _set_status(sm, arn: str, status: str, reason: str) -> None:
    sm.update_model_package(
        ModelPackageArn=arn,
        ModelApprovalStatus=status,
        ApprovalDescription=reason[:1000],
    )


def decide_package(sm, arn: str, curr: dict | None, curr_label: str | None,
                   cand: dict, cand_label: str,
                   min_precision: float = MIN_PRECISION, min_recall: float = MIN_RECALL,
                   dry_run: bool = False, basis: str = "registered",
                   extra_checks: list[tuple[str, bool, str]] | None = None) -> bool:
    """与えられた指標で判定し、Approved / Rejected を付ける。昇格したら True。

    basis は比較に使った数値の出どころ。記録に残して、後から判定の根拠を区別できるようにする。
      registered: 登録時の指標（各モデルが別々のデータで測った値）
      holdout:    同じホールドアウトで測った値
    """
    checks = judge(curr, curr_label, cand, cand_label, min_precision, min_recall,
                   extra_checks)
    source = f"promote.py ({basis})"

    if not all(ok for _, ok, _ in checks):
        print(f"REJECT: {cand_label} は昇格基準を満たしません")
        if not dry_run:
            # 判定結果をパッケージ側に残す。後から理由を追える。
            failed = ", ".join(f"{n} FAIL ({d})" for n, ok, d in checks if not ok)
            _set_status(sm, arn, "Rejected", f"{source}: {failed}")
            print("status: Rejected")
        return False

    if dry_run:
        print(f"PROMOTE (dry-run): {cand_label} は昇格可能です")
        return True

    detail = "; ".join(f"{n} PASS ({d})" for n, _, d in checks)
    _set_status(sm, arn, "Approved", f"{source}: {detail}")
    print(f"PROMOTED: {cand_label} -> Approved")
    return True


def promote_package(sm, arn: str, min_precision: float = MIN_PRECISION,
                    min_recall: float = MIN_RECALL, dry_run: bool = False) -> bool:
    """登録時の指標で判定する。ホールドアウトが無いときの手動判定用。"""
    curr, curr_label, cand, cand_label = _load_sagemaker(sm, arn)
    return decide_package(sm, arn, curr, curr_label, cand, cand_label,
                          min_precision, min_recall, dry_run, basis="registered")


# --- 共通 ---

def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--candidate", type=int,
                        help="昇格を検討する MLflow のモデルバージョン")
    target.add_argument("--model-package-arn",
                        help="昇格を検討する SageMaker Model Package の ARN")
    parser.add_argument("--dry-run", action="store_true",
                        help="判定のみ行い、alias も approval status も変更しない")
    parser.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL)
    args = parser.parse_args()

    if args.model_package_arn:
        sm = boto3.client("sagemaker", region_name=REGION)
        if not promote_package(sm, args.model_package_arn,
                               args.min_precision, args.min_recall, args.dry_run):
            raise SystemExit(1)
        return

    curr, curr_label, cand, cand_label = _load_mlflow(args.candidate)
    checks = judge(curr, curr_label, cand, cand_label,
                   args.min_precision, args.min_recall)

    if not all(ok for _, ok, _ in checks):
        print(f"REJECT: {cand_label} は昇格基準を満たしません")
        raise SystemExit(1)

    if args.dry_run:
        print(f"PROMOTE (dry-run): {cand_label} は昇格可能です")
        return

    _promote_mlflow(args.candidate)


if __name__ == "__main__":
    # 未捕捉の例外は終了コード 1（昇格拒否）と区別できない。
    # 想定外の失敗は判定不能 (2) にする。
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"promote failed: {type(e).__name__}: {e}")
        raise SystemExit(2) from e