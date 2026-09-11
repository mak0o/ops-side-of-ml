import logging

import mlflow

# エラーを詳細に表示する設定
logging.basicConfig(level=logging.INFO)

mlflow.set_experiment("debug_experiment")

with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    mlflow.log_param("debug", "True")
    
    with open("debug.txt", "w") as f:
        f.write("Debug test")
    
    try:
        print("Artifactのアップロードを開始します...")
        mlflow.log_artifact("debug.txt")
        print("Artifactのアップロード命令が完了しました。")
    except Exception as e:
        print(f"【エラー発生】アップロードに失敗しました: {e}")

print("プログラムが終了しました。")