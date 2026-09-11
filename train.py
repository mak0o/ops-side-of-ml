import time

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. データの準備
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. 実験の設定
mlflow.set_experiment("iris_classification_project")

with mlflow.start_run():
    # ハイパーパラメータの設定
    n_estimators = 100
    max_depth = 5
    
    # 【記録】パラメータをMLflowに送る
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # モデルの訓練
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 【記録】評価指標（メトリクス）をMLflowに送る
    mlflow.log_metric("accuracy", accuracy)
    
    time.sleep(1)

    # 【記録】学習済みモデルそのものをMinIOに保存（Artifactとして登録）
    mlflow.sklearn.log_model(
        clf,
        name="iris_model",
        pip_requirements=["scikit-learn", "mlflow"],
    )
    
    print(f"実験完了！ 精度: {accuracy}")