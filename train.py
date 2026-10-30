from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
(ROOT / "models").mkdir(exist_ok=True)
data = load_iris()
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"))
])
pipeline.fit(data.data, data.target)
joblib.dump(pipeline, ROOT / "models" / "iris_pipeline.joblib")
print("Model saved successfully.")
