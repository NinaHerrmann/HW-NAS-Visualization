import numpy
import pandas as pd

accres = pd.read_csv("result.csv")
allres = pd.read_csv("../all_cifar10_hwnas.csv")
print(accres)
print(allres)

out = accres.merge(
    allres,
    left_on=["dataset", "seed", "idx"],
    right_on=["dataset", "seed", "arch_index"],
    how="inner",          # or "left"/"right"/"outer"
    suffixes=("_acc", "_all")
)

out = out.drop(columns=["arch_index"])   # keep idx
out['bigdiff'] = out["acc"] - out["esp_acc"]
# out.to_csv("mergeddata.csv", index=False)
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# df must contain: bigdiff and arch_0..arch_5
arch_cols = [f"arch_{i}" for i in range(6)]
target = "bigdiff"

X = out[arch_cols]
y = out[target]

# Preprocess: one-hot encode categoricals (works for strings or ints)
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), arch_cols),
    ],
    remainder="drop",
)

# Small random forest (constrained to reduce overfitting)
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=10,
    min_samples_split=20,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", rf),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R^2:", r2_score(y_test, pred))

# Feature importances (on one-hot expanded features)
ohe = model.named_steps["prep"].named_transformers_["cat"]
feat_names = ohe.get_feature_names_out(arch_cols)
importances = model.named_steps["rf"].feature_importances_

imp = (pd.Series(importances, index=feat_names)
         .sort_values(ascending=False))

print("\nTop 25 one-hot features:")
print(imp.head(25))