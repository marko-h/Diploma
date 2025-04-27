import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.metrics       import accuracy_score, classification_report
from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.tree            import DecisionTreeClassifier

# ─── Load & Preprocess ─────────────────────────────────────────────────────────

columns = [
    'age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss',
    'hours-per-week','native-country','income'
]
df = pd.read_csv('./datasets/adult/adult_full.csv', skipinitialspace=True)
X = df.drop('income', axis=1)
y = df['income']

categorical = X.select_dtypes(include=['object']).columns
numeric     = X.select_dtypes(include=['int64','float64']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

le    = LabelEncoder()
y_enc = le.fit_transform(y)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)
X_tr_p = preprocessor.fit_transform(X_tr)
X_te_p = preprocessor.transform(X_te)

classes = np.unique(y_tr)

# Build 5 disjoint, stratified batches
sk5     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
batches = [test_idx for _, test_idx in sk5.split(X_tr_p, y_tr)]


# ─── MODEL LOOP ─────────────────────────────────────────────────────────────────
models = [
    ("MLP",         MLPClassifier(hidden_layer_sizes=(128,32), max_iter=50,   random_state=42)),
    ("Logistic",    LogisticRegression(solver='saga', warm_start=True, max_iter=1000, random_state=42)),
    ("KNN",         KNeighborsClassifier(n_neighbors=5)),
    #("SVM",         SVC(kernel='rbf', probability=True, random_state=42)),
    ("DecisionTree",DecisionTreeClassifier(random_state=42))
]

cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X_full_p = preprocessor.fit_transform(X)

for name, model in models:
    print(f"\n==== {name} ====")

    # 1) Regular 70/30
    model.fit(X_tr_p, y_tr)
    y_pr = model.predict(X_te_p)
    print(f"[Regular]   Acc: {accuracy_score(y_te, y_pr):.4f}")
    print(classification_report(y_te, y_pr, target_names=le.classes_))

    # 2) Partial Fit?
    if hasattr(model, "partial_fit"):
        print("[PartialFit]")        
        m_pf = model.__class__(**model.get_params())
        for i, idx in enumerate(batches, 1):
            Xb, yb = X_tr_p[idx], y_tr[idx]
            m_pf.partial_fit(Xb, yb, classes=classes)
            acc = accuracy_score(y_te, m_pf.predict(X_te_p))
            print(f" batch {i}/5 — {acc:.4f}")
        print(classification_report(y_te, m_pf.predict(X_te_p), target_names=le.classes_))
    else:
        print("[PartialFit]  SKIPPED (unsupported)")

    # 3) Warm Start?
    if getattr(model, "warm_start", False):
        print("[WarmStart]")
        m_ws = model.__class__(**model.get_params())
        m_ws.set_params(max_iter=1, warm_start=True)  # one epoch per fit()
        for epoch in range(10):
            m_ws.fit(X_tr_p, y_tr)
            acc = accuracy_score(y_te, m_ws.predict(X_te_p))
            print(f" epoch {epoch+1:02d}/10 — {acc:.4f}")
        print(classification_report(y_te, m_ws.predict(X_te_p), target_names=le.classes_))
    else:
        print("[WarmStart] SKIPPED (unsupported)")

    # 4) 10-Fold CV (Regular)
    scores = cross_val_score(model, X_full_p, y_enc, cv=cv10, n_jobs=-1)
    print(f"[10-Fold CV] Acc folds: {[f'{s:.3f}' for s in scores]}")
    preds  = cross_val_predict(model, X_full_p, y_enc, cv=cv10, n_jobs=-1)
    print(f"Average CV Acc: {np.mean(scores):.4f}")
    print(classification_report(y_enc, preds, target_names=le.classes_))


