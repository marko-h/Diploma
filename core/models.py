from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models():
    """Return a list of models to evaluate."""
    return [
        # Traditional sklearn models
        ("MLPClassifier", MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=50, random_state=42)),
        ("LogisticRegression", LogisticRegression(solver='saga', warm_start=True, max_iter=500, random_state=42)),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),

        # Models with partial_fit support
        ("SGDClassifier", SGDClassifier(max_iter=500, random_state=42)),
        ("PassiveAggressive", PassiveAggressiveClassifier(max_iter=500, random_state=42)),
        ("BernoulliNB", BernoulliNB()),

        # Models with warm_start support
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, warm_start=True, random_state=42)),
        ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, warm_start=True, random_state=42)),

        # Boosting libraries with init_model capability
        ("XGBoost", XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ("LightGBM", LGBMClassifier(n_estimators=50, random_state=42)),
        ("CatBoost", CatBoostClassifier(iterations=50, verbose=0, random_state=42))
    ]