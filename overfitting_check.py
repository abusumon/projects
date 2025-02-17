import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

def detect_overfitting(model, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    mean_cv, std_cv = np.mean(cv_scores), np.std(cv_scores)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    noise_std = 0.01 * (X_test.max(axis=0) - X_test.min(axis=0))
    X_test_noisy = X_test + np.random.normal(0, noise_std, X_test.shape)
    noisy_test_acc = model.score(X_test_noisy, y_test)

    mean_confidence = None
    try:
        decision_values = model.decision_function(X_test)
        mean_confidence = np.mean(np.abs(decision_values))
    except AttributeError:
        try:
            probs = model.predict_proba(X_test)
            mean_confidence = np.mean(np.max(probs, axis=1))
        except AttributeError:
            pass

    overfitting_warnings = []
    if train_acc > 0.98 and test_acc < (train_acc -2 * std_cv):
        overfitting_warnings.append(" Large train-test accuracy gap.")
    if test_acc > 0.98 and noisy_test_acc < (test_acc - 2* std_cv):
        overfitting_warnings.append("Model is sensitive to noise.")
    if mean_cv > 0.98 and std_cv > 0.05:
        overfitting_warnings.append("High cross-validation score variance.")

    status = "No clear overfitting signs. " if not overfitting_warnings else " Possible overfitting detected."
    return{
        "cross_val_mean": mean_cv,
        "cross_val_score": std_cv,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "noisy_test_accuracy": noisy_test_acc,
        "mean_confidence": mean_confidence,
        "overfitting_warnings": overfitting_warnings,
        "status": status
    }

