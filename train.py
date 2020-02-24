from sklearn.base import clone


def uplift_fit_predict(model, X_train, treatment_train, target_train, X_test):
    """
    Реализация простого способа построения uplift-модели.

    Обучаем два бинарных классификатора, которые оценивают вероятность target для клиента:
    1. с которым была произведена коммуникация (treatment=1)
    2. с которым не было коммуникации (treatment=0)

    В качестве оценки uplift для нового клиента берется разница оценок вероятностей:
    Predicted Uplift = P(target|treatment=1) - P(target|treatment=0)
    """
    X_treatment, y_treatment = (
        X_train[treatment_train == 1, :],
        target_train[treatment_train == 1],
    )
    X_control, y_control = (
        X_train[treatment_train == 0, :],
        target_train[treatment_train == 0],
    )
    print("fitting treatment model...")
    model_treatment = clone(model).fit(X_treatment, y_treatment)
    print("fitting control model...")
    model_control = clone(model).fit(X_control, y_control)
    print("predicting treatment and control...")
    predict_treatment = model_treatment.predict_proba(X_test)[:, 1]
    predict_control = model_control.predict_proba(X_test)[:, 1]
    print("predicting uplift...")
    predict_uplift = predict_treatment - predict_control
    return predict_uplift


def uplift_fit_predict_proba(model, X_train, treatment_train, target_train, X_test):
    """
    Реализация простого способа построения uplift-модели.

    Обучаем два бинарных классификатора, которые оценивают вероятность target для клиента:
    1. с которым была произведена коммуникация (treatment=1)
    2. с которым не было коммуникации (treatment=0)

    В качестве оценки uplift для нового клиента берется разница оценок вероятностей:
    Predicted Uplift = P(target|treatment=1) - P(target|treatment=0)
    """
    X_treatment, y_treatment = (
        X_train[treatment_train == 1, :],
        target_train[treatment_train == 1],
    )
    X_control, y_control = (
        X_train[treatment_train == 0, :],
        target_train[treatment_train == 0],
    )
    print("fitting treatment model...")
    model_treatment = clone(model).fit(X_treatment, y_treatment)
    print("fitting control model...")
    model_control = clone(model).fit(X_control, y_control)
    print("predicting treatment and control...")
    predict_treatment = model_treatment.predict(X_test)
    predict_control = model_control.predict(X_test)
    print("predicting uplift...")
    predict_uplift = predict_treatment - predict_control
    return predict_uplift
