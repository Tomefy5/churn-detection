from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


def oversampling_SMOTE(data: pd.DataFrame, target_col: str):
    y = data[target_col].map({ "No": 0, "Yes": 1 })
    X = data.drop(columns=[target_col])

    # Séparation des colonnes
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns

    # Encodage des colonnes catégorielles
    encoder = OrdinalEncoder()
    X_cat_encoded = encoder.fit_transform(X[cat_cols])  

    print(X_cat_encoded)

    X_all = pd.concat([
        pd.DataFrame(X_cat_encoded, columns=cat_cols),
        pd.DataFrame(X[num_cols].values, columns=num_cols)
    ], axis=1)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_sampled, y_sampled = smote.fit_resample(X_all, y)

    data_sampled = pd.DataFrame(X_sampled, columns=X_all.columns)
    data_sampled[target_col] = y_sampled

    return data_sampled


def main():
    print("Fontion main")
    data = pd.DataFrame({
    "Age": [25, 30, 45, 35, 22, 40, 28, 50],
    "Salaire": [50000, 60000, 80000, 56000, 45000, 78000, 52000, 90000],
    "Churn": ["No", "No", "Yes", "No", "No", "Yes", "No", "No"]
    })

    data_sampled = oversampling_SMOTE(data, "Churn")

    print("Ancien shape: ", data.shape)
    print("Shape actuel: ", data_sampled.shape)
    

if __name__ == "__main__":
    main()