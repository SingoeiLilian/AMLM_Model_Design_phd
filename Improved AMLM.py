import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def build_regression_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_classification_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load datasets
    historical_path = r'C:\\UNHCR_dataset.csv'
    real_time_path = r'C:\\google_trends.csv'

    df_hist = pd.read_csv(historical_path, encoding='latin1')
    df_hist_filtered = df_hist[df_hist["Series"] == "International migrant stock: Both sexes (number)"]
    df_hist_filtered = df_hist_filtered.dropna(subset=["Year", "migrated"])
    df_hist_filtered["Year"] = df_hist_filtered["Year"].astype(str).str.replace(',', '')
    df_hist_filtered["migrated"] = df_hist_filtered["migrated"].astype(str).str.replace(',', '')
    X = df_hist_filtered[["Year", "migrated"]].astype(float).values

    df_real = pd.read_csv(real_time_path, encoding='latin1')
    df_real = df_real.dropna(subset=["refugee", "migrated", "asylum", "border crossing"])
    y_reg = df_real["refugee"].astype(float).values
    y_class = (y_reg > np.mean(y_reg)).astype(int)

    # Match input-output length
    min_len = min(len(X), len(y_reg))
    X = X[:min_len]
    y_reg = y_reg[:min_len]
    y_class = y_class[:min_len]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.33, random_state=42)
    y_train_class = (y_train_reg > np.mean(y_train_reg)).astype(int)
    y_test_class = (y_test_reg > np.mean(y_train_reg)).astype(int)

    # Train Regression Models
    reg_ANN = build_regression_model(X_train.shape[1])
    reg_ANN.fit(X_train, y_train_reg, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=0)
    y_pred_ann_reg = reg_ANN.predict(X_test).flatten()

    reg_DT = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg_DT.fit(X_train, y_train_reg)
    y_pred_dt_reg = reg_DT.predict(X_test)

    reg_RF = RandomForestRegressor(n_estimators=150, max_depth=7, random_state=42)
    reg_RF.fit(X_train, y_train_reg)
    y_pred_rf_reg = reg_RF.predict(X_test)

    # Ensemble Regression
    y_pred_reg_ensemble = (0.4 * y_pred_ann_reg + 0.2 * y_pred_dt_reg + 0.4 * y_pred_rf_reg)

    # Train Classification Models
    clf_ANN = build_classification_model(X_train.shape[1])
    clf_ANN.fit(X_train, y_train_class, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)
    y_pred_ann_prob = clf_ANN.predict(X_test).flatten()

    clf_DT = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_DT.fit(X_train, y_train_class)
    y_pred_dt = clf_DT.predict_proba(X_test)[:, 1]

    clf_RF = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    clf_RF.fit(X_train, y_train_class)
    y_pred_rf = clf_RF.predict_proba(X_test)[:, 1]

    # Ensemble Classification
    y_pred_class_prob = (0.4 * y_pred_ann_prob + 0.2 * y_pred_dt + 0.4 * y_pred_rf)
    y_pred_class = (y_pred_class_prob > 0.5).astype(int)

    # Display Regression Metrics
    print("=== REGRESSION METRICS (Ensemble) ===")
    print(f"MAE:  {mean_absolute_error(y_test_reg, y_pred_reg_ensemble):.2f}")
    print(f"MSE:  {mean_squared_error(y_test_reg, y_pred_reg_ensemble):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_ensemble)):.2f}")
    print(f"R^2:  {r2_score(y_test_reg, y_pred_reg_ensemble):.2f}\n")

    # Display Classification Metrics
    print("=== CLASSIFICATION METRICS (Ensemble) ===")
    print(f"Accuracy:  {accuracy_score(y_test_class, y_pred_class):.2f}")
    print(f"Precision: {precision_score(y_test_class, y_pred_class):.2f}")
    print(f"Recall:    {recall_score(y_test_class, y_pred_class):.2f}")
    print(f"F1 Score:  {f1_score(y_test_class, y_pred_class):.2f}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test_class, y_pred_class)
    print("=== CONFUSION MATRIX ===")
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted: No', 'Predicted: Yes'],
                yticklabels=['Actual: No', 'Actual: Yes'])
    plt.title('Confusion Matrix (Ensemble Classifier)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()

    # Classification Report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test_class, y_pred_class, zero_division=0))

    # Plot True vs Predicted Regression
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_reg, label='True Refugee Migration', marker='o')
    plt.plot(y_pred_reg_ensemble, label='Predicted (Ensemble)', marker='x')
    plt.title('True vs Predicted Refugee Migration (Ensemble)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Refugee Migration')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display Predictions
    results_df = pd.DataFrame({
        'Sample Index': range(len(y_test_reg)),
        'True Refugee Migration': y_test_reg,
        'Predicted (Ensemble)': y_pred_reg_ensemble
    })
    print("\n=== Actual vs Predicted Refugee Migration (Top 20) ===")
    print(results_df.head(20))

if __name__ == "__main__":
    main()
