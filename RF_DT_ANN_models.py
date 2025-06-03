import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

# Load datasets
def load_datasets(historical_path, realtime_path):
    historical_df = pd.read_csv(historical_path, encoding='ISO-8859-1')
    realtime_df = pd.read_csv(realtime_path, encoding='ISO-8859-1')

    print("Historical columns:", historical_df.columns)
    print("Realtime columns:", realtime_df.columns)

    return historical_df, realtime_df

# Clean dataset
def clean_dataset(df):
    if 'Series' in df.columns:
        df = df[df['Series'].astype(str).str.contains(r'\(number\)', na=False)]
    else:
        print("Warning: 'Series' column not found; skipping filtering.")

    if 'migrated' in df.columns:
        df['migrated'] = df['migrated'].astype(str).str.replace(',', '').str.extract('(\d+\.?\d*)')[0]
        df['migrated'] = pd.to_numeric(df['migrated'], errors='coerce')
    else:
        raise ValueError("Missing 'migrated' column.")

    df = df.dropna(subset=['migrated'])
    return df

# Preprocess data
def preprocess_data(df):
    drop_cols = [col for col in ['Footnotes', 'Source'] if col in df.columns]
    df = df.dropna()

    if 'migrated' not in df.columns:
        raise ValueError("'migrated' column missing.")

    X = df.drop(['migrated'] + drop_cols, axis=1, errors='ignore')
    y = df['migrated']

    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    if X.empty:
        raise ValueError("No features found.")
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns

# Build ANN
def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Loss (MSE)')
    plt.plot(history.history['mae'], label='MAE')
    plt.title("ANN Training Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot prediction comparison
def plot_predictions(true, preds, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(true, preds, alpha=0.7)
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Train and evaluate models
def train_and_evaluate(historical_df):
    X, y, scaler, feature_columns = preprocess_data(historical_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)

    ann = build_ann(X_train.shape[1])
    history = ann.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
    ann_preds = ann.predict(X_test).flatten()

    # Metrics
    def print_metrics(model_name, true, pred):
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)
        print(f"\n[{model_name} Results]")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")





    def classification_metrics(true, pred, model_name):
        # Convert continuous values to 3 bins: low, medium, high
        bins = np.percentile(true, [0, 33.3, 66.6, 100])
        true_binned = np.digitize(true, bins) - 1
        pred_binned = np.digitize(pred, bins) - 1

        print(f"\n[{model_name} Classification Metrics]")
        print("Accuracy:", accuracy_score(true_binned, pred_binned))
        print("Precision:", precision_score(true_binned, pred_binned, average='weighted', zero_division=0))
        print("Recall:", recall_score(true_binned, pred_binned, average='weighted', zero_division=0))
        print("F1 Score:", f1_score(true_binned, pred_binned, average='weighted', zero_division=0))

    print_metrics("Random Forest", y_test, rf_preds)
    classification_metrics(y_test, rf_preds, "Random Forest")

    print_metrics("Decision Tree", y_test, dt_preds)
    classification_metrics(y_test, dt_preds, "Decision Tree")

    print_metrics("Artificial Neural Network", y_test, ann_preds)
    classification_metrics(y_test, ann_preds, "Artificial Neural Network")

    print_metrics("Random Forest", y_test, rf_preds)
    print_metrics("Decision Tree", y_test, dt_preds)
    print_metrics("Artificial Neural Network", y_test, ann_preds)

    # Plots
    plot_training_history(history)
    plot_predictions(y_test, rf_preds, "Random Forest Predictions")
    plot_predictions(y_test, dt_preds, "Decision Tree Predictions")
    plot_predictions(y_test, ann_preds, "ANN Predictions")

    
    return ann, rf, dt, scaler, feature_columns

# Real-time prediction comparison table
def real_time_prediction_comparison(realtime_df, scaler, feature_columns, ann, rf, dt):
    X_realtime = realtime_df.drop(['migrated'], axis=1, errors='ignore').copy()

    for col in X_realtime.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_realtime[col] = le.fit_transform(X_realtime[col].astype(str))

    # Align features
    for col in feature_columns:
        if col not in X_realtime.columns:
            X_realtime[col] = 0
    X_realtime = X_realtime[feature_columns]
    X_scaled = scaler.transform(X_realtime)

    rf_preds = rf.predict(X_scaled)
    dt_preds = dt.predict(X_scaled)
    ann_preds = ann.predict(X_scaled).flatten()
    combined_preds = (rf_preds + dt_preds + ann_preds) / 3

    # Build comparison DataFrame
    results = pd.DataFrame({
        "Random Forest": rf_preds,
        "Decision Tree": dt_preds,
        "ANN": ann_preds,
        "Combined": combined_preds
    })

    print("\nModel Comparison Table (first 10 rows):")
    print(results.head(10))

    results.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved model_comparison_results.csv")

# Main
def main():
    historical_path = r'C:\UNHCR_dataset.csv'
    realtime_path = r'C:\google_trends.csv'

    historical_df, realtime_df = load_datasets(historical_path, realtime_path)
    historical_df = clean_dataset(historical_df)
    realtime_df = clean_dataset(realtime_df)

    if historical_df.empty:
        print("Error: Historical dataset is empty.")
        return

    if realtime_df.empty:
        print("Error: Real-time dataset is empty.")
        return

    ann, rf, dt, scaler, feature_columns = train_and_evaluate(historical_df)

    # ⬇️ Run real-time comparison
    real_time_prediction_comparison(realtime_df, scaler, feature_columns, ann, rf, dt)

if __name__ == "__main__":
    main()
