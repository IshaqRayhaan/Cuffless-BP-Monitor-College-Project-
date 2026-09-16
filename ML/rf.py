import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mat73 import loadmat
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Optional: PyTorch for deep learning baseline
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Deep learning section will be skipped.")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
DATA_DIR = r"C:\p2bl"        # Folder containing your .mat file
SAMPLE_RATE = 125            # PulseDB signals are sampled at 125 Hz
SEGMENT_LENGTH = 1250        # 10 seconds * 125 Hz = 1250 samples


# =============================================================================
# 2. DATA LOADING
# =============================================================================
def load_subset(filepath, max_samples=None):
    """
    Load a PulseDB Kaggle subset .mat file (v7.3 format).

    Returns:
        signals: np.ndarray of shape (N, 2, L) or (N, L, 2)
                 Channel 0: ECG, Channel 1: PPG
        sbp: np.ndarray of shape (N,)
        dbp: np.ndarray of shape (N,)
        features: dict with Age, Gender, etc.
    """
    print(f"Loading {filepath} ...")
    data = loadmat(filepath)
    subset = data['Subset']

    signals = np.array(subset['Signals'], dtype=np.float32)  # <-- FIX 2: float32
    sbp = np.array(subset['SBP']).squeeze()
    dbp = np.array(subset['DBP']).squeeze()

    # FIX 1: Subsample if too large
    if max_samples is not None and len(sbp) > max_samples:
        print(f"  Subsampling to {max_samples:,} samples...")
        idx = np.random.choice(len(sbp), max_samples, replace=False)
        signals = signals[idx]
        sbp = sbp[idx]
        dbp = dbp[idx]
    else:
        idx = slice(None)

    # Extract demographic features if available
    features = {}
    for key in ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'HR']:
        if key in subset:
            arr = np.array(subset[key]).squeeze()
            features[key] = arr[idx] if isinstance(idx, np.ndarray) else arr

    print(f"  Signals shape: {signals.shape}")
    print(f"  SBP range: [{sbp.min():.1f}, {sbp.max():.1f}] mmHg")
    print(f"  DBP range: [{dbp.min():.1f}, {dbp.max():.1f}] mmHg")
    print(f"  Number of samples: {len(sbp)}")

    return signals, sbp, dbp, features

def load_all_data(data_dir):
    """Load train and test subsets. Requires a pre-split test file to exist."""
    train_path = os.path.join(data_dir, 'VitalDB_Train_Subset.mat')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    possible_test_names = [
        'VitalDB_CalFree_Test_Subset.mat',
        'VitalDB_CalBased_Test_Subset.mat',
        'VitalDB_Test_Subset.mat',
        'Test_Subset.mat'
    ]
    test_path = next((os.path.join(data_dir, n) for n in possible_test_names
                       if os.path.exists(os.path.join(data_dir, n))), None)

    if test_path is None:
        raise FileNotFoundError(
            f"No test subset file found in {data_dir}. Expected one of: "
            f"{possible_test_names}. Place the PulseDB pre-split test file there."
        )

    print(f"[LOAD] Train file: {train_path}")
    train_signals, train_sbp, train_dbp, train_feat = load_subset(train_path)
    print(f"[LOAD] Test file:  {test_path}")
    test_signals, test_sbp, test_dbp, test_feat = load_subset(test_path)

    return (train_signals, train_sbp, train_dbp, train_feat,
            test_signals, test_sbp, test_dbp, test_feat)


# =============================================================================
# 3. DATA INSPECTION & VISUALIZATION
# =============================================================================
def inspect_data(signals, sbp, dbp, features, title="Dataset"):
    """Print statistics and plot sample signals."""
    print(f"\n=== {title} Inspection ===")
    print(f"Total segments: {len(sbp)}")
    print(f"SBP: {sbp.mean():.2f} ± {sbp.std():.2f} mmHg")
    print(f"DBP: {dbp.mean():.2f} ± {dbp.std():.2f} mmHg")

    if 'Age' in features:
        print(f"Age: {features['Age'].mean():.1f} ± {features['Age'].std():.1f} years")
    if 'Gender' in features:
        print(f"Gender (M/F): {np.sum(features['Gender']==1)}/{np.sum(features['Gender']==0)}")

    # Plot random samples
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{title}: Sample ECG & PPG Signals', fontsize=14)

    for i in range(3):
        idx = np.random.randint(0, len(sbp))
        sig = signals[idx]

        # Handle different signal shapes
        if sig.ndim == 1:
            # Single channel fallback
            ecg = sig
            ppg = sig
        elif sig.shape[0] == 2:
            ecg = sig[0, :]
            ppg = sig[1, :]
        else:
            ecg = sig[:, 0]
            ppg = sig[:, 1]

        t = np.arange(len(ecg)) / SAMPLE_RATE

        axes[0, i].plot(t, ecg, 'b-', linewidth=0.8)
        axes[0, i].set_title(f'Sample {idx} | SBP:{sbp[idx]:.0f} DBP:{dbp[idx]:.0f}')
        axes[0, i].set_ylabel('ECG')
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(t, ppg, 'r-', linewidth=0.8)
        axes[1, i].set_ylabel('PPG')
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_samples.png', dpi=150)
    plt.show()
    print(f"Saved plot: {title.lower().replace(' ', '_')}_samples.png")


def plot_bp_distribution(sbp, dbp, title="BP Distribution"):
    """Plot histograms of SBP and DBP."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(sbp, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    axes[0].axvline(sbp.mean(), color='darkred', linestyle='--', label=f'Mean={sbp.mean():.1f}')
    axes[0].set_xlabel('Systolic BP (mmHg)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('SBP Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(dbp, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(dbp.mean(), color='darkblue', linestyle='--', label=f'Mean={dbp.mean():.1f}')
    axes[1].set_xlabel('Diastolic BP (mmHg)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('DBP Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('bp_distribution.png', dpi=150)
    plt.show()
    print("Saved plot: bp_distribution.png")


# =============================================================================
# 4. FEATURE EXTRACTION (Hand-crafted for ML baseline)
# =============================================================================
def extract_ppg_features(ppg_signal):
    """
    Extract simple statistical and morphological features from PPG signal.
    This is a lightweight baseline — real research uses more sophisticated features.
    """
    ppg = np.array(ppg_signal).squeeze()

    # Basic statistics
    features = {
        'ppg_mean': np.mean(ppg),
        'ppg_std': np.std(ppg),
        'ppg_max': np.max(ppg),
        'ppg_min': np.min(ppg),
        'ppg_range': np.max(ppg) - np.min(ppg),
        'ppg_skewness': pd.Series(ppg).skew(),
        'ppg_kurtosis': pd.Series(ppg).kurtosis(),
    }

    # Frequency domain (simple spectral energy)
    fft_vals = np.abs(np.fft.rfft(ppg))
    features['ppg_spectral_energy'] = np.sum(fft_vals**2)
    features['ppg_peak_freq'] = np.argmax(fft_vals)

    return features


def prepare_ml_features(signals, features_dict, log_every=500):
    n = len(signals)
    print(f"[FEATURES] Starting extraction on {n} samples...")
    feat_list = []

    for i in range(n):
        sig = signals[i]
        if sig.ndim == 1:
            ppg = sig
        elif sig.shape[0] == 2:
            ppg = sig[1, :]
        else:
            ppg = sig[:, 1]

        feats = extract_ppg_features(ppg)
        if 'Age' in features_dict: feats['age'] = features_dict['Age'][i]
            if 'Gender' in features_dict:
                g = features_dict['Gender'][i]
                feats['gender'] = 1.0 if str(g).strip().upper() == 'M' else 0.0
        if 'BMI' in features_dict: feats['bmi'] = features_dict['BMI'][i]
        if 'HR' in features_dict: feats['hr'] = features_dict['HR'][i]
        feat_list.append(feats)

        if (i + 1) % log_every == 0 or (i + 1) == n:
            print(f"[FEATURES]   {i+1}/{n} samples processed")

    print(f"[FEATURES] Done. Extracted {len(feat_list)} feature rows.")
    return pd.DataFrame(feat_list)


# =============================================================================
# 5. EVALUATION METRICS
# =============================================================================
def evaluate_bp_predictions(y_true_sbp, y_pred_sbp, y_true_dbp, y_pred_dbp, title="Model"):
    """
    Standard BP estimation metrics.
    Reference: IEEE 1708-2014 and AAMI SP-10 standards.
    """
    def _metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        std = np.std(y_pred - y_true)
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        # AAMI standard: <= 5 mmHg mean error, <= 8 mmHg std deviation
        aami_pass = (mae <= 5) and (std <= 8)

        # BHS grades: % of predictions within 5, 10, 15 mmHg
        diff = np.abs(y_pred - y_true)
        within_5 = np.mean(diff <= 5) * 100
        within_10 = np.mean(diff <= 10) * 100
        within_15 = np.mean(diff <= 15) * 100

        # BHS grading
        if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
            bhs_grade = 'A'
        elif within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
            bhs_grade = 'B'
        elif within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
            bhs_grade = 'C'
        else:
            bhs_grade = 'D'

        print(f"\n{name} Results:")
        print(f"  MAE:  {mae:.2f} mmHg")
        print(f"  RMSE: {rmse:.2f} mmHg")
        print(f"  STD:  {std:.2f} mmHg")
        print(f"  Corr: {corr:.3f}")
        print(f"  AAMI: {'PASS' if aami_pass else 'FAIL'} (ME<=5, SD<=8)")
        print(f"  BHS Grade: {bhs_grade} (≤5:{within_5:.1f}%, ≤10:{within_10:.1f}%, ≤15:{within_15:.1f}%)")

        return {'mae': mae, 'rmse': rmse, 'std': std, 'corr': corr,
                'bhs_grade': bhs_grade, 'within_5': within_5}

    sbp_res = _metrics(y_true_sbp, y_pred_sbp, f"{title} SBP")
    dbp_res = _metrics(y_true_dbp, y_pred_dbp, f"{title} DBP")

    # Bland-Altman style scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SBP
    axes[0].scatter(y_true_sbp, y_pred_sbp, alpha=0.3, s=10, color='salmon')
    axes[0].plot([50, 200], [50, 200], 'k--', label='Perfect prediction')
    axes[0].set_xlabel('True SBP (mmHg)')
    axes[0].set_ylabel('Predicted SBP (mmHg)')
    axes[0].set_title(f'SBP: MAE={sbp_res["mae"]:.2f} mmHg, Grade={sbp_res["bhs_grade"]}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # DBP
    axes[1].scatter(y_true_dbp, y_pred_dbp, alpha=0.3, s=10, color='skyblue')
    axes[1].plot([30, 140], [30, 140], 'k--', label='Perfect prediction')
    axes[1].set_xlabel('True DBP (mmHg)')
    axes[1].set_ylabel('Predicted DBP (mmHg)')
    axes[1].set_title(f'DBP: MAE={dbp_res["mae"]:.2f} mmHg, Grade={dbp_res["bhs_grade"]}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'{title} Predictions vs Ground Truth', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_results.png', dpi=150)
    plt.show()
    print(f"\nSaved plot: {title.lower().replace(' ', '_')}_results.png")

    return sbp_res, dbp_res


# =============================================================================
# 6. MACHINE LEARNING BASELINE (Random Forest)
# =============================================================================
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def run_ml_baseline(x_train, y_train_sbp, y_train_dbp,
                    x_test, y_test_sbp, y_test_dbp,
                    export_path_prefix="model"):
    """
    Train a simple Random Forest baseline and export artifacts
    for the firmware demo pipeline.
    """
    print("\n" + "=" * 60)
    print("Training Random Forest Baseline...")
    print("=" * 60)

    # 1. Fit scaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 2. Train SBP model
    print("Training SBP model...")
    rf_sbp = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    rf_sbp.fit(x_train_scaled, y_train_sbp)
    pred_sbp = rf_sbp.predict(x_test_scaled)

    # 3. Train DBP model
    print("Training DBP model...")
    rf_dbp = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    rf_dbp.fit(x_train_scaled, y_train_dbp)
    pred_dbp = rf_dbp.predict(x_test_scaled)

    # 4. Export artifacts for hardware team integration
    joblib.dump(rf_sbp, f"{export_path_prefix}_sbp.pkl")
    joblib.dump(rf_dbp, f"{export_path_prefix}_dbp.pkl")
    joblib.dump(scaler, f"{export_path_prefix}_scaler.pkl")
    print(f"\nExported: {export_path_prefix}_sbp.pkl, {export_path_prefix}_dbp.pkl, {export_path_prefix}_scaler.pkl")

    # 5. Evaluate
    return evaluate_bp_predictions(
        y_test_sbp, pred_sbp,
        y_test_dbp, pred_dbp,
        title="Random Forest"
    )


# =============================================================================
# 7. MAIN EXECUTION — Invoke everything here
# =============================================================================
if __name__ == "__main__":
    
    DATA_DIR = r"D:\P2BL\ML\data"  # Folder containing VitalDB_Train_Subset.mat
    
    print("=" * 60)
    print("STARTING BLOOD PRESSURE ESTIMATION PIPELINE")
    print("=" * 60)

    print("\n>>> [STAGE] Loading dataset...")
    train_signals, train_sbp, train_dbp, train_feat, \
    test_signals, test_sbp, test_dbp, test_feat = load_all_data(DATA_DIR)
    print(">>> [STAGE] Loading complete.")

    print("\n>>> [STAGE] Inspecting training data...")
    inspect_data(train_signals, train_sbp, train_dbp, train_feat, title="Train")
    plot_bp_distribution(train_sbp, train_dbp, title="Train BP Distribution")
    print(">>> [STAGE] Inspection complete.")

    print("\n>>> [STAGE] Extracting features...")
    X_train = prepare_ml_features(train_signals, train_feat)
    X_test = prepare_ml_features(test_signals, test_feat)
    print(f">>> [STAGE] Feature extraction complete. Shape: {X_train.shape}")

    print("\n>>> [STAGE] Training Random Forest models...")
    sbp_results, dbp_results = run_ml_baseline(
        X_train, train_sbp, train_dbp, X_test, test_sbp, test_dbp,
        export_path_prefix="bp_model"
    )
    print(">>> [STAGE] Training + evaluation complete.")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)