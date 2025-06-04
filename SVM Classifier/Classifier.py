import os
import pickle
import librosa
import numpy as np

from sklearn.svm import SVC
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# --- CONFIG ---
DATA_DIRS = {
    'Cat': 'Cats',
    'Dog': 'Dogs',
    'Bird': 'Birds',
    'Cow': 'Cows',
    'Frog': 'Frogs'
}

# MFCC parameters
HTK = True
SR = 44100
FMIN = 100
FMAX = 22050
LIFTER = 22
N_MFCC = 24
N_MELS = 128
FRAME_LEN = 0.025
FRAME_HOP = 0.010
N_FFT = int(FRAME_LEN * SR)
HOP_LENGTH = int(FRAME_HOP * SR)

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
N_SPLITS = 8  # Number of folds

# Classifier hyperparams
CLASSIFIERS = {
    'SVM': SVC(kernel='rbf', C=2.19, gamma='scale', random_state=RANDOM_STATE, class_weight= 'balanced'),
}

def extract_audio_features(path):
    y, sr = librosa.load(path, sr=SR)
    
    # --- Initialize feature list ---
    all_features = []

    # --- Time-Domain Features ---
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    
    # --- Spectral Features ---
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    spectral_flatness = librosa.feature.spectral_flatness(S=S)
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    
    # --- Pitch Features ---
    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr, 
                    frame_length=N_FFT, hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0).reshape(1, -1)  # Handle NaNs in pitch
    
    # --- MFCC Features ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, 
                               n_mels=N_MELS, htk=HTK,
                               n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # --- List of raw features ---
    features = [
        ("MFCC", mfcc),
        ("ΔMFCC", delta1),
        ("ΔΔMFCC", delta2),
        ("ZCR", zcr),
        ("RMS", rms),
        ("SpectralFlatness", spectral_flatness),
        ("SpectralCentroid", spectral_centroid),
        ("SpectralBandwidth", spectral_bandwidth),
        ("SpectralRolloff", spectral_rolloff),
        ("F0", f0)
    ]
    
    # --- Compute statistics per feature type ---
    for name, feat in features:
        # Ensure 2D shape: (n_features, n_frames)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
            
        # Calculate statistics
        means = np.mean(feat, axis=1)
        medians = np.median(feat, axis=1)
        stds = np.std(feat, axis=1)
        skews = skew(feat, axis=1)
        kurts = kurtosis(feat, axis=1, fisher=False)
        
        # Replace NaNs in skew/kurtosis caused by zero variance
        skews = np.nan_to_num(skews)
        kurts = np.nan_to_num(kurts)
        
        # Stack stats and flatten
        stats = np.vstack([means, medians, stds, skews, kurts]).T
        all_features.append(stats.flatten())

    # --- Final feature vector ---
    return np.concatenate(all_features)

# --- Main Pipeline ---
if __name__ == "__main__":
    # 1. Load and preprocess data
    X, y = [], []
    for label, folder in DATA_DIRS.items():
        for fname in os.listdir(folder):
            feat = extract_audio_features(os.path.join(folder, fname))
            X.append(feat)
            y.append(label)
    X = np.vstack(X)
    y = np.array(y)
    
    # 2. Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_norm = scaler.fit_transform(X)
    '''
    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)

    # Plot
    plt.figure(figsize=(10, 6))
    for label in DATA_DIRS.keys():
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.6)
    plt.legend()
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Feature Space Visualization")
    plt.show() 
    '''
    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # 4. Cross-validation and evaluation
    for clf_name, clf in CLASSIFIERS.items():
        print(f"\n{'='*30}\n{clf_name}\n{'='*30}")
        
        # K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_accuracies = []
        
        print(f"\nK-Fold Cross-Validation (k={N_SPLITS}):")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            # Split data
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]
            
            # Train and predict
            clf.fit(X_fold_train, y_fold_train)
            y_pred = clf.predict(X_fold_val)
            
            # Calculate metrics
            acc = accuracy_score(y_fold_val, y_pred)
            fold_accuracies.append(acc)
            print(f"Fold {fold}: Accuracy = {acc:.4f}")
        
        # Print CV summary
        print(f"\nCV Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        
        # Final evaluation on test set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print("\nTest Set Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        with open(f'{clf_name.lower()}_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    '''    
    # Save scaler
    with open('pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)    
    '''    