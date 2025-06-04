import pickle
import librosa
import numpy as np
import sounddevice as sd
#from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew

# --- CONFIG --- #
SR = 44100
N_MFCC = 24
N_MELS = 128
FRAME_LEN = 0.030
FRAME_HOP = 0.015
N_FFT = int(FRAME_LEN * SR)
HOP_LENGTH = int(FRAME_HOP * SR)
LIFTER = 22
FMIN = 100
FMAX = 22050
HTK = True

model_filename = 'svm_model.pkl'
scaler_filename = 'scaler.pkl'
#pca_filename = 'pca_model.pkl'  

try:
    with open(model_filename, 'rb') as f:
        classifier = pickle.load(f)
    print(f"Model loaded successfully from {model_filename}")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
    exit()

try:
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded successfully from {scaler_filename}")
except FileNotFoundError:
    print(f"Error: Scaler file '{scaler_filename}' not found.")
    exit()
'''    
try:
    with open(pca_filename, 'rb') as f:
        pca = pickle.load(f)
    print(f"PCA model loaded successfully from {pca_filename}")
except FileNotFoundError:
    print(f"Error: PCA file '{pca_filename}' not found.")
    exit()    
'''

def extract_audio_features(y):
    sr = SR
    
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

def record_audio(duration=5):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def predict_audio(duration=5):
    audio_data = record_audio(duration=duration)
    
    if np.max(np.abs(audio_data)) < 0.01:
        print("Warning: Detected silence or very low volume in recording.")
        return

    feature_vector = extract_audio_features(audio_data)
    print("Feature vector min/max before scaling:", np.min(feature_vector), np.max(feature_vector))

    try:
        feature_vector_norm = scaler.transform([feature_vector])
        #feature_vector_norm_pca = pca.transform(feature_vector_norm)
    except Exception as e:
        print(f"Error during feature normalization: {e}")
        return

    print("Feature vector min/max after scaling:", np.min(feature_vector_norm), np.max(feature_vector_norm))
    print("Live Feature Shape:", feature_vector.shape)

    try:
        prediction = classifier.predict(feature_vector_norm)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    label_map = {"cat": "Cat", "dog": "Dog", "bird": "Bird", "cow": "Cow", "frog":"Frog"}
    predicted_label = label_map.get(prediction[0].lower(), "Unknown")
    print(f"Predicted class: {predicted_label}")
    (f"Predicted Class ID: {prediction[0]}")

    if hasattr(classifier, "decision_function"):
        decision_scores = classifier.decision_function(feature_vector_norm)
        print("Decision scores:", decision_scores)

while(1):
    if __name__ == "__main__":
        predict_audio(duration=5)
