"""
Voice Detection Model - AI vs Human Detection
Detects whether voice samples are AI-generated or human-spoken
"""

import numpy as np
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path


class VoiceDetector:
    """Machine learning model for detecting AI vs Human voices"""
    
    def __init__(self, model_path: str = None, language: str = "en"):
        """
        Initialize the voice detector
        
        Args:
            model_path: Path to pre-trained model
            language: Language code (en, ta, hi, ml, te)
        """
        self.language = language
        self.model_path = model_path or "ml/models/voice_detector_model.joblib"
        self.model = None
        self.scaler = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize or load the model"""
        # Try to load existing model first
        if os.path.exists(self.model_path):
            print(f"📂 Loading pre-trained model from {self.model_path}...")
            try:
                self.load_model(self.model_path)
                print("✅ Model loaded successfully!")
                return
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                print("🔄 Training new model...")
        else:
            print(f"📂 Model file not found: {self.model_path}")
            print("🔄 Training new model...")
        
        # If no model exists or loading failed, train a new one
        print("ℹ️  Creating improved voice detection model...")
        self._create_improved_model()
        
        # Save the newly trained model
        self.save_model(self.model_path)
    
    def _create_improved_model(self):
        """Create an IMPROVED trained model with better audio characteristics"""
        # Generate more realistic synthetic training data
        n_samples = 500  # Increased samples
        X_train = []
        y_train = []
        
        print(f"Generating {n_samples} training samples...")
        
        # ========== AI-GENERATED VOICE SAMPLES (label: 1) ==========
        # Characteristics: Too perfect, consistent, clean
        print("Generating AI samples...")
        for i in range(n_samples // 2):
            t = np.linspace(0, 3, 48000)  # 3 seconds at 16kHz
            
            # AI characteristics: Pure tones, perfect harmonics
            fundamental = 200 + np.random.randn() * 5
            
            # Perfect sine waves with EXACT harmonics
            audio = 0.5 * np.sin(2 * np.pi * fundamental * t)
            audio += 0.2 * np.sin(2 * np.pi * fundamental * 2 * t + np.pi/4)
            audio += 0.1 * np.sin(2 * np.pi * fundamental * 3 * t + np.pi/2)
            audio += 0.05 * np.sin(2 * np.pi * fundamental * 4 * t + 3*np.pi/4)
            
            # EXTREMELY low noise (too clean)
            audio += 0.001 * np.random.randn(len(t))
            
            # Very consistent amplitude
            if i % 2 == 0:
                vibrato = 0.03 * np.sin(2 * np.pi * 5.0 * t)
                audio *= (1.0 + vibrato)
            
            # Perfect normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            features = self.extract_features(audio, 16000)
            X_train.append(features)
            y_train.append(1)  # AI_GENERATED
        
        # ========== HUMAN VOICE SAMPLES (label: 0) ==========
        # Characteristics: Natural variation, imperfections
        print("Generating Human samples...")
        for i in range(n_samples // 2):
            t = np.linspace(0, 3, 48000)
            
            # Human characteristics: Natural pitch variation
            base_freq = 120 + np.random.rand() * 180
            
            # Time-varying frequency
            freq_variation = base_freq + 50 * np.sin(2 * np.pi * np.random.uniform(0.2, 3) * t)
            freq_variation += 40 * np.random.randn(len(t))
            
            # Complex waveform
            audio = np.zeros(len(t))
            for harmonic in range(1, 15):
                amp = 1.0 / (harmonic ** 0.8)
                freq = freq_variation * harmonic * (1 + np.random.randn() * 0.02)
                phase_noise = np.random.randn() * 0.3
                audio += amp * np.sin(2 * np.pi * freq * t / base_freq + phase_noise)
            
            # Natural amplitude variation
            speech_rhythm = 0.4 * np.abs(np.sin(2 * np.pi * np.random.uniform(1, 4) * t))
            speech_rhythm += 0.3 * np.abs(np.sin(2 * np.pi * np.random.uniform(5, 10) * t))
            
            envelope = 0.6 + speech_rhythm
            for _ in range(np.random.randint(2, 5)):
                start = np.random.randint(0, len(t) - 2000)
                envelope[start:start+2000] *= np.random.uniform(0.1, 0.5)
            
            audio *= envelope
            audio += 0.25 * np.random.randn(len(t))
            
            # Add occasional clicks
            if np.random.rand() > 0.5:
                click_pos = np.random.randint(1000, len(t) - 1000)
                audio[click_pos:click_pos+50] += 0.5 * np.random.randn(50)
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            features = self.extract_features(audio, 16000)
            X_train.append(features)
            y_train.append(0)  # HUMAN
        
        # ========== TRAIN THE MODEL ==========
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training data: X={X_train.shape}, y={y_train.shape}")
        print(f"  AI samples: {np.sum(y_train == 1)}")
        print(f"  Human samples: {np.sum(y_train == 0)}")
        
        # Create and train model
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Use better model parameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Check training accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        # Print feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop 5 feature importances:")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = ['ZCR', 'Energy', 'EnergyVar', 'SpectralCentroid', 
                           'SpectralSpread', 'MFCC_Mean', 'MFCC_Std', 'Periodicity']
            for i in range(min(5, len(feature_names))):
                print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract audio features for detection
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Feature vector as numpy array
        """
        # Normalize audio
        if len(audio_data) == 0:
            return np.zeros(8)
        
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        features = []
        
        # 1. Zero Crossing Rate
        zcr = self._calculate_zcr(audio_data)
        features.append(zcr)
        
        # 2. Energy
        energy = np.sqrt(np.mean(audio_data ** 2))
        features.append(energy)
        
        # 3. Energy variance
        frame_size = min(2048, len(audio_data) // 8)
        if frame_size > 100:
            energy_frames = []
            for i in range(0, len(audio_data) - frame_size, frame_size // 2):
                e = np.sqrt(np.mean(audio_data[i:i+frame_size] ** 2))
                energy_frames.append(e)
            energy_variance = np.var(energy_frames) if energy_frames else 0
        else:
            energy_variance = 0
        features.append(energy_variance)
        
        # 4. Spectral Centroid
        spectral_centroid = self._calculate_spectral_centroid(audio_data, sample_rate)
        features.append(spectral_centroid)
        
        # 5. Spectral spread
        spectral_spread = self._calculate_spectral_spread(audio_data, sample_rate)
        features.append(spectral_spread)
        
        # 6. MFCC-like mean
        mfcc_mean = self._calculate_mfcc_like(audio_data, sample_rate)
        features.append(mfcc_mean)
        
        # 7. MFCC-like std
        mfcc_std = self._calculate_mfcc_std(audio_data, sample_rate)
        features.append(mfcc_std)
        
        # 8. Periodicity
        periodicity = self._calculate_periodicity(audio_data, sample_rate)
        features.append(periodicity)
        
        return np.array(features)
    
    def _calculate_zcr(self, audio_data: np.ndarray) -> float:
        """Calculate Zero Crossing Rate"""
        if len(audio_data) < 2:
            return 0.0
        sign_changes = np.sum(np.abs(np.diff(np.sign(audio_data))))
        return float(sign_changes / (2 * len(audio_data)))
    
    def _calculate_spectral_centroid(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate Spectral Centroid"""
        if len(audio_data) < 10:
            return 0.5
        
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]
        
        if np.sum(magnitude) == 0:
            return 0.5
        
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        return float(min(centroid / (sample_rate / 2), 1.0))
    
    def _calculate_spectral_spread(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate Spectral Spread"""
        if len(audio_data) < 10:
            return 0.0
        
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        spread = np.sqrt(np.sum(magnitude * (freqs - centroid) ** 2) / np.sum(magnitude))
        return float(min(spread / (sample_rate / 2), 1.0))
    
    def _calculate_mfcc_like(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate MFCC-like mean feature"""
        if len(audio_data) < 10:
            return 0.5
        
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        if len(magnitude) == 0:
            return 0.5
        
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return float(np.mean(magnitude))
    
    def _calculate_mfcc_std(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate MFCC-like standard deviation"""
        if len(audio_data) < 10:
            return 0.1
        
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        if len(magnitude) == 0:
            return 0.1
        
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return float(np.std(magnitude))
    
    def _calculate_periodicity(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate periodicity"""
        if len(audio_data) < 1000:
            return 0.5
        
        segment_length = min(4096, len(audio_data))
        segment = audio_data[:segment_length]
        
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) < 2:
            return 0.5
        
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        if len(autocorr) > 100:
            max_peak = np.max(autocorr[20:100])
            return float(min(max_peak, 1.0))
        
        return 0.5
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict whether audio is AI-generated or Human
        
        Args:
            features: Feature vector from extract_features
            
        Returns:
            Tuple of (result: str, confidence: float)
            result: "AI_GENERATED" or "HUMAN"
            confidence: Score between 0.0 and 1.0
        """
        if self.model is None or self.scaler is None:
            print("⚠️  Model not initialized, returning default")
            return "HUMAN", 0.5
        
        # Scale features
        features_reshaped = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_reshaped)
        
        # Get prediction and confidence
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Map prediction to result
        if prediction == 1:
            result = "AI_GENERATED"
            confidence = probabilities[1]
        else:
            result = "HUMAN"
            confidence = probabilities[0]
        
        # Ensure confidence is reasonable
        confidence = max(0.0, min(1.0, confidence))
        
        return result, float(confidence)
    
    def save_model(self, path: str):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            print(f"✅ Model loaded from {path}")
        except Exception as e:
            print(f"❌ Failed to load model from {path}: {e}")
            raise