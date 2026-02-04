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
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the model"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except:
                self._create_default_model()
        else:
            self._create_default_model()
    
    def _create_default_model(self):
        """Create a default trained model with synthetic audio data"""
        # Generate synthetic training data from actual audio
        n_samples = 300
        X_train = []
        y_train = []
        
        # Generate AI-generated voice samples (label: 1)
        # EXTREMELY CLEAN: Single pure frequency, no variation
        for i in range(n_samples // 2):
            t = np.linspace(0, 2, 32000)  # 2 seconds at 16kHz
            
            # PURE TONE: Single frequency, mathematically perfect
            audio = np.sin(2 * np.pi * 220 * t)
            
            # NO HARMONICS - just pure frequency
            # ALMOST NO NOISE
            audio += 0.001 * np.random.randn(len(t))
            
            features = self.extract_features(audio, 16000)
            X_train.append(features)
            y_train.append(1)  # AI_GENERATED
        
        # Generate human voice samples (label: 0)
        # EXTREMELY NATURAL: Lots of variation, noise, speech-like
        for i in range(n_samples // 2):
            t = np.linspace(0, 2, 32000)
            
            # Random pitch variation (natural speech)
            base_freq = np.random.uniform(100, 250)
            freq_var = base_freq + 100 * np.sin(2 * np.pi * np.random.uniform(1, 5) * t)
            freq_var += 50 * np.random.randn(len(t))  # Random frequency jitter
            
            # Complex audio with many harmonics
            audio = np.zeros(len(t))
            for harmonic in range(1, 10):
                amp = 1.0 / (harmonic ** 1.5)  # Natural harmonic decay
                audio += amp * np.sin(2 * np.pi * (freq_var * harmonic) * t / base_freq)
            
            # LOTS of noise (breath, natural artifacts)
            audio += 0.3 * np.random.randn(len(t))
            
            # Highly variable amplitude (speech rhythm)
            envelope = np.abs(np.sin(2 * np.pi * np.random.uniform(1, 3) * t))
            envelope = envelope ** 0.5  # Make it more speech-like
            audio *= envelope
            
            # Add random "silence" sections
            for _ in range(np.random.randint(2, 5)):
                start = np.random.randint(0, len(t) - 3000)
                audio[start:start + 3000] *= np.random.uniform(0, 0.3)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-6)
            
            features = self.extract_features(audio, 16000)
            X_train.append(features)
            y_train.append(0)  # HUMAN
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"AI samples: {np.sum(y_train == 1)}, Human samples: {np.sum(y_train == 0)}")
        
        # Create and train model
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,  # Very deep to capture differences
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Print feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("Feature importances:")
            for i, imp in enumerate(self.model.feature_importances_):
                print(f"  Feature {i}: {imp:.4f}")
    
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
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        features = []
        
        # 1. Zero Crossing Rate - varies more in human speech
        zcr = self._calculate_zcr(audio_data)
        features.append(zcr)
        
        # 2. Energy - human speech has more variation
        energy = np.sqrt(np.mean(audio_data ** 2))
        features.append(energy)
        
        # 3. Energy variance - human speech has more variation
        frame_size = min(1024, len(audio_data) // 4)
        energy_frames = []
        for i in range(0, len(audio_data) - frame_size, frame_size // 2):
            e = np.sqrt(np.mean(audio_data[i:i+frame_size] ** 2))
            energy_frames.append(e)
        energy_variance = np.var(energy_frames) if energy_frames else 0
        features.append(energy_variance)
        
        # 4. Spectral Centroid - human voices have lower centroid
        spectral_centroid = self._calculate_spectral_centroid(audio_data, sample_rate)
        features.append(spectral_centroid)
        
        # 5. Spectral spread - human voices have more spread
        spectral_spread = self._calculate_spectral_spread(audio_data, sample_rate)
        features.append(spectral_spread)
        
        # 6. MFCC-like mean
        mfcc_mean = self._calculate_mfcc_like(audio_data, sample_rate)
        features.append(mfcc_mean)
        
        # 7. MFCC-like std - human voices have higher variation
        mfcc_std = self._calculate_mfcc_std(audio_data, sample_rate)
        features.append(mfcc_std)
        
        # 8. Periodicity - AI voices are more periodic
        periodicity = self._calculate_periodicity(audio_data, sample_rate)
        features.append(periodicity)
        
        return np.array(features)
    
    def _calculate_zcr(self, audio_data: np.ndarray) -> float:
        """Calculate Zero Crossing Rate - human speech has higher ZCR"""
        if len(audio_data) < 2:
            return 0.0
        zcr = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        return float(zcr)
    
    def _calculate_spectral_centroid(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate Spectral Centroid - normalized to 0-1 range"""
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]
        
        if np.sum(magnitude) == 0:
            return 0.5
        
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        # Normalize to 0-1 range (max freq = sample_rate/2)
        return float(min(centroid / (sample_rate / 2), 1.0))
    
    def _calculate_spectral_spread(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate Spectral Spread - human voices have more spread"""
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        # Normalize magnitude
        mag_norm = magnitude / np.sum(magnitude)
        
        # Calculate centroid
        centroid = np.sum(freqs * mag_norm)
        
        # Calculate spread (standard deviation of frequency)
        spread = np.sqrt(np.sum(mag_norm * (freqs - centroid) ** 2))
        
        # Normalize to 0-1
        return float(min(spread / (sample_rate / 2), 1.0))
    
    def _calculate_mfcc_like(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate MFCC-like mean feature"""
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Normalize
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return float(np.mean(magnitude))
    
    def _calculate_mfcc_std(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate MFCC-like standard deviation"""
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return float(np.std(magnitude))
    
    def _calculate_periodicity(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate periodicity - AI voices are more periodic"""
        # Simple autocorrelation-based periodicity
        frame_size = min(4096, len(audio_data) // 2)
        if frame_size < 256:
            return 0.5
        
        frame = audio_data[:frame_size]
        
        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        # Find periodicity peak (excluding lag 0)
        if len(autocorr) > 100:
            max_autocorr = np.max(autocorr[10:100])
            return float(min(max_autocorr, 1.0))
        
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
            return "HUMAN", 0.5
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
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
        
        return result, float(confidence)
    
    def save_model(self, path: str):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
    
    def load_model(self, path: str):
        """Load model from file"""
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
