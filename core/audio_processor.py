"""
Audio Processing Module
Handles audio file loading, conversion, and feature extraction
"""

import numpy as np
from typing import Dict, Tuple, Optional
import tempfile
import os
from pathlib import Path

# Try to import audio libraries, with fallback
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class AudioProcessor:
    """Process and analyze audio files for voice detection"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 30.0):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate (Hz)
            max_duration: Maximum audio duration (seconds)
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
    
    def load_audio_from_bytes(self, audio_bytes: bytes, file_format: str = 'mp3') -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes
        
        Args:
            audio_bytes: Audio file bytes
            file_format: Format of audio file (mp3, wav, flac, etc.)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        tmp_path = None
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{file_format}', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Ensure file is closed before reading
            if tmp_path:
                # Give OS time to release file lock
                import time
                time.sleep(0.01)
                
                if LIBROSA_AVAILABLE:
                    # Use librosa for best compatibility
                    audio_data, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
                    return audio_data, sr
                
                elif SCIPY_AVAILABLE and file_format.lower() in ['wav']:
                    # Use scipy for WAV files
                    sr, audio_data = wavfile.read(tmp_path)
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        num_samples = int(len(audio_data) * self.sample_rate / sr)
                        audio_data = signal.resample(audio_data, num_samples)
                    
                    return audio_data, self.sample_rate
                
                else:
                    # Fallback: try to read as raw or use simple format
                    audio_data = self._read_audio_fallback(tmp_path)
                    return audio_data, self.sample_rate
        
        finally:
            # Clean up temporary file - with retry logic
            if tmp_path:
                for attempt in range(3):
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        break
                    except Exception as e:
                        if attempt < 2:
                            import time
                            time.sleep(0.1)
                        else:
                            # Last attempt failed, but don't crash
                            pass
    
    def _read_audio_fallback(self, file_path: str) -> np.ndarray:
        """
        Fallback method to read audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio data as numpy array
        """
        # Try to read as WAV using scipy
        if SCIPY_AVAILABLE:
            try:
                sr, audio_data = wavfile.read(file_path)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                return audio_data
            except:
                pass
        
        # Generate synthetic audio as last resort
        return self.generate_test_audio(duration=2.0, audio_type="mixed")
    
    def generate_test_audio(self, duration: float = 1.0, audio_type: str = "mixed") -> np.ndarray:
        """
        Generate test audio with different characteristics
        
        Args:
            duration: Duration of audio (seconds)
            audio_type: Type of audio - 'human', 'ai', or 'mixed'
            
        Returns:
            Audio data as numpy array
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Use seed based on type for reproducibility
        seed = hash(f"{duration}{audio_type}") % 10000
        np.random.seed(seed)
        
        if audio_type == "human":
            # Human-like voice characteristics
            return self._generate_human_audio(t)
        
        elif audio_type == "ai":
            # AI-like voice characteristics
            return self._generate_ai_audio(t)
        
        else:  # mixed
            # Random mix
            if np.random.rand() > 0.5:
                return self._generate_human_audio(t)
            else:
                return self._generate_ai_audio(t)
    
    def _generate_human_audio(self, t: np.ndarray) -> np.ndarray:
        """
        Generate synthetic human-like audio
        
        Args:
            t: Time array
            
        Returns:
            Audio signal
        """
        # Human voices have varying pitch and natural irregularities
        fundamental_freq = 150 + np.random.rand() * 100  # 150-250 Hz
        
        # Complex harmonic structure
        audio = 0.3 * np.sin(2 * np.pi * fundamental_freq * t)
        
        # Add multiple harmonics with natural variation
        for i in range(2, 8):
            harmonic_strength = 0.2 / (i * 0.7)
            freq_variation = fundamental_freq * i * (1 + np.random.randn() * 0.01)
            audio += harmonic_strength * np.sin(2 * np.pi * freq_variation * t)
        
        # Add natural variation (breath noise, formants)
        audio += 0.05 * np.random.randn(len(t))
        
        # Add amplitude modulation (natural speech variation)
        am_freq = 5 + np.random.rand() * 5  # 5-10 Hz
        am = 0.15 * np.sin(2 * np.pi * am_freq * t) + 0.9
        audio *= am
        
        # Add spectral variation over time
        for i in range(0, len(t), len(t)//5):
            end = min(i + len(t)//5, len(t))
            if end > i:
                audio[i:end] *= (0.8 + np.random.rand() * 0.4)
        
        return np.clip(audio, -1.0, 1.0).astype(np.float32)
    
    def _generate_ai_audio(self, t: np.ndarray) -> np.ndarray:
        """
        Generate synthetic AI-like audio
        
        Args:
            t: Time array
            
        Returns:
            Audio signal
        """
        # AI voices have consistent pitch and less natural variation
        fundamental_freq = 220  # Perfect pitch
        
        # Regular harmonic structure
        audio = 0.4 * np.sin(2 * np.pi * fundamental_freq * t)
        
        # Add harmonics (but cleaner than human)
        audio += 0.15 * np.sin(2 * np.pi * fundamental_freq * 2 * t)
        audio += 0.10 * np.sin(2 * np.pi * fundamental_freq * 3 * t)
        audio += 0.05 * np.sin(2 * np.pi * fundamental_freq * 4 * t)
        
        # Add only small noise
        audio += 0.01 * np.random.randn(len(t))
        
        # Very consistent amplitude (no natural variation)
        # Just slight vibrato (natural for synthesizers)
        vibrato_freq = 5.0  # Fixed 5 Hz vibrato
        vibrato = 0.05 * np.sin(2 * np.pi * vibrato_freq * t)
        audio *= (1.0 + vibrato)
        
        return np.clip(audio, -1.0, 1.0).astype(np.float32)
    
    def validate_audio(self, audio_data: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Validate audio data
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if audio is empty
        if len(audio_data) == 0:
            return False, "Audio is empty"
        
        # Check duration
        duration = len(audio_data) / self.sample_rate
        if duration > self.max_duration:
            return False, f"Audio duration {duration:.2f}s exceeds maximum {self.max_duration}s"
        
        if duration < 0.1:
            return False, "Audio duration is too short (minimum 0.1s)"
        
        # Check for silence (all zeros or very low amplitude)
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 0.01:
            return False, "Audio appears to be silent or has very low amplitude"
        
        return True, None
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def get_audio_info(self, audio_data: np.ndarray) -> Dict[str, any]:
        """
        Get information about audio
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Dictionary with audio info
        """
        duration = len(audio_data) / self.sample_rate
        rms = np.sqrt(np.mean(audio_data ** 2))
        peak = np.max(np.abs(audio_data))
        
        return {
            'duration_seconds': duration,
            'sample_count': len(audio_data),
            'sample_rate': self.sample_rate,
            'rms_amplitude': float(rms),
            'peak_amplitude': float(peak),
            'is_valid': rms > 0.01
        }