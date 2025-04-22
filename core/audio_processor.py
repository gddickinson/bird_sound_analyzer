"""
Audio processing system for Sound Analyzer.
Handles signal processing and feature extraction.
"""
import numpy as np
import scipy.signal as signal
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio processing and feature extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio processor.
        
        Args:
            config: Configuration settings for audio processing
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 44100)
        self.chunk_size = config.get('chunk_size', 1024)
        
        # Spectrogram settings
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.window = config.get('window', 'hann')
        
        # Feature extraction settings
        self.min_freq = config.get('min_freq', 0)
        self.max_freq = config.get('max_freq', self.sample_rate // 2)
        
        # Buffer for audio history
        self.history_seconds = config.get('history_seconds', 5)
        self.max_history_samples = int(self.history_seconds * self.sample_rate)
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        
        # Callbacks for processing results
        self.callbacks = []
    
    def process_audio(self, audio_chunk: np.ndarray, sample_rate: int):
        """
        Process an audio chunk and update the buffer.
        
        Args:
            audio_chunk: The audio data as numpy array
            sample_rate: The sample rate of the audio
            
        Returns:
            Dict containing processing results
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_chunk = self._resample(audio_chunk, sample_rate)
        
        # Update the audio buffer
        self._update_buffer(audio_chunk)
        
        # Calculate features
        results = {
            'waveform': audio_chunk,
            'rms': self._calculate_rms(audio_chunk),
        }
        
        # Calculate spectrogram if we have enough data
        if len(self.audio_buffer) >= self.n_fft:
            # Use the most recent data for the spectrogram
            data_for_spectrogram = self.audio_buffer[-self.n_fft:]
            
            # Calculate spectrogram
            frequencies, times, spectrogram = self._calculate_spectrogram(data_for_spectrogram)
            
            results.update({
                'spectrogram': spectrogram,
                'spectrogram_frequencies': frequencies,
                'spectrogram_times': times,
            })
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(results)
            except Exception as e:
                logger.error(f"Error in audio processor callback: {e}")
        
        return results
    
    def _update_buffer(self, audio_chunk: np.ndarray):
        """
        Update the audio buffer with new data.
        
        Args:
            audio_chunk: The new audio data to add
        """
        # Append new data
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        
        # Trim to max history size
        if len(self.audio_buffer) > self.max_history_samples:
            self.audio_buffer = self.audio_buffer[-self.max_history_samples:]
    
    def _resample(self, audio_chunk: np.ndarray, original_rate: int) -> np.ndarray:
        """
        Resample audio to the target sample rate.
        
        Args:
            audio_chunk: The audio data to resample
            original_rate: The original sample rate
            
        Returns:
            Resampled audio data
        """
        # Calculate resampling ratio
        ratio = self.sample_rate / original_rate
        
        # Calculate new length
        new_length = int(len(audio_chunk) * ratio)
        
        # Resample using scipy.signal.resample
        resampled = signal.resample(audio_chunk, new_length)
        
        return resampled
    
    def _calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """
        Calculate the Root Mean Square (RMS) of audio data.
        
        Args:
            audio_chunk: The audio data
            
        Returns:
            RMS value
        """
        return np.sqrt(np.mean(np.square(audio_chunk)))
    
    def _calculate_spectrogram(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the spectrogram of audio data.
        
        Args:
            audio_data: The audio data
            
        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        # Create the window
        window_func = getattr(signal.windows, self.window)(self.n_fft)
        
        # Calculate the Short-Time Fourier Transform (STFT)
        frequencies, times, stft = signal.stft(
            audio_data,
            fs=self.sample_rate,
            window=window_func,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            return_onesided=True
        )
        
        # Convert to magnitude spectrogram
        spectrogram = np.abs(stft)
        
        # Convert to dB scale (log scale)
        spectrogram = 20 * np.log10(spectrogram + 1e-10)  # Add small epsilon to avoid log(0)
        
        return frequencies, times, spectrogram
    
    def get_fft(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the FFT of audio data.
        
        Args:
            audio_data: The audio data
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        # Calculate FFT
        fft = np.fft.rfft(audio_data * signal.windows.hann(len(audio_data)))
        
        # Calculate magnitude
        magnitude = np.abs(fft)
        
        # Calculate frequency array
        freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        
        return freqs, magnitude
    
    def register_callback(self, callback):
        """
        Register a callback function to receive processing results.
        
        Args:
            callback: Function that takes a results dictionary
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback):
        """
        Unregister a callback function.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
