"""
Audio capture system for Sound Analyzer.
Handles microphone input and audio file playback.
"""
import numpy as np
import pyaudio
import wave
import threading
import logging
import time
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)

class AudioCapture:
    """
    Handles audio capture from microphone and audio files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio capture system.
        
        Args:
            config: Audio configuration settings
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 1024)
        self.sample_rate = config.get('sample_rate', 44100)
        self.channels = config.get('channels', 1)
        self.format = pyaudio.paFloat32
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.record_thread = None
        self.callbacks = []
        
        # Get available devices
        self.devices = self._get_available_devices()
        
        # Default device
        self.device_index = config.get('device_index', None)
        if self.device_index is None:
            self.device_index = self.audio.get_default_input_device_info()['index']
    
    def _get_available_devices(self):
        """
        Get a list of available audio input devices.
        
        Returns:
            List of input device info dictionaries
        """
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append(device_info)
            except Exception as e:
                logger.error(f"Error getting device info for index {i}: {e}")
        
        return devices
    
    def get_device_list(self):
        """
        Get a list of available input devices.
        
        Returns:
            List of (device_index, device_name) tuples
        """
        return [(d['index'], d['name']) for d in self.devices]
    
    def set_device(self, device_index):
        """
        Set the audio input device.
        
        Args:
            device_index: Index of the audio device to use
        """
        if self.is_recording:
            self.stop_recording()
        
        self.device_index = device_index
        logger.info(f"Set audio input device to index {device_index}")
    
    def start_recording(self):
        """
        Start recording audio from the selected input device.
        """
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            logger.info("Started audio recording")
            
        except Exception as e:
            logger.error(f"Error starting audio recording: {e}")
            self.is_recording = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio stream.
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Stream status
            
        Returns:
            (None, pyaudio.paContinue) to continue the stream
        """
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert byte data to numpy array (float32)
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Notify all registered callbacks
        for callback in self.callbacks:
            try:
                callback(audio_data, self.sample_rate)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
        
        return (None, pyaudio.paContinue)
    
    def stop_recording(self):
        """
        Stop recording audio.
        """
        if not self.is_recording:
            return
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.is_recording = False
        logger.info("Stopped audio recording")
    
    def register_callback(self, callback: Callable[[np.ndarray, int], None]):
        """
        Register a callback function to receive audio data.
        
        Args:
            callback: Function that takes (audio_data, sample_rate) parameters
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[np.ndarray, int], None]):
        """
        Unregister a callback function.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def load_audio_file(self, file_path):
        """
        Load audio data from a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            with wave.open(file_path, 'rb') as wf:
                # Get file properties
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # Read all frames
                frames = wf.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 2:  # 16-bit audio
                    dtype = np.int16
                elif sample_width == 4:  # 32-bit audio
                    dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                audio_data = np.frombuffer(frames, dtype=dtype)
                
                # Convert to float32 normalized to [-1.0, 1.0]
                audio_data = audio_data.astype(np.float32)
                if dtype == np.int16:
                    audio_data /= 32768.0
                elif dtype == np.int32:
                    audio_data /= 2147483648.0
                
                # If stereo, convert to mono by averaging channels
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                return audio_data, sample_rate
                
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None, None
    
    def play_audio_file(self, file_path, callback=None):
        """
        Play an audio file and process it with the registered callbacks.
        
        Args:
            file_path: Path to the audio file
            callback: Optional callback for playback progress
            
        Returns:
            True if playback started successfully, False otherwise
        """
        audio_data, sample_rate = self.load_audio_file(file_path)
        if audio_data is None:
            return False
        
        # Start a thread to simulate real-time processing
        self.playback_thread = threading.Thread(
            target=self._playback_thread,
            args=(audio_data, sample_rate, callback)
        )
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
        return True
    
    def _playback_thread(self, audio_data, sample_rate, progress_callback=None):
        """
        Thread function to process audio file data in chunks.
        
        Args:
            audio_data: The audio data as numpy array
            sample_rate: The sample rate of the audio
            progress_callback: Optional callback for playback progress
        """
        # Calculate chunk size based on original sample rate
        chunk_samples = int(self.chunk_size * (sample_rate / self.sample_rate))
        
        # Process audio in chunks
        num_chunks = len(audio_data) // chunk_samples
        for i in range(num_chunks):
            if not self.is_recording:  # Check if playback was stopped
                break
            
            # Get the current chunk
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = audio_data[start:end]
            
            # Process the chunk with all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(chunk, sample_rate)
                except Exception as e:
                    logger.error(f"Error in audio playback callback: {e}")
            
            # Update progress if callback provided
            if progress_callback:
                progress = (i + 1) / num_chunks
                progress_callback(progress)
            
            # Sleep to simulate real-time processing
            time.sleep(self.chunk_size / self.sample_rate)
    
    def cleanup(self):
        """
        Clean up resources.
        """
        self.stop_recording()
        self.audio.terminate()
        logger.info("Audio capture cleaned up")
