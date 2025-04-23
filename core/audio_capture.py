"""
Enhanced audio capture system for Sound Analyzer.
Handles microphone input and audio file playback with improved controls.
Prevents feedback by ensuring recording and playback are mutually exclusive.
"""
import numpy as np
import pyaudio
import wave
import threading
import logging
import time
from typing import Callable, Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

class AudioCapture:
    """
    Handles audio capture from microphone and audio files.
    Provides enhanced playback controls including pause/resume and seeking.
    Ensures recording and playback are mutually exclusive to prevent feedback.
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
        self.input_stream = None
        self.output_stream = None
        self.is_recording = False
        self.record_thread = None
        self.callbacks = []

        # Speaker output settings
        self.enable_speaker_output = config.get('enable_speaker_output', True)
        self.output_volume = config.get('output_volume', 1.0)  # Range: 0.0 to 1.0
        self.output_device_index = config.get('output_device_index', None)

        # Get available devices
        self.input_devices = self._get_available_input_devices()
        self.output_devices = self._get_available_output_devices()

        # Default input device
        self.input_device_index = config.get('input_device_index', None)
        if self.input_device_index is None:
            default_info = self.audio.get_default_input_device_info()
            self.input_device_index = default_info['index']

        # Default output device
        if self.output_device_index is None:
            try:
                default_info = self.audio.get_default_output_device_info()
                self.output_device_index = default_info['index']
            except Exception as e:
                logger.warning(f"Could not get default output device: {e}")
                self.output_device_index = None

        # Playback control
        self.playback_thread = None
        self.playback_paused = threading.Event()  # Event to signal pause/resume
        self.playback_stop = threading.Event()    # Event to signal stop
        self.playback_position = 0                # Current playback position in samples
        self.playback_data = None                 # Current audio data being played
        self.playback_sample_rate = 0             # Sample rate of current playback
        self.playback_callbacks = []              # Callbacks for playback progress
        self.current_file = None                  # Currently loaded file

    def _get_available_input_devices(self):
        """
        Get a list of available audio input devices.

        Returns:
            List of input device info dictionaries
        """
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    devices.append(device_info)
            except Exception as e:
                logger.error(f"Error getting input device info for index {i}: {e}")

        return devices

    def _get_available_output_devices(self):
        """
        Get a list of available audio output devices.

        Returns:
            List of output device info dictionaries
        """
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info.get('maxOutputChannels') > 0:
                    devices.append(device_info)
            except Exception as e:
                logger.error(f"Error getting output device info for index {i}: {e}")

        return devices

    def get_input_device_list(self):
        """
        Get a list of available input devices.

        Returns:
            List of (device_index, device_name) tuples
        """
        return [(d['index'], d['name']) for d in self.input_devices]

    def get_output_device_list(self):
        """
        Get a list of available output devices.

        Returns:
            List of (device_index, device_name) tuples
        """
        return [(d['index'], d['name']) for d in self.output_devices]

    def set_input_device(self, device_index):
        """
        Set the audio input device.

        Args:
            device_index: Index of the audio device to use
        """
        if self.is_recording:
            self.stop_recording()

        self.input_device_index = device_index
        logger.info(f"Set audio input device to index {device_index}")

    def set_output_device(self, device_index):
        """
        Set the audio output device.

        Args:
            device_index: Index of the audio device to use
        """
        # Close any existing output stream
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None

        self.output_device_index = device_index
        logger.info(f"Set audio output device to index {device_index}")

        # Restart playback if needed
        if self.is_playing() and not self.is_playback_paused():
            # Store current position
            current_pos = self.get_playback_progress()

            # Stop and restart playback
            self.stop_playback()
            if self.current_file:
                self.play_audio_file(self.current_file)
                # Seek to previous position
                if current_pos:
                    self.seek_playback(current_pos)

    def set_speaker_output(self, enable):
        """
        Enable or disable speaker output.

        Args:
            enable: True to enable speaker output, False to disable
        """
        self.enable_speaker_output = enable
        logger.info(f"Speaker output {'enabled' if enable else 'disabled'}")

    def set_output_volume(self, volume):
        """
        Set the output volume.

        Args:
            volume: Volume level from 0.0 (mute) to 1.0 (full volume)
        """
        self.output_volume = max(0.0, min(1.0, volume))
        logger.info(f"Output volume set to {self.output_volume:.2f}")

    def start_recording(self):
        """
        Start recording audio from the selected input device.

        Returns:
            True if recording started successfully, False otherwise
        """
        # Check if playback is active - stop it to prevent feedback
        if self.is_playing():
            logger.info("Stopping playback before recording to prevent feedback")
            self.stop_playback()

        if self.is_recording:
            logger.warning("Already recording")
            return False

        try:
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )

            self.is_recording = True
            logger.info("Started audio recording")
            return True

        except Exception as e:
            logger.error(f"Error starting audio recording: {e}")
            self.is_recording = False
            return False

    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio input stream.

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

        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

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
            # Store the current file path for reference
            self.current_file = file_path

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

                logger.info(f"Loaded audio file: {file_path}, {len(audio_data)} samples at {sample_rate}Hz")
                return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return None, None

    def play_audio_file(self, file_path, progress_callback=None):
        """
        Play an audio file and process it with the registered callbacks.

        Args:
            file_path: Path to the audio file
            progress_callback: Optional callback for playback progress

        Returns:
            True if playback started successfully, False otherwise
        """
        # Check if recording is active - stop it to prevent feedback
        if self.is_recording:
            logger.info("Stopping recording before playback")
            self.stop_recording()

        # Load the audio file
        audio_data, sample_rate = self.load_audio_file(file_path)
        if audio_data is None:
            return False

        # Stop any existing playback
        self.stop_playback()

        # Store audio data for playback controls
        self.playback_data = audio_data
        self.playback_sample_rate = sample_rate
        self.playback_position = 0
        self.current_file = file_path

        # Reset control events
        self.playback_paused.clear()  # Not paused
        self.playback_stop.clear()    # Not stopped

        # Register progress callback if provided
        if progress_callback and progress_callback not in self.playback_callbacks:
            self.playback_callbacks.append(progress_callback)

        # Initialize output stream for speaker output if enabled
        if self.enable_speaker_output and self.output_device_index is not None:
            try:
                if self.output_stream:
                    self.output_stream.stop_stream()
                    self.output_stream.close()

                self.output_stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=sample_rate,  # Use the file's sample rate
                    output=True,
                    output_device_index=self.output_device_index,
                    frames_per_buffer=self.chunk_size
                )
                logger.info(f"Initialized output stream for playback on device {self.output_device_index}")
            except Exception as e:
                logger.error(f"Error initializing output stream for playback: {e}")
                self.output_stream = None

        # Start a thread to simulate real-time processing
        self.playback_thread = threading.Thread(
            target=self._playback_thread,
            args=(audio_data, sample_rate)
        )
        self.playback_thread.daemon = True
        self.playback_thread.start()

        logger.info(f"Started playback of {file_path}")
        return True

    def _playback_thread(self, audio_data, sample_rate):
        """
        Thread function to process audio file data in chunks.

        Args:
            audio_data: The audio data as numpy array
            sample_rate: The sample rate of the audio
        """
        # Calculate chunk size based on original sample rate
        chunk_samples = int(self.chunk_size * (sample_rate / self.sample_rate))

        # Process audio in chunks
        num_chunks = len(audio_data) // chunk_samples

        # Start from current position
        current_chunk = self.playback_position // chunk_samples

        # Loop until the end of the file or until stopped
        while current_chunk < num_chunks and not self.playback_stop.is_set():
            # Check if playback is paused
            if self.playback_paused.is_set():
                time.sleep(0.1)  # Sleep briefly to reduce CPU usage
                continue

            # Get the current chunk
            start = current_chunk * chunk_samples
            end = min(start + chunk_samples, len(audio_data))  # Ensure we don't go past the end
            chunk = audio_data[start:end]

            # Update playback position
            self.playback_position = start

            # Send to output stream if enabled
            if self.enable_speaker_output and self.output_stream:
                try:
                    # Apply volume
                    output_data = chunk * self.output_volume
                    self.output_stream.write(output_data.tobytes())
                except Exception as e:
                    logger.error(f"Error writing to output stream during playback: {e}")

            # Process the chunk with all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(chunk, sample_rate)
                except Exception as e:
                    logger.error(f"Error in audio playback callback: {e}")

            # Update progress if callbacks provided
            progress = start / len(audio_data)
            for callback in self.playback_callbacks:
                try:
                    callback(progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

            # Calculate sleep time based on chunk size and sample rate
            # This ensures playback at the correct speed
            sleep_time = chunk_samples / sample_rate
            time.sleep(sleep_time)

            # Move to next chunk
            current_chunk += 1

        # If playback completed normally, send final progress update
        if current_chunk >= num_chunks and not self.playback_stop.is_set():
            for callback in self.playback_callbacks:
                try:
                    callback(1.0)  # 100% complete
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

        logger.info("Playback thread completed")

    def pause_playback(self):
        """
        Pause audio playback.

        Returns:
            True if successful, False otherwise
        """
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_paused.set()
            logger.info("Playback paused")
            return True
        return False

    def resume_playback(self):
        """
        Resume paused audio playback.

        Returns:
            True if successful, False otherwise
        """
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_paused.clear()
            logger.info("Playback resumed")
            return True
        return False

    def stop_playback(self):
        """
        Stop audio playback completely.

        Returns:
            True if successful, False otherwise
        """
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_stop.set()
            self.playback_paused.clear()

            # Wait for the thread to finish (with timeout)
            self.playback_thread.join(timeout=1.0)

            # Close output stream
            if self.output_stream:
                try:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                    self.output_stream = None
                except Exception as e:
                    logger.error(f"Error closing output stream: {e}")

            # Reset playback state
            self.playback_thread = None
            self.playback_position = 0

            logger.info("Playback stopped")
            return True
        return False

    def is_playing(self):
        """
        Check if audio is currently playing.

        Returns:
            True if playing, False otherwise
        """
        return (self.playback_thread and
                self.playback_thread.is_alive() and
                not self.playback_paused.is_set())

    def is_playback_paused(self):
        """
        Check if playback is paused.

        Returns:
            True if paused, False otherwise
        """
        return (self.playback_thread and
                self.playback_thread.is_alive() and
                self.playback_paused.is_set())

    def seek_playback(self, position):
        """
        Seek to a position in the current audio playback.

        Args:
            position: Position as a value between 0.0 and 1.0

        Returns:
            True if successful, False otherwise
        """
        if (self.playback_data is None or
            self.playback_thread is None or
            not self.playback_thread.is_alive()):
            return False

        # Calculate new position in samples
        position = max(0.0, min(1.0, position))  # Clamp to [0.0, 1.0]
        new_position = int(position * len(self.playback_data))

        # Update position
        self.playback_position = new_position

        logger.info(f"Playback position set to {position:.2f}")
        return True

    def get_playback_progress(self):
        """
        Get the current playback progress.

        Returns:
            Progress as a value between 0.0 and 1.0, or None if not playing
        """
        if self.playback_data is None:
            return None

        return self.playback_position / len(self.playback_data)

    def get_playback_duration(self):
        """
        Get the duration of the current audio.

        Returns:
            Duration in seconds, or None if no audio is loaded
        """
        if self.playback_data is None or self.playback_sample_rate <= 0:
            return None

        return len(self.playback_data) / self.playback_sample_rate

    def register_progress_callback(self, callback):
        """
        Register a callback function to receive playback progress updates.

        Args:
            callback: Function that takes a progress value (0.0 to 1.0)
        """
        if callback not in self.playback_callbacks:
            self.playback_callbacks.append(callback)

    def unregister_progress_callback(self, callback):
        """
        Unregister a progress callback function.

        Args:
            callback: The callback function to remove
        """
        if callback in self.playback_callbacks:
            self.playback_callbacks.remove(callback)

    def cleanup(self):
        """
        Clean up resources.
        """
        self.stop_recording()
        self.stop_playback()

        # Close any open streams
        if self.input_stream:
            self.input_stream.close()
            self.input_stream = None

        if self.output_stream:
            self.output_stream.close()
            self.output_stream = None

        # Terminate PyAudio instance
        self.audio.terminate()

        logger.info("Audio capture cleaned up")
