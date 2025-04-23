"""
BirdNET plugin for Sound Analyzer with working automatic analysis.
Identifies bird species in audio using the BirdNET library.
"""
import numpy as np
import logging
import time
import threading
import traceback
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import os
import tempfile
import wave
import uuid
import sys
import sqlite3

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QSlider, QCheckBox,
    QComboBox, QHeaderView, QGroupBox, QFormLayout, QDoubleSpinBox,
    QProgressBar, QTextEdit, QSplitter, QDialog, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QTextCursor
from utils.detection_db import get_db_instance

try:
    # Try to import birdnet package
    from birdnet import SpeciesPredictions, SpeciesPrediction, predict_species_within_audio_file
    from birdnet.models import ModelV2M4
    BIRDNET_AVAILABLE = True
    BIRDNET_TYPE = "birdnet"
except ImportError:
    BIRDNET_AVAILABLE = False
    BIRDNET_TYPE = None

    # Check if birdnetlib is available as an alternative
    import importlib.util
    if importlib.util.find_spec("birdnetlib") is not None:
        try:
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer
            BIRDNET_AVAILABLE = True
            BIRDNET_TYPE = "birdnetlib"
        except ImportError:
            pass

from plugins.base_plugin import BasePlugin

# Set up a separate logger for this plugin
logger = logging.getLogger(__name__)

# Make sure the logger is set to capture DEBUG level messages
logger.setLevel(logging.DEBUG)

class PluginLogHandler(logging.Handler):
    """Custom log handler to capture logs for display in plugin UI."""

    def __init__(self):
        super().__init__()
        self.logs = []
        self.log_observers = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

        # Only keep the last 100 logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

        # Notify observers
        for callback in self.log_observers:
            callback(log_entry)

    def get_logs(self):
        return self.logs

    def add_observer(self, callback):
        self.log_observers.append(callback)

    def remove_observer(self, callback):
        if callback in self.log_observers:
            self.log_observers.remove(callback)

# Create a custom log handler for the plugin
plugin_log_handler = PluginLogHandler()
plugin_log_handler.setLevel(logging.DEBUG)
plugin_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the custom handler to the logger
logger.addHandler(plugin_log_handler)


class BirdNETAnalysisThread(QThread):
    """
    Thread for running BirdNET analysis in the background.
    """
    analysis_complete = pyqtSignal(list)
    analysis_progress = pyqtSignal(int)  # Progress in percentage (0-100)

    def __init__(self, model, audio_data, sample_rate, detection_window,
                 confidence_threshold, lat=None, lon=None, week=None):
        """
        Initialize the analysis thread.

        Args:
            model: The BirdNET model to use
            audio_data: Audio data to analyze
            sample_rate: Sample rate of the audio data
            detection_window: Size of detection window in seconds
            confidence_threshold: Minimum confidence threshold
            lat: Optional latitude for species filtering
            lon: Optional longitude for species filtering
            week: Optional week number for species filtering
        """
        super().__init__()

        # Generate a unique ID for this analysis
        self.analysis_id = str(uuid.uuid4())[:8]

        self.model = model
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.detection_window = detection_window
        self.confidence_threshold = confidence_threshold
        self.lat = lat
        self.lon = lon
        self.week = week

        # Add debug logging for audio data
        logger.info(f"[Analysis {self.analysis_id}] Created with {len(audio_data)} samples, sample_rate={sample_rate}, confidence_threshold={confidence_threshold}")

        # Log audio stats (min, max, mean, non-zero percentage)
        if len(audio_data) > 0:
            try:
                non_zero_percentage = np.count_nonzero(audio_data) / len(audio_data) * 100
                rms = np.sqrt(np.mean(np.square(audio_data)))
                logger.info(f"[Analysis {self.analysis_id}] Audio stats: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean={np.mean(audio_data):.4f}, rms={rms:.4f}, non-zero={non_zero_percentage:.2f}%")

                # Check if audio is too quiet
                if rms < 0.01:
                    logger.warning(f"[Analysis {self.analysis_id}] Audio may be too quiet for good detection (RMS={rms:.4f})")
            except Exception as e:
                logger.error(f"[Analysis {self.analysis_id}] Error calculating audio stats: {e}")

    def run(self):
        """
        Run the analysis.
        """
        try:
            logger.info(f"[Analysis {self.analysis_id}] Starting BirdNET analysis using {BIRDNET_TYPE}")
            self.analysis_progress.emit(10)  # Starting

            if BIRDNET_TYPE == "birdnet":
                results = self._analyze_with_birdnet()
            elif BIRDNET_TYPE == "birdnetlib":
                results = self._analyze_with_birdnetlib()
            else:
                results = []

            logger.info(f"[Analysis {self.analysis_id}] Analysis complete, found {len(results)} potential bird species")
            for idx, result in enumerate(results):
                logger.info(f"[Analysis {self.analysis_id}] Detection {idx+1}: {result[1]} ({result[0]}) with confidence {result[2]:.4f}")

            self.analysis_progress.emit(100)  # Complete
            self.analysis_complete.emit(results)

        except Exception as e:
            logger.error(f"[Analysis {self.analysis_id}] Error in BirdNET analysis: {e}")
            logger.error(f"[Analysis {self.analysis_id}] Traceback: {traceback.format_exc()}")
            self.analysis_progress.emit(100)  # Complete
            self.analysis_complete.emit([])

    def _analyze_with_birdnet(self):
        """
        Analyze audio using birdnet package.

        Returns:
            List of (species, confidence) tuples
        """
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write audio data to the temporary file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                # Convert float32 to int16
                audio_int16 = (self.audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            logger.debug(f"[Analysis {self.analysis_id}] Created temporary WAV file at {temp_path}")
            self.analysis_progress.emit(30)  # File created

            # Get species in the area if location is specified
            filter_species = None
            if self.lat is not None and self.lon is not None and self.week is not None:
                logger.info(f"[Analysis {self.analysis_id}] Getting species for location: lat={self.lat}, lon={self.lon}, week={self.week}")
                species_in_area = self.model.predict_species_at_location_and_time(
                    self.lat, self.lon, week=self.week
                )
                filter_species = set(species_in_area.keys())
                logger.info(f"[Analysis {self.analysis_id}] Found {len(filter_species)} potential species for this location and time")

            # Analyze the audio file
            logger.info(f"[Analysis {self.analysis_id}] Running BirdNET analysis on audio file")
            self.analysis_progress.emit(50)  # Analysis started

            predictions = self.model.predict_species_within_audio_file(
                Path(temp_path),
                filter_species=filter_species
            )

            self.analysis_progress.emit(80)  # Processing results

            # Process the results
            results = []
            for time_idx, prediction in predictions.items():
                logger.debug(f"[Analysis {self.analysis_id}] Processing predictions at time {time_idx}")
                for species, conf in prediction.items():
                    logger.debug(f"[Analysis {self.analysis_id}] Raw prediction: {species} with confidence {conf:.4f}")
                    if conf >= self.confidence_threshold:
                        # Extract common name from species string (format: "Scientific_Common")
                        if '_' in species:
                            scientific, common = species.split('_', 1)
                        else:
                            scientific, common = species, ""

                        results.append((scientific, common, conf, time_idx[0]))
                        logger.info(f"[Analysis {self.analysis_id}] Accepted prediction: {scientific} ({common}) with confidence {conf:.4f}")
                    else:
                        logger.debug(f"[Analysis {self.analysis_id}] Rejected prediction (below threshold): {species} with confidence {conf:.4f}")

            return results

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"[Analysis {self.analysis_id}] Removed temporary WAV file: {temp_path}")

    def _analyze_with_birdnetlib(self):
        """
        Analyze audio using birdnetlib package with maximum API compatibility.

        Returns:
            List of (species, confidence) tuples
        """
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write audio data to the temporary file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                # Convert float32 to int16
                audio_int16 = (self.audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            logger.debug(f"[Analysis {self.analysis_id}] Created temporary WAV file at {temp_path}, size: {os.path.getsize(temp_path)} bytes")
            self.analysis_progress.emit(30)  # File created

            # Import required modules
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer

            # Create analyzer first
            analyzer = Analyzer()
            logger.info(f"[Analysis {self.analysis_id}] Created analyzer")

            # Create a Recording object with analyzer
            try:
                recording = Recording(analyzer, temp_path)
                logger.info(f"[Analysis {self.analysis_id}] Created recording with analyzer parameter")
            except TypeError as e:
                # Try alternative API where analyzer is passed to analyze_recording
                logger.warning(f"[Analysis {self.analysis_id}] Error with analyzer parameter, trying alternative API: {e}")
                recording = Recording(temp_path)
                logger.info(f"[Analysis {self.analysis_id}] Created recording without analyzer parameter")

            # Try to set location data if supported
            try:
                if self.lat is not None and self.lon is not None:
                    if hasattr(recording, 'lat') and hasattr(recording, 'lon'):
                        recording.lat = self.lat
                        recording.lon = self.lon
                        logger.info(f"[Analysis {self.analysis_id}] Set location: lat={self.lat}, lon={self.lon}")

                    if hasattr(recording, 'week') and self.week is not None:
                        recording.week = self.week
                        logger.info(f"[Analysis {self.analysis_id}] Set week: {self.week}")
            except Exception as e:
                logger.warning(f"[Analysis {self.analysis_id}] Could not set location data: {e}")

            # Run analysis
            logger.info(f"[Analysis {self.analysis_id}] Running BirdNETLib analysis on audio file")
            self.analysis_progress.emit(50)  # Analysis started

            try:
                # Try with analyzer already in recording
                recording.analyze()
                logger.info(f"[Analysis {self.analysis_id}] Analysis complete using recording.analyze()")
            except (AttributeError, TypeError) as e:
                # Try with analyzer as parameter
                logger.warning(f"[Analysis {self.analysis_id}] Error with recording.analyze(), trying analyzer.analyze_recording: {e}")
                analyzer.analyze_recording(recording)
                logger.info(f"[Analysis {self.analysis_id}] Analysis complete using analyzer.analyze_recording()")

            # Get the detections
            if hasattr(recording, 'detections'):
                detections = recording.detections
            elif hasattr(analyzer, 'get_detections'):
                detections = analyzer.get_detections(recording)
            else:
                raise AttributeError("Cannot find detections in either recording or analyzer")

            logger.info(f"[Analysis {self.analysis_id}] Found {len(detections)} detections")
            self.analysis_progress.emit(80)  # Processing results

            # Process the results
            results = []
            for detection in detections:
                # Handle different detection formats
                if isinstance(detection, dict):
                    # Dictionary format
                    confidence = detection.get('confidence', 0)
                    common_name = detection.get('common_name', 'Unknown')
                    scientific_name = detection.get('scientific_name', 'Unknown')
                    time_start = detection.get('start_time', 0)
                elif hasattr(detection, 'confidence') and hasattr(detection, 'common_name'):
                    # Object format
                    confidence = detection.confidence
                    common_name = detection.common_name
                    scientific_name = detection.scientific_name if hasattr(detection, 'scientific_name') else 'Unknown'
                    time_start = detection.start_time if hasattr(detection, 'start_time') else 0
                else:
                    logger.warning(f"[Analysis {self.analysis_id}] Unexpected detection format: {detection}")
                    continue

                logger.debug(f"[Analysis {self.analysis_id}] Raw detection: {common_name} with confidence {confidence:.4f}")

                if confidence >= self.confidence_threshold:
                    results.append((scientific_name, common_name, confidence, time_start))
                    logger.info(f"[Analysis {self.analysis_id}] Accepted detection: {scientific_name} ({common_name}) with confidence {confidence:.4f}")
                else:
                    logger.debug(f"[Analysis {self.analysis_id}] Rejected detection (below threshold): {common_name} with confidence {confidence:.4f}")

            return results

        except Exception as e:
            logger.error(f"[Analysis {self.analysis_id}] Error in birdnetlib analysis: {traceback.format_exc()}")

            # Try to fall back to local file analysis without the library
            logger.warning(f"[Analysis {self.analysis_id}] Attempting fallback to manual analysis")

            try:
                # Create a very basic bird sound detector using frequency analysis
                # This is just for demonstration - it's not a real bird detector

                # Convert audio to spectrogram
                from scipy import signal

                # Create the window
                window = signal.windows.hann(1024)

                # Calculate the Short-Time Fourier Transform (STFT)
                frequencies, times, stft = signal.stft(
                    self.audio_data,
                    fs=self.sample_rate,
                    window=window,
                    nperseg=1024,
                    noverlap=512,
                    nfft=1024,
                    return_onesided=True
                )

                # Convert to magnitude spectrogram
                spectrogram = np.abs(stft)

                # Look for energy in bird frequency range (typically 1000-8000 Hz)
                bird_freq_mask = (frequencies >= 1000) & (frequencies <= 8000)
                bird_spectrogram = spectrogram[bird_freq_mask, :]

                # Check if there's significant energy in the bird range
                bird_energy = np.mean(bird_spectrogram)
                background_energy = np.mean(spectrogram)

                logger.info(f"[Analysis {self.analysis_id}] Manual analysis: bird_energy={bird_energy:.5f}, background_energy={background_energy:.5f}")

                if bird_energy > 0.01 and bird_energy > background_energy * 1.5:
                    # Simulated bird detection
                    logger.info(f"[Analysis {self.analysis_id}] Manual analysis detected potential bird sounds")
                    return [("Unknown", "Unknown Bird", 0.6, 0)]
                else:
                    logger.info(f"[Analysis {self.analysis_id}] Manual analysis did not detect bird sounds")
                    return []

            except Exception as fallback_error:
                logger.error(f"[Analysis {self.analysis_id}] Fallback analysis also failed: {fallback_error}")
                return []

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"[Analysis {self.analysis_id}] Removed temporary WAV file: {temp_path}")


class BirdNETPlugin(BasePlugin):
    """
    BirdNET plugin for bird species identification.
    """
    # Declare class attributes
    auto_analysis_timer = None
    buffer_update_timer = None
    debug_timer = None

    def __init__(self, audio_processor, config):
        """
        Initialize the BirdNET plugin.

        Args:
            audio_processor: The audio processor instance
            config: Plugin configuration
        """
        super().__init__(audio_processor, config)

        # BirdNET settings
        self.confidence_threshold = config.get('confidence_threshold', 0.3)  # Lower default
        self.analysis_interval = config.get('analysis_interval', 3.0)  # Analyze every 3 seconds
        self.detection_window = config.get('detection_window', 3.0)   # Window size in seconds
        self.overlap = config.get('overlap', 0.0)  # Overlap between windows in seconds
        self.last_analysis_time = 0

        # Location settings - Default to a popular birding location (Central Park, NYC)
        self.use_location = config.get('use_location', True)  # Default to true
        self.latitude = config.get('latitude', 40.7829)
        self.longitude = config.get('longitude', -73.9654)

        # Get current week (1-52)
        self.week = datetime.now().isocalendar()[1]

        # Buffer for analysis
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = audio_processor.sample_rate

        # Results storage
        self.results = []
        self.max_results = config.get('max_results', 10)

        # Background analysis thread
        self.analysis_thread = None
        self.analysis_thread_running = False
        self.analysis_progress = 0

        # Recording status
        self.is_recording = False

        # UI components
        self.ui_widget = None
        self.results_table = None
        self.status_label = None
        self.log_display = None
        self.progress_bar = None
        self.buffer_label = None

        # Initialize auto-analysis timer
        self.auto_analysis_timer = QTimer()
        self.auto_analysis_timer.setObjectName("auto_analysis_timer")
        self.auto_analysis_timer.timeout.connect(self.check_auto_analysis)
        self.auto_analysis_timer.start(500)  # Check every 500ms
        logger.info(f"Auto-analysis timer started with {self.analysis_interval}s interval")

        # Debug timer to log buffer status periodically
        self.debug_timer = QTimer()
        self.debug_timer.timeout.connect(self._log_buffer_status)
        self.debug_timer.start(50000)  # Log every 5 seconds

        # Initialize BirdNET model if available
        self.model = None
        self.init_birdnet_model()

        # Initialize detection database
        self.db = None
        self.session_id = None

        # Log initialization details
        logger.info(f"BirdNET Plugin initialized with: confidence_threshold={self.confidence_threshold}, "
                    f"analysis_interval={self.analysis_interval}s, detection_window={self.detection_window}s, "
                    f"use_location={self.use_location}, latitude={self.latitude}, longitude={self.longitude}")

        # Force check auto-analysis soon after initialization
        QTimer.singleShot(5000, self.check_auto_analysis)

    def _log_buffer_status(self):
        """
        Log buffer status periodically for debugging.
        """
        buffer_seconds = len(self.buffer) / self.sample_rate if self.sample_rate > 0 else 0
        logger.debug(f"Buffer status: {len(self.buffer)} samples ({buffer_seconds:.2f} seconds), is_recording={self.is_recording}")

        # Calculate RMS of buffer (if not empty)
        if len(self.buffer) > 0:
            rms = np.sqrt(np.mean(np.square(self.buffer)))
            logger.debug(f"Buffer audio stats: rms={rms:.5f}")

    def init_birdnet_model(self):
        """
        Initialize the BirdNET model.
        """
        if not BIRDNET_AVAILABLE:
            logger.warning("BirdNET package not found. Install with: pip install birdnet or pip install birdnetlib")
            return

        try:
            if BIRDNET_TYPE == "birdnet":
                # Initialize the birdnet model
                self.model = ModelV2M4()
                logger.info("Initialized BirdNET model using birdnet package")
            elif BIRDNET_TYPE == "birdnetlib":
                # For birdnetlib, we don't need to initialize a model here
                # It will be created when needed in the analysis thread
                self.model = True  # Just a placeholder to indicate model is available
                logger.info("Using birdnetlib for bird species identification")
        except Exception as e:
            logger.error(f"Error initializing BirdNET model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.model = None

    def process_audio(self, audio_chunk, sample_rate):
        """
        Process an audio chunk for bird species identification.

        Args:
            audio_chunk: The audio data
            sample_rate: The audio sample rate

        Returns:
            Dict with processing results
        """
        # Debug log the audio chunk
        if audio_chunk is not None:
            logger.debug(f"Received audio chunk: {len(audio_chunk)} samples, sample_rate={sample_rate}")

        # Update recording status
        if audio_chunk is not None and len(audio_chunk) > 0:
            self.is_recording = True
            if self.status_label:
                self.status_label.setText("Status: Recording/Processing")
        else:
            self.is_recording = False
            if self.status_label:
                self.status_label.setText("Status: Idle")

        # If BirdNET is not available, return empty results
        if not BIRDNET_AVAILABLE or self.model is None:
            return {'species': []}

        # Add the new chunk to the buffer
        if audio_chunk is not None and len(audio_chunk) > 0:
            try:
                if sample_rate != self.sample_rate:
                    # Resample if needed (simple method - not ideal for production)
                    ratio = self.sample_rate / sample_rate
                    resampled_chunk = np.interp(
                        np.arange(0, len(audio_chunk) * ratio),
                        np.arange(0, len(audio_chunk) * ratio, ratio),
                        audio_chunk
                    )
                    self.buffer = np.append(self.buffer, resampled_chunk)
                    logger.debug(f"Resampled audio chunk from {sample_rate}Hz to {self.sample_rate}Hz")
                else:
                    self.buffer = np.append(self.buffer, audio_chunk)

                # Log some stats about the incoming audio
                if len(audio_chunk) > 0:
                    rms = np.sqrt(np.mean(np.square(audio_chunk)))
                    logger.debug(f"Audio chunk stats: rms={rms:.5f}, min={np.min(audio_chunk):.5f}, max={np.max(audio_chunk):.5f}")
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

        # Limit buffer size to detection window
        max_samples = int(self.detection_window * self.sample_rate)
        if len(self.buffer) > max_samples:
            self.buffer = self.buffer[-max_samples:]

        # Return current species list
        return {'species': self.results}

    def check_auto_analysis(self):
        """
        Check if it's time for automatic analysis.
        This is called by the timer at regular intervals.
        """
        # Always log current status for debugging
        current_time = time.time()
        time_since_last = current_time - self.last_analysis_time
        max_samples = int(self.detection_window * self.sample_rate)
        buffer_seconds = len(self.buffer) / self.sample_rate if self.sample_rate > 0 else 0
        buffer_percentage = len(self.buffer) / max_samples * 100 if max_samples > 0 else 0

        # Log every check to see what's happening
        logger.debug(
            f"Auto-analysis check: time_since={time_since_last:.1f}s (interval={self.analysis_interval:.1f}s), "
            f"buffer={buffer_seconds:.2f}s/{self.detection_window:.1f}s ({buffer_percentage:.0f}%), "
            f"recording={self.is_recording}, running={self.analysis_thread_running}"
        )

        # Skip if analysis is already running
        if self.analysis_thread_running:
            logger.debug("Auto-analysis skipped: Analysis already running")
            return

        # Detailed conditions check with logging
        if not self.is_recording:
            logger.debug("Auto-analysis skipped: Not recording")
            return

        if time_since_last < self.analysis_interval:
            logger.debug(f"Auto-analysis skipped: Last analysis too recent ({time_since_last:.1f}s < {self.analysis_interval:.1f}s)")
            return

        if len(self.buffer) < max_samples * 0.4:
            logger.debug(f"Auto-analysis skipped: Buffer too small ({buffer_percentage:.0f}% < 40%)")
            return

        # If we got here, all conditions are met - trigger analysis
        logger.info(f"Auto-analysis triggered after {time_since_last:.2f}s with {buffer_seconds:.2f}s of audio ({buffer_percentage:.0f}%)")
        self.start_analysis()
        self.last_analysis_time = current_time

    def start_analysis(self):
        """
        Start bird species analysis in a background thread.
        """
        if self.analysis_thread_running:
            logger.debug("Analysis already in progress, skipping")
            return

        if len(self.buffer) == 0:
            logger.warning("Cannot start analysis: Buffer is empty")
            if self.status_label:
                self.status_label.setText("Status: Cannot analyze - No audio data")
            return

        logger.info("Starting BirdNET analysis thread")
        self.analysis_thread_running = True
        self.analysis_progress = 0

        if self.progress_bar:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

        # Get location parameters if enabled
        lat, lon, week = None, None, None
        if self.use_location and self.latitude is not None and self.longitude is not None:
            lat = self.latitude
            lon = self.longitude
            week = self.week
            logger.info(f"Using location for filtering: lat={lat}, lon={lon}, week={week}")

        # Log buffer stats before analysis
        buffer_seconds = len(self.buffer) / self.sample_rate if self.sample_rate > 0 else 0
        if len(self.buffer) > 0:
            rms = np.sqrt(np.mean(np.square(self.buffer)))
            logger.info(f"Starting analysis with buffer: {len(self.buffer)} samples ({buffer_seconds:.2f} seconds), RMS={rms:.5f}")

        # Create and start the analysis thread
        try:
            self.analysis_thread = BirdNETAnalysisThread(
                self.model,
                self.buffer.copy(),  # Copy buffer to avoid modification during analysis
                self.sample_rate,
                self.detection_window,
                self.confidence_threshold,
                lat, lon, week
            )

            # Connect signals
            self.analysis_thread.analysis_complete.connect(self.handle_analysis_results)
            self.analysis_thread.analysis_progress.connect(self.update_progress)

            # Start the thread
            self.analysis_thread.start()

        except Exception as e:
            logger.error(f"Error starting analysis thread: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.analysis_thread_running = False
            if self.status_label:
                self.status_label.setText(f"Status: Analysis error: {str(e)[:30]}...")

    def update_progress(self, progress):
        """
        Update the analysis progress.

        Args:
            progress: Progress percentage (0-100)
        """
        self.analysis_progress = progress
        if self.progress_bar:
            self.progress_bar.setValue(progress)

    def handle_analysis_results(self, results):
        """
        Handle the results from BirdNET analysis.

        Note: This is a replacement for the original handle_analysis_results
        method to add database storage.

        Args:
            results: List of (scientific_name, common_name, confidence, time) tuples
        """
        logger.info(f"Received analysis results: {len(results)} detections")

        # Make sure database is initialized
        if not hasattr(self, 'db') or self.db is None:
            self._init_database()

        # Update results list with new species
        for scientific, common, confidence, time_start in results:
            # Save to database first - use a copy of the buffer for the specific detection
            # This gives us the audio segment that triggered this detection
            buffer_copy = self.buffer.copy()
            self._save_detection_to_db(scientific, common, confidence, buffer_copy)

            # Check if species is already in the list
            existing = next((r for r in self.results if r[0] == scientific), None)

            if existing:
                # Update existing entry if new confidence is higher
                if confidence > existing[2]:
                    logger.info(f"Updating existing species: {scientific} ({common}) with higher confidence: {confidence:.4f} > {existing[2]:.4f}")
                    existing[2] = confidence  # Update confidence
                    existing[3] = time.time()  # Update timestamp
            else:
                # Add new species to the list
                logger.info(f"Adding new species: {scientific} ({common}) with confidence: {confidence:.4f}")
                self.results.append([scientific, common, confidence, time.time()])

                # Sort by confidence (highest first) and keep only max_results
                self.results.sort(key=lambda x: x[2], reverse=True)
                if len(self.results) > self.max_results:
                    removed = self.results[self.max_results:]
                    self.results = self.results[:self.max_results]
                    logger.info(f"Removed {len(removed)} results to keep only the top {self.max_results}")

        # Update the UI
        self.update_results_table()

        # Clear the analysis thread
        self.analysis_thread_running = False
        self.analysis_thread = None
        logger.info("Analysis thread completed")

        # Hide progress bar
        if self.progress_bar:
            self.progress_bar.setVisible(False)

        # Update status label
        if self.status_label:
            if len(results) > 0:
                self.status_label.setText(f"Status: Detected and saved {len(results)} species")
            else:
                self.status_label.setText("Status: No birds detected")

    def update_results_table(self):
        """
        Update the results table in the UI.
        """
        if self.results_table is None:
            logger.warning("Results table is None, cannot update UI")
            return

        logger.info(f"Updating results table with {len(self.results)} species")
        self.results_table.setRowCount(len(self.results))

        for row, (scientific, common, confidence, timestamp) in enumerate(self.results):
            # Scientific name
            self.results_table.setItem(row, 0, QTableWidgetItem(scientific))

            # Common name
            self.results_table.setItem(row, 1, QTableWidgetItem(common))

            # Confidence
            conf_item = QTableWidgetItem(f"{confidence:.2f}")
            self.results_table.setItem(row, 2, conf_item)

            # Detected time
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            self.results_table.setItem(row, 3, QTableWidgetItem(time_str))

            # Color code by confidence
            color = self.get_confidence_color(confidence)
            for col in range(4):
                self.results_table.item(row, col).setBackground(color)

    def get_confidence_color(self, confidence):
        """
        Get a color based on confidence level.

        Args:
            confidence: Confidence value (0.0 to 1.0)

        Returns:
            QColor for the confidence level
        """
        if confidence >= 0.8:
            return QColor(200, 255, 200)  # Light green for high confidence
        elif confidence >= 0.6:
            return QColor(255, 255, 200)  # Light yellow for medium confidence
        else:
            return QColor(255, 200, 200)  # Light red for low confidence

    def initialize_ui(self, main_window):
        """
        Initialize the plugin's UI components.
        """
        logger.info("Initializing BirdNET plugin UI")

        # Check audio connection
        if self.audio_processor:
            logger.info(f"Audio processor exists, sample_rate={self.audio_processor.sample_rate}")
            # Register directly for audio updates
            self.audio_processor.register_callback(self._on_audio_processor_update)
        else:
            logger.error("No audio processor connected to plugin!")

        # Create the main widget
        self.ui_widget = QWidget()
        layout = QVBoxLayout(self.ui_widget)

        # Create a splitter for the main content and debug area
        main_splitter = QSplitter(Qt.Vertical)

        # Main content widget
        main_content = QWidget()
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Status and settings section
        status_group = QGroupBox("BirdNET Status")
        status_layout = QVBoxLayout(status_group)

        # Status label
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Status: Initializing")
        status_layout.addWidget(self.status_label)

        # Add buffer status display
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Buffer:"))
        self.buffer_label = QLabel("Empty")
        buffer_layout.addWidget(self.buffer_label)
        status_layout.addLayout(buffer_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        if not BIRDNET_AVAILABLE or self.model is None:
            status_label = QLabel("BirdNET is not available. Install with:\npip install birdnet\nor\npip install birdnetlib")
            status_label.setStyleSheet("color: red;")
            status_layout.addWidget(status_label)
            logger.warning("BirdNET is not available, showing installation instructions in UI")
            self.status_label.setText("Status: Not Available")
        else:
            info_label = QLabel(f"BirdNET is active using {BIRDNET_TYPE}")
            info_label.setStyleSheet("color: green;")
            status_layout.addWidget(info_label)
            logger.info(f"BirdNET is active using {BIRDNET_TYPE}")
            self.status_label.setText("Status: Ready")

            # Add settings section
            settings_layout = QFormLayout()

            # Confidence threshold slider
            threshold_layout = QHBoxLayout()
            threshold_label = QLabel(f"Confidence Threshold: {self.confidence_threshold:.2f}")
            threshold_slider = QSlider(Qt.Horizontal)
            threshold_slider.setMinimum(0)
            threshold_slider.setMaximum(100)
            threshold_slider.setValue(int(self.confidence_threshold * 100))
            threshold_slider.setTickPosition(QSlider.TicksBelow)
            threshold_slider.setTickInterval(10)

            def update_threshold(value):
                self.confidence_threshold = value / 100.0
                threshold_label.setText(f"Confidence Threshold: {self.confidence_threshold:.2f}")
                logger.info(f"Confidence threshold changed to {self.confidence_threshold:.2f}")

            threshold_slider.valueChanged.connect(update_threshold)
            threshold_layout.addWidget(threshold_label)
            threshold_layout.addWidget(threshold_slider)
            settings_layout.addRow("", threshold_layout)

            # Analysis interval
            interval_spin = QDoubleSpinBox()
            interval_spin.setMinimum(1.0)
            interval_spin.setMaximum(10.0)
            interval_spin.setSingleStep(0.5)
            interval_spin.setValue(self.analysis_interval)
            interval_spin.valueChanged.connect(lambda value: self._update_analysis_interval(value))
            settings_layout.addRow("Analysis Interval (s):", interval_spin)

            # Detection window
            window_spin = QDoubleSpinBox()
            window_spin.setMinimum(1.0)
            window_spin.setMaximum(10.0)
            window_spin.setSingleStep(0.5)
            window_spin.setValue(self.detection_window)
            window_spin.valueChanged.connect(lambda value: self._update_detection_window(value))
            settings_layout.addRow("Detection Window (s):", window_spin)

            # Location settings
            location_check = QCheckBox("Use Location for Species Filtering")
            location_check.setChecked(self.use_location)
            location_check.stateChanged.connect(lambda state: self._update_use_location(state))
            settings_layout.addRow("", location_check)

            # Latitude and longitude
            lat_layout = QHBoxLayout()
            lat_spin = QDoubleSpinBox()
            lat_spin.setMinimum(-90.0)
            lat_spin.setMaximum(90.0)
            lat_spin.setDecimals(6)
            lat_spin.setSingleStep(0.1)
            lat_spin.setValue(self.latitude if self.latitude is not None else 0.0)
            lat_spin.valueChanged.connect(lambda value: setattr(self, 'latitude', value))
            lat_layout.addWidget(lat_spin)
            lat_layout.addWidget(QLabel("°N"))
            settings_layout.addRow("Latitude:", lat_layout)

            lon_layout = QHBoxLayout()
            lon_spin = QDoubleSpinBox()
            lon_spin.setMinimum(-180.0)
            lon_spin.setMaximum(180.0)
            lon_spin.setDecimals(6)
            lon_spin.setSingleStep(0.1)
            lon_spin.setValue(self.longitude if self.longitude is not None else 0.0)
            lon_spin.valueChanged.connect(lambda value: setattr(self, 'longitude', value))
            lon_layout.addWidget(lon_spin)
            lon_layout.addWidget(QLabel("°E"))
            settings_layout.addRow("Longitude:", lon_layout)

            # Force analysis button
            force_analysis_btn = QPushButton("Force Analysis Now")
            force_analysis_btn.clicked.connect(self._force_analysis)
            settings_layout.addRow("", force_analysis_btn)

            # Test sound button - generate a test sound
            test_sound_btn = QPushButton("Test With Bird Sound")
            test_sound_btn.clicked.connect(self._generate_test_sound)
            settings_layout.addRow("", test_sound_btn)

            # Add auto-analysis toggle
            auto_analysis_check = QCheckBox("Enable Auto Analysis")
            auto_analysis_check.setChecked(True)  # Default to enabled
            auto_analysis_check.stateChanged.connect(lambda state: self._toggle_auto_analysis(state))
            settings_layout.addRow("", auto_analysis_check)

            status_layout.addLayout(settings_layout)

        main_layout.addWidget(status_group)

        # Add results table
        results_group = QGroupBox("Detected Species")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Scientific Name", "Common Name", "Confidence", "Time"])

        # Set column widths
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        results_layout.addWidget(self.results_table)

        # Add control buttons
        buttons_layout = QHBoxLayout()

        # Clear button
        clear_button = QPushButton("Clear Results")
        clear_button.clicked.connect(self.clear_results)
        buttons_layout.addWidget(clear_button)

        # Lower threshold button
        lower_threshold_btn = QPushButton("Lower Threshold (-0.1)")
        lower_threshold_btn.clicked.connect(self._lower_threshold)
        buttons_layout.addWidget(lower_threshold_btn)

        # Clear buffer button
        clear_buffer_btn = QPushButton("Clear Buffer")
        clear_buffer_btn.clicked.connect(self._clear_buffer)
        buttons_layout.addWidget(clear_buffer_btn)

        results_layout.addLayout(buttons_layout)

        main_layout.addWidget(results_group)

        # Add debug log section
        debug_group = QGroupBox("Debug Log")
        debug_layout = QVBoxLayout(debug_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(200)
        self.log_display.setStyleSheet("font-family: monospace; font-size: 10px;")
        debug_layout.addWidget(self.log_display)

        # Register for log updates
        plugin_log_handler.add_observer(self._on_log_update)

        # Initialize log display with existing logs
        for log_entry in plugin_log_handler.get_logs():
            self._on_log_update(log_entry)

        # Add to main splitter
        main_splitter.addWidget(main_content)
        main_splitter.addWidget(debug_group)

        # Set splitter sizes (70% main content, 30% debug)
        main_splitter.setSizes([700, 300])

        layout.addWidget(main_splitter)

        # Add the plugin's widget to the main window
        main_window.add_plugin_tab("birdnet", self.ui_widget, "BirdNET")
        logger.info("BirdNET plugin UI initialized")


        # Add after all other UI initialization, but before starting timers

        # Add database controls section
        db_group = QGroupBox("Detection Database")
        db_layout = QVBoxLayout(db_group)

        # Database status
        db_status_label = QLabel("Database: Not initialized")
        db_layout.addWidget(db_status_label)
        self.db_status_label = db_status_label

        # Add controls
        db_buttons_layout = QHBoxLayout()

        # Start new session button
        new_session_btn = QPushButton("Start New Session")
        new_session_btn.clicked.connect(self._start_new_db_session)
        db_buttons_layout.addWidget(new_session_btn)

        # View detections button
        view_detections_btn = QPushButton("View Detections")
        view_detections_btn.clicked.connect(self._show_detections)
        db_buttons_layout.addWidget(view_detections_btn)

        # Export detections button
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self._export_detections)
        db_buttons_layout.addWidget(export_btn)

        db_layout.addLayout(db_buttons_layout)

        # Add to main layout - find where to add it
        if main_content is not None and isinstance(main_content, QWidget):
            # Try to add it to the main_layout
            for child in main_content.children():
                if isinstance(child, QVBoxLayout):
                    child.addWidget(db_group)
                    break
        else:
            # Fallback to adding it to the main splitter
            for i in range(main_splitter.count()):
                widget = main_splitter.widget(i)
                if isinstance(widget, QWidget):
                    # Find a suitable layout to add to
                    for child in widget.children():
                        if isinstance(child, QVBoxLayout):
                            child.addWidget(db_group)
                            break

        # Initialize database and update status
        try:
            self._init_database()
            if self.db and self.session_id:
                self.db_status_label.setText(f"Database: Active (Session ID: {self.session_id})")
                self.db_status_label.setStyleSheet("color: green;")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.db_status_label.setText(f"Database: Error initializing")
            self.db_status_label.setStyleSheet("color: red;")


        # Start the buffer update timer
        self.buffer_update_timer = QTimer()
        self.buffer_update_timer.timeout.connect(self._update_buffer_display)
        self.buffer_update_timer.start(1000)  # Update every second

        # Force debugging output soon after UI initialization
        QTimer.singleShot(3000, lambda: logger.info("UI initialized and timers running"))

    def _toggle_auto_analysis(self, state):
        """
        Toggle auto-analysis on or off.

        Args:
            state: Qt.Checked or Qt.Unchecked
        """
        if state == Qt.Checked:
            if not self.auto_analysis_timer.isActive():
                self.auto_analysis_timer.start(500)
                logger.info("Auto-analysis enabled")
                if self.status_label:
                    self.status_label.setText("Status: Auto-analysis enabled")
        else:
            if self.auto_analysis_timer.isActive():
                self.auto_analysis_timer.stop()
                logger.info("Auto-analysis disabled")
                if self.status_label:
                    self.status_label.setText("Status: Auto-analysis disabled")

    def _update_buffer_display(self):
        """
        Update the buffer status display.
        """
        if self.buffer_label:
            buffer_seconds = len(self.buffer) / self.sample_rate if self.sample_rate > 0 else 0
            if len(self.buffer) > 0:
                rms = np.sqrt(np.mean(np.square(self.buffer)))
                self.buffer_label.setText(f"{len(self.buffer)} samples ({buffer_seconds:.2f} s), RMS: {rms:.5f}")
            else:
                self.buffer_label.setText("Empty")

    def _generate_test_sound(self):
        """
        Generate a test bird sound and add it to the buffer.
        """
        try:
            logger.info("Generating test bird sound")

            # Generate a simple chirp sound (sine wave with frequency modulation)
            duration = 2.0  # seconds
            t = np.linspace(0, duration, int(duration * self.sample_rate), endpoint=False)

            # Create a chirp (frequency sweep)
            f0 = 2000  # Start frequency in Hz
            f1 = 4000  # End frequency in Hz

            # Use multiple chirps for a more bird-like sound
            chirps = []

            # First chirp
            phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
            chirp1 = np.sin(phase) * 0.5  # Amplitude 0.5

            # Apply envelope
            envelope = np.exp(-(t - duration/2)**2 / (2 * (duration/10)**2))
            chirp1 = chirp1 * envelope

            # Add a second chirp after a short delay
            delay_samples = int(0.3 * self.sample_rate)
            chirp2 = np.zeros_like(chirp1)
            chirp2[delay_samples:] = chirp1[:-delay_samples] * 0.8

            # Final sound
            test_sound = chirp1 + chirp2

            # Normalize to prevent clipping
            test_sound = test_sound / np.max(np.abs(test_sound)) * 0.9

            # Add sound to buffer for analysis
            self.buffer = test_sound

            logger.info(f"Added test sound to buffer: {len(self.buffer)} samples")
            self.status_label.setText("Status: Test sound added to buffer")

            # Force analysis
            self._force_analysis()

        except Exception as e:
            logger.error(f"Error generating test sound: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _update_analysis_interval(self, value):
        """
        Update the analysis interval.

        Args:
            value: New interval value in seconds
        """
        self.analysis_interval = value
        logger.info(f"Analysis interval changed to {self.analysis_interval:.1f}s")

    def _update_detection_window(self, value):
        """
        Update the detection window size.

        Args:
            value: New window size in seconds
        """
        self.detection_window = value
        logger.info(f"Detection window changed to {self.detection_window:.1f}s")

    def _update_use_location(self, state):
        """
        Update the use location flag.

        Args:
            state: Qt.Checked or Qt.Unchecked
        """
        self.use_location = (state == Qt.Checked)
        logger.info(f"Use location changed to {self.use_location}")

    def _force_analysis(self):
        """
        Force an analysis regardless of time interval.
        """
        if len(self.buffer) == 0:
            logger.warning("Cannot force analysis: buffer is empty")
            self.status_label.setText("Status: Cannot analyze - Buffer empty")
            return

        if self.analysis_thread_running:
            logger.warning("Analysis already in progress")
            return

        logger.info("Forcing immediate analysis")
        self.start_analysis()
        self.last_analysis_time = time.time()

    def _lower_threshold(self):
        """
        Lower the confidence threshold by 0.1.
        """
        if self.confidence_threshold > 0.1:
            self.confidence_threshold -= 0.1
            logger.info(f"Lowered confidence threshold to {self.confidence_threshold:.2f}")

            # Update the status label
            if self.status_label:
                self.status_label.setText(f"Status: Threshold lowered to {self.confidence_threshold:.2f}")

    def _clear_buffer(self):
        """
        Clear the audio buffer.
        """
        self.buffer = np.array([], dtype=np.float32)
        logger.info("Audio buffer cleared")
        if self.status_label:
            self.status_label.setText("Status: Buffer cleared")

    def _on_log_update(self, log_entry):
        """
        Handle a new log entry.

        Args:
            log_entry: The log entry string
        """
        if self.log_display:
            self.log_display.append(log_entry)
            # Scroll to bottom
            cursor = self.log_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.log_display.setTextCursor(cursor)

    def clear_results(self):
        """
        Clear all detection results.
        """
        logger.info("Clearing all detection results")
        self.results = []
        if self.results_table:
            self.results_table.setRowCount(0)

    def get_settings_widget(self):
        """
        Get the plugin settings widget.

        Returns:
            The settings widget
        """
        return None  # Settings are included in the main plugin widget

    def get_visualization_widget(self):
        """
        Get the plugin visualization widget.

        Returns:
            The visualization widget
        """
        return None  # No separate visualization widget

    def shutdown(self):
        """
        Perform cleanup when shutting down the plugin.

        Note: This replaces the original shutdown method to add database session closing.
        """
        # End the database session if we have one
        if hasattr(self, 'session_id') and self.session_id is not None and hasattr(self, 'db'):
            try:
                self.db.end_session(self.session_id, notes="Session ended normally")
                logger.info(f"Ended database session {self.session_id}")
            except Exception as e:
                logger.error(f"Error ending database session: {e}")

        # First stop the auto-analysis timer
        if self.auto_analysis_timer and self.auto_analysis_timer.isActive():
            logger.info("Stopping auto-analysis timer")
            self.auto_analysis_timer.stop()

        # Stop any ongoing analysis
        if self.analysis_thread and self.analysis_thread.isRunning():
            logger.info("Stopping analysis thread")
            self.analysis_thread.quit()
            self.analysis_thread.wait()

        # Stop other timers
        if hasattr(self, 'buffer_update_timer') and self.buffer_update_timer and self.buffer_update_timer.isActive():
            self.buffer_update_timer.stop()
        if hasattr(self, 'debug_timer') and self.debug_timer and self.debug_timer.isActive():
            self.debug_timer.stop()

        # Remove log observer
        plugin_log_handler.remove_observer(self._on_log_update)

        logger.info("BirdNET plugin shutdown")
        super().shutdown()

    def _on_audio_processor_update(self, results):
        """
        Handle audio processor updates directly.

        Args:
            results: Results from audio processor
        """
        if 'waveform' in results:
            waveform = results['waveform']
            if len(waveform) > 0:
                # Only log occasionally to avoid flooding
                if time.time() - self.last_analysis_time > 2.0:
                    rms = np.sqrt(np.mean(np.square(waveform)))
                    logger.debug(f"Direct audio processor update: {len(waveform)} samples, RMS={rms:.5f}")

                # Add to buffer directly
                self.buffer = np.append(self.buffer, waveform)

                # Limit buffer size
                max_samples = int(self.detection_window * self.sample_rate)
                if len(self.buffer) > max_samples:
                    self.buffer = self.buffer[-max_samples:]

                # Set recording to true to ensure automatic analysis works
                self.is_recording = True

    def _init_database(self):
        """
        Initialize the detection database connection.
        """
        # Get database instance
        self.db = get_db_instance()

        # Start a new session when the plugin initializes
        if not hasattr(self, 'session_id') or self.session_id is None:
            notes = f"Session started with {BIRDNET_TYPE}. "
            notes += f"Confidence threshold: {self.confidence_threshold}, "
            notes += f"Analysis interval: {self.analysis_interval}s, "
            notes += f"Detection window: {self.detection_window}s"

            # Add location if enabled
            if self.use_location and self.latitude is not None and self.longitude is not None:
                self.session_id = self.db.start_session(
                    latitude=self.latitude,
                    longitude=self.longitude,
                    notes=notes
                )
            else:
                self.session_id = self.db.start_session(notes=notes)

            logger.info(f"Started detection database session with ID {self.session_id}")

    def _save_detection_to_db(self, scientific_name, common_name, confidence, audio_data):
        """
        Save a detection to the database.

        Args:
            scientific_name: Scientific name of the bird species
            common_name: Common name of the bird species
            confidence: Detection confidence score (0.0 to 1.0)
            audio_data: Audio data as numpy array
        """
        # Make sure we have a session
        if not hasattr(self, 'session_id') or self.session_id is None:
            self._init_database()

        try:
            # Get latitude and longitude if available
            lat = self.latitude if self.use_location else None
            lon = self.longitude if self.use_location else None

            # Save to database
            detection_id = self.db.add_detection(
                session_id=self.session_id,
                scientific_name=scientific_name,
                common_name=common_name,
                confidence=confidence,
                audio_data=audio_data,
                sample_rate=self.sample_rate,
                latitude=lat,
                longitude=lon
            )

            logger.info(f"Saved detection to database with ID {detection_id}")

            # Update status message if we have a UI
            if self.status_label:
                self.status_label.setText(f"Status: Saved detection ID {detection_id} to database")

        except Exception as e:
            logger.error(f"Error saving detection to database: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Show error in UI
            if self.status_label:
                self.status_label.setText(f"Status: Error saving to database")

    def _start_new_db_session(self):
        """
        Start a new database session.
        """
        # End the current session if any
        if hasattr(self, 'session_id') and self.session_id is not None and hasattr(self, 'db') and self.db is not None:
            try:
                self.db.end_session(self.session_id, notes="Session ended by user")
                logger.info(f"Ended database session {self.session_id}")
            except Exception as e:
                logger.error(f"Error ending database session: {e}")

        # Start a new session
        try:
            notes = f"Session started with {BIRDNET_TYPE}. "
            notes += f"Confidence threshold: {self.confidence_threshold}, "
            notes += f"Analysis interval: {self.analysis_interval}s, "
            notes += f"Detection window: {self.detection_window}s"

            # Add location if enabled
            if self.use_location and self.latitude is not None and self.longitude is not None:
                self.session_id = self.db.start_session(
                    latitude=self.latitude,
                    longitude=self.longitude,
                    notes=notes
                )
            else:
                self.session_id = self.db.start_session(notes=notes)

            logger.info(f"Started detection database session with ID {self.session_id}")

            # Update UI
            if hasattr(self, 'db_status_label') and self.db_status_label is not None:
                self.db_status_label.setText(f"Database: Active (Session ID: {self.session_id})")
                self.db_status_label.setStyleSheet("color: green;")

            if self.status_label:
                self.status_label.setText(f"Status: Started new database session {self.session_id}")

        except Exception as e:
            logger.error(f"Error starting new database session: {e}")

            # Update UI
            if hasattr(self, 'db_status_label') and self.db_status_label is not None:
                self.db_status_label.setText(f"Database: Error starting session")
                self.db_status_label.setStyleSheet("color: red;")


    def _show_detections(self):
        """
        Show a dialog with all detections in the current session.
        """
        if not hasattr(self, 'db') or self.db is None:
            logger.error("Cannot show detections - database not initialized")
            if self.status_label:
                self.status_label.setText("Status: Database not initialized")
            return

        try:
            # Create a dialog to show detections
            dialog = QDialog(None)
            dialog.setWindowTitle("Bird Detections")
            dialog.resize(800, 600)

            layout = QVBoxLayout(dialog)

            # Add session selector
            session_layout = QHBoxLayout()
            session_layout.addWidget(QLabel("Session:"))

            session_combo = QComboBox()
            session_combo.addItem("All Sessions", None)

            # Get all sessions
            sessions = self.db.get_sessions()
            for session in sessions:
                session_id = session['session_id']
                start_time = session['start_time']
                detection_count = session['detection_count']
                label = f"Session {session_id} - {start_time} ({detection_count} detections)"
                session_combo.addItem(label, session_id)

                # Select current session
                if hasattr(self, 'session_id') and self.session_id == session_id:
                    session_combo.setCurrentIndex(session_combo.count() - 1)

            session_layout.addWidget(session_combo)

            # Add confidence filter
            session_layout.addWidget(QLabel("Min Confidence:"))
            confidence_spin = QDoubleSpinBox()
            confidence_spin.setMinimum(0.0)
            confidence_spin.setMaximum(1.0)
            confidence_spin.setSingleStep(0.05)
            confidence_spin.setValue(0.0)
            confidence_spin.setDecimals(2)
            session_layout.addWidget(confidence_spin)

            # Add filter button
            filter_btn = QPushButton("Filter")
            session_layout.addWidget(filter_btn)

            layout.addLayout(session_layout)

            # Add table of detections
            detections_table = QTableWidget()
            detections_table.setColumnCount(6)
            detections_table.setHorizontalHeaderLabels([
                "ID", "Time", "Scientific Name", "Common Name", "Confidence", "Audio"
            ])

            # Set column widths
            header = detections_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.Stretch)
            header.setSectionResizeMode(3, QHeaderView.Stretch)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

            layout.addWidget(detections_table)

            # Add play controls
            play_layout = QHBoxLayout()
            play_btn = QPushButton("Play Selected")
            play_layout.addWidget(play_btn)

            delete_btn = QPushButton("Delete Selected")
            play_layout.addWidget(delete_btn)

            export_selected_btn = QPushButton("Export Selected")
            play_layout.addWidget(export_selected_btn)

            layout.addLayout(play_layout)

            # Function to update the table
            def update_table():
                # Get selected session ID
                session_id = session_combo.currentData()

                # Get min confidence
                min_confidence = confidence_spin.value()

                # Get detections
                detections = self.db.get_detections(
                    session_id=session_id,
                    min_confidence=min_confidence
                )

                # Update table
                detections_table.setRowCount(len(detections))

                for row, detection in enumerate(detections):
                    # ID
                    id_item = QTableWidgetItem(str(detection['detection_id']))
                    detections_table.setItem(row, 0, id_item)

                    # Time
                    time_str = detection['timestamp'].split(' ')[1] if ' ' in detection['timestamp'] else detection['timestamp']
                    time_item = QTableWidgetItem(time_str)
                    detections_table.setItem(row, 1, time_item)

                    # Scientific Name
                    scientific_item = QTableWidgetItem(detection['scientific_name'])
                    detections_table.setItem(row, 2, scientific_item)

                    # Common Name
                    common_item = QTableWidgetItem(detection['common_name'])
                    detections_table.setItem(row, 3, common_item)

                    # Confidence
                    conf_str = f"{detection['confidence']:.2f}"
                    conf_item = QTableWidgetItem(conf_str)
                    detections_table.setItem(row, 4, conf_item)

                    # Audio button
                    audio_btn = QPushButton("Play")
                    audio_btn.setProperty("detection_id", detection['detection_id'])
                    audio_btn.clicked.connect(lambda checked, d_id=detection['detection_id']: play_audio(d_id))
                    detections_table.setCellWidget(row, 5, audio_btn)

                    # Color code by confidence
                    confidence = detection['confidence']
                    color = self.get_confidence_color(confidence)
                    for col in range(5):  # Don't color the button
                        detections_table.item(row, col).setBackground(color)

            # Function to play audio
            def play_audio(detection_id):
                try:
                    audio_path = self.db.get_audio_path(detection_id)
                    if audio_path and os.path.exists(audio_path):
                        # Use system default audio player
                        import platform
                        import subprocess

                        if platform.system() == 'Windows':
                            os.startfile(audio_path)
                        elif platform.system() == 'Darwin':  # macOS
                            subprocess.call(['open', audio_path])
                        elif platform.system() == 'Linux':
                            subprocess.call(['xdg-open', audio_path])

                        logger.info(f"Playing audio clip: {audio_path}")
                    else:
                        logger.warning(f"Audio file not found for detection ID {detection_id}")
                        QMessageBox.warning(dialog, "Audio Not Found", f"Audio file for detection ID {detection_id} not found")
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")
                    QMessageBox.critical(dialog, "Error", f"Error playing audio: {e}")

            # Function to delete selected detections
            def delete_selected():
                selected_rows = set()
                for index in detections_table.selectedIndexes():
                    selected_rows.add(index.row())

                if not selected_rows:
                    QMessageBox.information(dialog, "No Selection", "Please select one or more detections to delete")
                    return

                # Confirm deletion
                confirm = QMessageBox.question(
                    dialog,
                    "Confirm Deletion",
                    f"Are you sure you want to delete {len(selected_rows)} detection(s)?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if confirm != QMessageBox.Yes:
                    return

                # Delete detections
                deleted_count = 0
                for row in sorted(selected_rows, reverse=True):  # Delete in reverse order
                    detection_id = int(detections_table.item(row, 0).text())
                    if self.db.delete_detection(detection_id):
                        deleted_count += 1

                # Update table
                update_table()

                # Show result
                QMessageBox.information(dialog, "Deletion Complete", f"Deleted {deleted_count} detection(s)")

            # Function to export selected detections
            def export_selected():
                selected_rows = set()
                for index in detections_table.selectedIndexes():
                    selected_rows.add(index.row())

                if not selected_rows:
                    QMessageBox.information(dialog, "No Selection", "Please select one or more detections to export")
                    return

                # Get detection IDs
                detection_ids = []
                for row in selected_rows:
                    detection_id = int(detections_table.item(row, 0).text())
                    detection_ids.append(detection_id)

                # Export to CSV
                filepath, _ = QFileDialog.getSaveFileName(
                    dialog,
                    "Save Detections to CSV",
                    "",
                    "CSV files (*.csv);;All files (*)"
                )

                if not filepath:
                    return

                # Ensure .csv extension
                if not filepath.lower().endswith('.csv'):
                    filepath += '.csv'

                try:
                    # Create temporary table with selected detections
                    conn = sqlite3.connect(self.db.db_path)
                    cursor = conn.cursor()

                    # Create temp view
                    cursor.execute('''
                    CREATE TEMP VIEW selected_detections AS
                    SELECT * FROM bird_detections
                    WHERE detection_id IN ({})
                    '''.format(','.join(['?'] * len(detection_ids))), detection_ids)

                    # Export to CSV
                    import csv

                    cursor.execute('''
                    SELECT d.detection_id, d.session_id, d.timestamp,
                           d.scientific_name, d.common_name, d.confidence,
                           d.audio_file, d.latitude, d.longitude,
                           s.start_time as session_start
                    FROM selected_detections d
                    JOIN detection_sessions s ON d.session_id = s.session_id
                    ORDER BY d.timestamp DESC
                    ''')

                    # Get column names from cursor description
                    columns = [desc[0] for desc in cursor.description]

                    # Fetch all rows
                    rows = cursor.fetchall()

                    # Write to CSV
                    with open(filepath, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(columns)
                        writer.writerows(rows)

                    # Close connection
                    conn.close()

                    # Show result
                    QMessageBox.information(
                        dialog,
                        "Export Complete",
                        f"Exported {len(rows)} detection(s) to {filepath}"
                    )

                except Exception as e:
                    logger.error(f"Error exporting detections: {e}")
                    QMessageBox.critical(dialog, "Error", f"Error exporting detections: {e}")

            # Connect signals
            filter_btn.clicked.connect(update_table)
            delete_btn.clicked.connect(delete_selected)
            export_selected_btn.clicked.connect(export_selected)

            # Initial update
            update_table()

            # Show dialog
            dialog.exec_()

        except Exception as e:
            logger.error(f"Error showing detections: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _export_detections(self):
        """
        Export all detections in the current session to a CSV file.
        """
        if not hasattr(self, 'db') or self.db is None:
            logger.error("Cannot export detections - database not initialized")
            if self.status_label:
                self.status_label.setText("Status: Database not initialized")
            return

        if not hasattr(self, 'session_id') or self.session_id is None:
            logger.error("Cannot export detections - no active session")
            if self.status_label:
                self.status_label.setText("Status: No active session")
            return

        try:
            # Get file path from dialog
            filepath, _ = QFileDialog.getSaveFileName(
                None,
                "Save Detections to CSV",
                "",
                "CSV files (*.csv);;All files (*)"
            )

            if not filepath:
                return

            # Ensure .csv extension
            if not filepath.lower().endswith('.csv'):
                filepath += '.csv'

            # Export to CSV
            count = self.db.export_detections_csv(filepath, self.session_id)

            if count > 0:
                logger.info(f"Exported {count} detections to {filepath}")
                if self.status_label:
                    self.status_label.setText(f"Status: Exported {count} detections to CSV")

                # Show success message
                QMessageBox.information(
                    None,
                    "Export Complete",
                    f"Exported {count} detections to {filepath}"
                )
            else:
                logger.warning("No detections to export")
                if self.status_label:
                    self.status_label.setText("Status: No detections to export")

                # Show warning message
                QMessageBox.warning(
                    None,
                    "Export Warning",
                    "No detections found to export"
                )

        except Exception as e:
            logger.error(f"Error exporting detections: {e}")
            QMessageBox.critical(
                None,
                "Export Error",
                f"Error exporting detections: {e}"
            )
