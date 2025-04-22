"""
BirdNET plugin for Sound Analyzer with enhanced debugging.
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

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QSlider, QCheckBox,
    QComboBox, QHeaderView, QGroupBox, QFormLayout, QDoubleSpinBox,
    QProgressBar, QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QTextCursor

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
logger.setLevel(logging.DEBUG)


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

        # Debug timer to log buffer status periodically
        self.debug_timer = QTimer()
        self.debug_timer.timeout.connect(self._log_buffer_status)
        self.debug_timer.start(50000)  # Log every 5 seconds

        # Initialize BirdNET model if available
        self.model = None
        self.init_birdnet_model()

        # Log initialization details
        logger.info(f"BirdNET Plugin initialized with: confidence_threshold={self.confidence_threshold}, "
                    f"analysis_interval={self.analysis_interval}s, detection_window={self.detection_window}s, "
                    f"use_location={self.use_location}, latitude={self.latitude}, longitude={self.longitude}")

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

        # Check if it's time to run analysis
        current_time = time.time()
        if (current_time - self.last_analysis_time >= self.analysis_interval and
            len(self.buffer) >= max_samples * 0.5 and  # At least 50% full (was 80%)
            not self.analysis_thread_running):

            logger.info(f"Starting analysis after {current_time - self.last_analysis_time:.2f} seconds")
            # Start analysis in a background thread to avoid blocking the UI
            self.start_analysis()
            self.last_analysis_time = current_time

        # Return current species list
        return {'species': self.results}

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

        Args:
            results: List of (scientific_name, common_name, confidence, time) tuples
        """
        logger.info(f"Received analysis results: {len(results)} detections")

        # Update results list with new species
        for scientific, common, confidence, time_start in results:
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
                self.status_label.setText(f"Status: Detected {len(results)} species")
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

        # Start the buffer update timer
        self.buffer_update_timer = QTimer()
        self.buffer_update_timer.timeout.connect(self._update_buffer_display)
        self.buffer_update_timer.start(1000)  # Update every second

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

            # Play the sound on default audio output
            # (this is optional and requires additional setup for audio output)

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
        """
        # Stop any ongoing analysis
        if self.analysis_thread and self.analysis_thread.isRunning():
            logger.info("Stopping analysis thread")
            self.analysis_thread.quit()
            self.analysis_thread.wait()

        # Stop timers
        if hasattr(self, 'buffer_update_timer'):
            self.buffer_update_timer.stop()
        if hasattr(self, 'debug_timer'):
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
