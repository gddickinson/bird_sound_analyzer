"""
Visualization module for Sound Analyzer.
Provides common utilities for audio visualization components.
"""
import numpy as np
import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter
from PyQt5.QtCore import Qt, pyqtSignal

from gui.widgets.waveform import WaveformWidget
from gui.widgets.spectrogram import SpectrogramWidget

logger = logging.getLogger(__name__)

class VisualizationPanel(QWidget):
    """
    Combined visualization panel that includes waveform and spectrogram.
    """
    
    def __init__(self, audio_processor):
        """
        Initialize the visualization panel.
        
        Args:
            audio_processor: The audio processor instance
        """
        super().__init__()
        
        self.audio_processor = audio_processor
        
        # Set up the UI
        self.init_ui()
        
        logger.info("Visualization panel initialized")
    
    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for adjustable views
        self.splitter = QSplitter(Qt.Vertical)
        
        # Create visualization widgets
        self.waveform_widget = WaveformWidget(self.audio_processor)
        self.spectrogram_widget = SpectrogramWidget(self.audio_processor)
        
        # Add widgets to splitter
        self.splitter.addWidget(self.waveform_widget)
        self.splitter.addWidget(self.spectrogram_widget)
        
        # Set initial sizes (30% waveform, 70% spectrogram)
        self.splitter.setSizes([300, 700])
        
        # Add splitter to layout
        layout.addWidget(self.splitter)
    
    def clear(self):
        """
        Clear all visualizations.
        """
        self.waveform_widget.clear()
        self.spectrogram_widget.clear()
    
    def set_frequency_range(self, min_freq, max_freq):
        """
        Set the frequency range for the spectrogram.
        
        Args:
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
        """
        self.spectrogram_widget.set_frequency_range(min_freq, max_freq)
    
    def set_intensity_range(self, min_intensity, max_intensity):
        """
        Set the intensity range for the spectrogram.
        
        Args:
            min_intensity: Minimum intensity in dB
            max_intensity: Maximum intensity in dB
        """
        self.spectrogram_widget.set_intensity_range(min_intensity, max_intensity)
    
    def set_color_map(self, color_map):
        """
        Set the color map for the spectrogram.
        
        Args:
            color_map: Name of the color map
        """
        self.spectrogram_widget.set_color_map(color_map)
