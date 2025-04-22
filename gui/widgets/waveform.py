"""
Waveform visualization widget for Sound Analyzer.
"""
import numpy as np
import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg

logger = logging.getLogger(__name__)

class WaveformWidget(QWidget):
    """
    Widget for visualizing audio waveforms.
    """
    
    def __init__(self, audio_processor):
        """
        Initialize the waveform widget.
        
        Args:
            audio_processor: The audio processor instance
        """
        super().__init__()
        
        self.audio_processor = audio_processor
        
        # Initialize plot data
        self.buffer_size = 1000  # Number of samples to display
        self.plot_data = np.zeros(self.buffer_size)
        
        # Set up the UI
        self.init_ui()
        
        # Register for audio processing updates
        self.audio_processor.register_callback(self.update_plot_data)
        
        logger.info("Waveform widget initialized")
    
    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("Waveform")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setYRange(-1, 1)  # Audio data normalized to [-1, 1]
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create the plot curve
        pen = pg.mkPen(color='b', width=1)
        self.plot_curve = self.plot_widget.plot(self.plot_data, pen=pen)
        
        # Add the plot widget to the layout
        layout.addWidget(self.plot_widget)
        
        # Set minimum height
        self.setMinimumHeight(150)
    
    def update_plot_data(self, results):
        """
        Update the plot with new audio data.
        
        Args:
            results: Audio processing results
        """
        if 'waveform' in results:
            # Get new data
            new_data = results['waveform']
            
            # Update the plot data buffer
            if len(new_data) >= self.buffer_size:
                # If new data is larger than buffer, take the most recent samples
                self.plot_data = new_data[-self.buffer_size:]
            else:
                # Shift the buffer and add new data
                self.plot_data = np.roll(self.plot_data, -len(new_data))
                self.plot_data[-len(new_data):] = new_data
            
            # Update the plot
            self.plot_curve.setData(self.plot_data)
    
    def clear(self):
        """
        Clear the plot.
        """
        self.plot_data = np.zeros(self.buffer_size)
        self.plot_curve.setData(self.plot_data)
