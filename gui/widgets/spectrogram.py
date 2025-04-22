"""
Spectrogram visualization widget for Sound Analyzer.
"""
import numpy as np
import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt
import pyqtgraph as pg

logger = logging.getLogger(__name__)

class SpectrogramWidget(QWidget):
    """
    Widget for visualizing audio spectrograms.
    """

    def __init__(self, audio_processor):
        """
        Initialize the spectrogram widget.

        Args:
            audio_processor: The audio processor instance
        """
        super().__init__()

        self.audio_processor = audio_processor

        # Initialize display settings
        self.min_freq = 0  # Hz
        self.max_freq = audio_processor.sample_rate // 2  # Nyquist frequency
        self.min_intensity = -100  # dB
        self.max_intensity = 0  # dB

        # Initialize spectrogram image data
        self.history_size = 100  # Number of time frames to display
        self.spectrogram_data = np.zeros((1, self.history_size))
        self.frequencies = np.zeros(1)

        # Create basic colormaps
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.color_maps = {
            "Grayscale": pg.ColorMap(pos, np.array([[0, 0, 0, 255],
                                                    [64, 64, 64, 255],
                                                    [128, 128, 128, 255],
                                                    [192, 192, 192, 255],
                                                    [255, 255, 255, 255]])),

            "Thermal": pg.ColorMap(pos, np.array([[0, 0, 0, 255],
                                                  [128, 0, 0, 255],
                                                  [255, 128, 0, 255],
                                                  [255, 255, 0, 255],
                                                  [255, 255, 255, 255]])),

            "Flame": pg.ColorMap(pos, np.array([[0, 0, 0, 255],
                                                [128, 0, 0, 255],
                                                [255, 0, 0, 255],
                                                [255, 128, 0, 255],
                                                [255, 255, 0, 255]])),

            "Blues": pg.ColorMap(pos, np.array([[0, 0, 64, 255],
                                                [0, 0, 128, 255],
                                                [0, 64, 192, 255],
                                                [0, 128, 255, 255],
                                                [128, 255, 255, 255]])),

            "Green": pg.ColorMap(pos, np.array([[0, 0, 0, 255],
                                                [0, 64, 0, 255],
                                                [0, 128, 0, 255],
                                                [0, 192, 0, 255],
                                                [0, 255, 0, 255]]))
        }
        self.current_color_map = "Thermal"

        # Set up the UI
        self.init_ui()

        # Register for audio processing updates
        self.audio_processor.register_callback(self.update_spectrogram)

        logger.info("Spectrogram widget initialized")

    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Title
        title_label = QLabel("Spectrogram")
        title_label.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(title_label)

        # Add color map selector
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Color Map:"))

        self.color_map_combo = QComboBox()
        for map_name in self.color_maps.keys():
            self.color_map_combo.addItem(map_name)
        self.color_map_combo.setCurrentText(self.current_color_map)
        self.color_map_combo.currentTextChanged.connect(self.change_color_map)
        controls_layout.addWidget(self.color_map_combo)

        layout.addLayout(controls_layout)

        # Create image view for spectrogram
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()  # Hide the histogram
        self.image_view.ui.roiBtn.hide()     # Hide the ROI button
        self.image_view.ui.menuBtn.hide()    # Hide the menu button
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Set the color map
        self.set_color_map(self.current_color_map)

        # Add the image view to the layout
        layout.addWidget(self.image_view)

        # Set minimum height
        self.setMinimumHeight(200)

    def set_color_map(self, map_name):
        """
        Set the color map for the spectrogram.

        Args:
            map_name: Name of the color map to use
        """
        if map_name in self.color_maps:
            cmap = self.color_maps[map_name]
            self.image_view.setColorMap(cmap)
            self.current_color_map = map_name

    def change_color_map(self, map_name):
        """
        Change the color map based on user selection.

        Args:
            map_name: Name of the selected color map
        """
        self.set_color_map(map_name)
        self.update_image()

    def update_spectrogram(self, results):
        """
        Update the spectrogram with new audio data.

        Args:
            results: Audio processing results
        """
        if 'spectrogram' in results and 'spectrogram_frequencies' in results:
            # Get new spectrogram data
            new_spectrogram = results['spectrogram']
            self.frequencies = results['spectrogram_frequencies']

            # Check if this is the first valid spectrogram
            if self.spectrogram_data.shape[0] == 1:
                # Initialize with proper shape based on first spectrogram
                freq_bins = new_spectrogram.shape[0]
                self.spectrogram_data = np.zeros((freq_bins, self.history_size))
                self.spectrogram_data = self.spectrogram_data - 100  # Initialize with low dB values

            # Add new column(s) to the spectrogram data
            num_new_cols = new_spectrogram.shape[1]

            if num_new_cols >= self.history_size:
                # If new data is larger than history, take the most recent columns
                self.spectrogram_data = new_spectrogram[:, -self.history_size:]
            else:
                # Shift columns left and add new columns
                self.spectrogram_data = np.roll(self.spectrogram_data, -num_new_cols, axis=1)
                self.spectrogram_data[:, -num_new_cols:] = new_spectrogram

            # Update the display
            self.update_image()

    def update_image(self):
        """
        Update the image view with current spectrogram data.
        """
        if self.frequencies.size > 1:
            # Check frequency range (only display frequencies within the range)
            freq_mask = (self.frequencies >= self.min_freq) & (self.frequencies <= self.max_freq)
            display_data = self.spectrogram_data[freq_mask, :]

            # Clip data to intensity range
            display_data = np.clip(display_data, self.min_intensity, self.max_intensity)

            # Update the image (transpose to get frequency on y-axis)
            self.image_view.setImage(
                display_data.T,  # Transpose to get frequency on Y-axis
                autoLevels=False,
                levels=[self.min_intensity, self.max_intensity]
            )

    def set_frequency_range(self, min_freq, max_freq):
        """
        Set the frequency range to display.

        Args:
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.update_image()

    def set_intensity_range(self, min_intensity, max_intensity):
        """
        Set the intensity range to display.

        Args:
            min_intensity: Minimum intensity in dB
            max_intensity: Maximum intensity in dB
        """
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.update_image()

    def clear(self):
        """
        Clear the spectrogram display.
        """
        if self.spectrogram_data.shape[0] > 1:
            self.spectrogram_data = np.zeros_like(self.spectrogram_data) - 100
            self.update_image()
