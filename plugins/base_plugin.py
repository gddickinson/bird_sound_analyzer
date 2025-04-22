"""
Base plugin class for Sound Analyzer.
All plugins must inherit from this class.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BasePlugin(ABC):
    """
    Base class for all sound analyzer plugins.

    Plugins must implement the abstract methods to process audio
    and manage UI integration.
    """

    def __init__(self, audio_processor, config: Dict[str, Any]):
        """
        Initialize the plugin.

        Args:
            audio_processor: The audio processor instance
            config: Plugin-specific configuration
        """
        self.audio_processor = audio_processor
        self.config = config
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"plugins.{self.name}")
        self.logger.info(f"Initializing plugin: {self.name}")

    @abstractmethod
    def process_audio(self, audio_chunk, sample_rate):
        """
        Process an audio chunk.

        Args:
            audio_chunk: The audio data (numpy array)
            sample_rate: The audio sample rate in Hz

        Returns:
            Any results from processing the audio
        """
        pass

    @abstractmethod
    def initialize_ui(self, main_window):
        """
        Initialize the plugin's UI components.

        Args:
            main_window: The main application window
        """
        pass

    def get_settings_widget(self):
        """
        Get a widget for plugin settings.

        Returns:
            A QWidget for plugin settings, or None if not needed
        """
        return None

    def get_visualization_widget(self):
        """
        Get a widget for plugin visualization.

        Returns:
            A QWidget for plugin visualization, or None if not needed
        """
        return None

    def shutdown(self):
        """
        Perform any cleanup when shutting down the plugin.
        """
        self.logger.info(f"Shutting down plugin: {self.name}")
