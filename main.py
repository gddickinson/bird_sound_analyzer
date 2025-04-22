#!/usr/bin/env python3
"""
Sound Analyzer - Real-time audio visualization and analysis.
Main entry point for the application.
"""
import sys
import logging
from PyQt5.QtWidgets import QApplication

from core.plugin_manager import PluginManager
from gui.main_window import MainWindow
from utils.config import load_config
from utils.logging_setup import setup_logging

def main():
    """Initialize and start the application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Sound Analyzer application")

    # Load configuration
    config = load_config()

    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("Sound Analyzer")

    # Initialize plugin manager
    plugin_manager = PluginManager(config.get('plugins', {}))

    # Create and show the main window
    window = MainWindow(config, plugin_manager)
    window.show()

    # Start the application event loop
    sys.exit(app.exec_())

def check_microphone():
    """Test if the microphone is working."""
    import pyaudio
    import wave
    import numpy as np

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 3

    p = pyaudio.PyAudio()

    print("Checking microphone...")

    # List available devices
    print("\nAvailable input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device {i}: {device_info.get('name')}")

    # Try to record from default device
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("\nRecording test sample...")
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

            # Check if we're getting audio
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            print(f"Chunk {i}: RMS = {rms:.2f}")

        print("Finished recording test sample")

        stream.stop_stream()
        stream.close()

        # Save test file
        test_file = "microphone_test.wav"
        wf = wave.open(test_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"Test recording saved to {test_file}")

    except Exception as e:
        print(f"Error testing microphone: {e}")

    p.terminate()


if __name__ == "__main__":
    # Uncomment to test:
    #check_microphone()
    main()
