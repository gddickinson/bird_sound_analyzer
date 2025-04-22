# Sound Analyzer with BirdNET Integration

A real-time sound analysis application with bird species identification capabilities, built in Python using PyQt5 and integrating with BirdNET.

![Sound Analyzer Screenshot](docs/images/screenshot.png)

## Features

- **Real-time Audio Visualization**: Display live waveform and spectrogram visualizations
- **Bird Species Identification**: Automatically detect and identify bird species using BirdNET
- **Modular Plugin Architecture**: Extensible design allows adding new analysis capabilities
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Multiple Visualization Options**: Customizable spectrogram and waveform displays
- **Audio Recording**: Record and analyze live audio from a microphone
- **File Analysis**: Load and analyze audio files

## Installation

### Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- PyAudio (for microphone access)
- BirdNET or BirdNETlib package

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sound_analyzer.git
   cd sound_analyzer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install BirdNET dependencies:
   ```
   pip install birdnetlib resampy librosa
   ```

5. Run the application:
   ```
   python main.py
   ```

## Usage

### Main Interface

The application has three main areas:
- **Waveform Display**: Shows the audio amplitude over time
- **Spectrogram Display**: Shows the frequency content of the audio
- **Plugin Panel**: Contains plugin-specific controls and displays

### Recording Audio

1. Select your input device from the dropdown in the toolbar
2. Click "Start Recording" to begin capturing audio
3. Visualizations will update in real-time
4. Click "Stop Recording" to stop

### Using the BirdNET Plugin

1. Open the "BirdNET" tab in the plugin panel
2. Set the confidence threshold (lower values catch more species but with less certainty)
3. Optionally enable location filtering by checking "Use Location for Species Filtering"
4. Record audio containing bird sounds
5. Detected species will appear in the results table
6. Use "Force Analysis Now" to immediately analyze the current audio buffer
7. Use "Test With Bird Sound" to generate a test sound for verifying functionality

### Adjusting Settings

- **Confidence Threshold**: Controls how certain BirdNET must be before reporting a species
- **Analysis Interval**: Controls how often BirdNET analyzes the audio buffer
- **Detection Window**: Sets the length of audio analyzed each time
- **Location Settings**: Enables filtering for species likely to be in your geographic area

## Project Structure

```
sound_analyzer/
├── core/                 # Core functionality
│   ├── audio_capture.py  # Audio input handling
│   ├── audio_processor.py # Audio processing utilities
│   └── plugin_manager.py # Plugin management system
├── gui/                  # GUI components
│   ├── main_window.py    # Main application window
│   ├── visualization.py  # Visualization components
│   └── widgets/          # Reusable UI components
│       ├── spectrogram.py
│       └── waveform.py
├── plugins/              # Analysis plugins
│   ├── base_plugin.py    # Base class for plugins
│   └── birdnet/          # BirdNET plugin
│       ├── birdnet_plugin.py
│       └── models.py
├── utils/                # Utility functions
│   ├── config.py         # Configuration handling
│   └── logging_setup.py  # Logging configuration
├── logs/                 # Log files
├── main.py               # Application entry point
├── config.yaml           # Configuration file
└── requirements.txt      # Project dependencies
```

## Extending with Plugins

The application has a modular plugin architecture that allows adding new analysis capabilities. To create a new plugin:

1. Create a new directory under `plugins/`
2. Create a plugin class that inherits from `BasePlugin`
3. Implement required methods:
   - `process_audio`: Process audio data
   - `initialize_ui`: Set up the UI components
4. Register the plugin in `config.yaml` under `plugins.enabled`

## Troubleshooting

### No Audio Input

If you're not seeing any audio input:
1. Check that your microphone is connected and working
2. Select the correct input device from the dropdown
3. Check system permissions for microphone access
4. Try increasing the input volume

### BirdNET Not Detecting Birds

If BirdNET isn't detecting birds:
1. Lower the confidence threshold
2. Make sure the audio contains clear bird sounds
3. Try the "Test With Bird Sound" button to verify functionality
4. Check if you need to install additional dependencies (like `resampy`)
5. Enable location filtering with your correct coordinates

### Visualization Issues

If the visualizations aren't displaying properly:
1. Check if PyQtGraph is installed correctly
2. Make sure your graphics drivers are up to date
3. Try reducing the window size if display is slow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BirdNET for the bird species identification algorithm
- PyQt5 for the GUI framework
- PyQtGraph for the visualization components
- The K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
