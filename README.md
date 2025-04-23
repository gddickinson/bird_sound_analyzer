# Sound Analyzer with BirdNET Integration

A real-time audio analysis application with automatic bird species detection, built in Python using PyQt5 and integrated with BirdNET.

![Sound Analyzer Screenshot](docs/images/screenshot.png)

## Features

- **Real-time Audio Visualization**: Live waveform and spectrogram visualizations
- **Automatic Bird Species Identification**: Detect bird species in real-time using BirdNET
- **Detection Database**: Store and manage bird detections with associated audio clips
- **Audio Recording**: Record and analyze audio from microphone input
- **Multiple Visualization Options**: Customizable spectrogram and waveform displays
- **File Analysis**: Load and analyze audio files
- **Data Export**: Export detection data to CSV for further analysis
- **Location Tagging**: Tag detections with geographic coordinates for mapping

## Installation

### Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- PyAudio (for microphone access)
- PyQt5 (for the user interface)
- NumPy and SciPy (for signal processing)
- BirdNETlib package (for bird species detection)
- psutil (for memory monitoring)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/gddickinson/sound_analyzer.git
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
   pip install birdnetlib resampy librosa psutil
   ```

5. Run the application:
   ```
   python main.py
   ```

## Usage

### Main Interface

The application has four main areas:
- **Waveform Display**: Shows the audio amplitude over time
- **Spectrogram Display**: Shows the frequency content of the audio
- **BirdNET Plugin**: Controls for bird detection settings and displays results
- **Detection Database**: Manages stored bird detections and audio clips

### Recording Audio

1. Select your input device from the dropdown in the toolbar
2. Click "Start Recording" to begin capturing audio
3. The app will automatically analyze the audio for bird sounds
4. Detected species will appear in the results table
5. Click "Stop Recording" when finished

### Using the BirdNET Plugin

1. Adjust the confidence threshold slider (lower values catch more species but with less certainty)
2. Set the analysis interval (how often BirdNET analyzes the audio buffer)
3. Set the detection window (the size of the audio segment analyzed each time)
4. Enable location filtering by checking "Use Location for Species Filtering"
5. Start recording to automatically detect bird species
6. Use "Force Analysis Now" to immediately analyze the current audio buffer
7. Use "Test With Bird Sound" to generate a test sound for verifying functionality

### Managing Detections

1. All detections are automatically saved to the database with their audio clips
2. Click "View Detections" to browse, filter, and play saved detections
3. Export detections to CSV for analysis in other software
4. Start a new recording session with the "Start New Session" button
5. Delete unwanted or false detections through the detections browser

## Configuration

The `config.yaml` file contains settings for:

- **Audio parameters**: Sample rate, buffer size, FFT settings
- **BirdNET settings**: Confidence threshold, analysis interval, location
- **UI preferences**: Theme, window size
- **Plugin settings**: Enabled plugins and plugin-specific configurations

Customize these settings to optimize for your specific use case and hardware.

## Project Structure

```
sound_analyzer/
├── core/                 # Core functionality
│   ├── audio_capture.py  # Audio input handling
│   ├── audio_processor.py # Audio processing utilities
│   └── plugin_manager.py # Plugin management system
├── data/                 # Data storage
│   ├── audio_clips/      # Saved bird sound clips
│   └── detections.db     # SQLite database file
├── gui/                  # GUI components
│   ├── main_window.py    # Main application window
│   ├── visualization.py  # Visualization components
│   └── widgets/          # Reusable UI components
│       ├── spectrogram.py
│       └── waveform.py
├── plugins/              # Analysis plugins
│   ├── base_plugin.py    # Base class for plugins
│   └── birdnet/          # BirdNET plugin
│       └── birdnet_plugin.py
├── utils/                # Utility functions
│   ├── config.py         # Configuration handling
│   ├── detection_db.py   # Database management
│   └── logging_setup.py  # Logging configuration
├── logs/                 # Log files
├── main.py               # Application entry point
├── config.yaml           # Configuration file
└── requirements.txt      # Project dependencies
```

## Bird Detection Database

The application includes a SQLite database for storing and managing bird detections:

- **Detection Metadata**: Species, confidence, timestamp, location
- **Audio Storage**: Audio clips saved for each detection
- **Session Management**: Group detections into recording sessions
- **Export Capabilities**: Export detection data to CSV
- **Filtering**: Filter by species, confidence, date, or session

## Troubleshooting

### Audio Input Issues

If you're not seeing any audio input:
1. Check that your microphone is connected and working
2. Select the correct input device from the dropdown
3. Check system permissions for microphone access
4. Try increasing the input volume

### BirdNET Detection Issues

If BirdNET isn't detecting birds:
1. Ensure you're in an environment with bird sounds
2. Lower the confidence threshold (try 0.2 or lower for testing)
3. Increase the detection window to 5-10 seconds
4. Check if the audio level is too low (RMS value shown in the buffer display)
5. Try the "Test With Bird Sound" button to verify functionality

### Visualization Issues

If the visualizations aren't displaying properly:
1. Check if PyQtGraph is installed correctly
2. Make sure your graphics drivers are up to date
3. Try reducing the window size if display is slow

### Application Crashes

If the application crashes:
1. Check the log files in the `logs` directory for error messages
2. Ensure you have enough available memory (at least 2GB free)
3. Update all dependencies to the latest versions
4. Try reducing the detection window size
5. Close other memory-intensive applications

## Advanced Features

### Location-based Filtering

The BirdNET plugin can filter species based on geographic location and time of year:

1. Enable location filtering in the BirdNET settings
2. Set your latitude and longitude
3. The system will prioritize species likely to be in your area during the current season

### Customizing Detection Settings

For better detection in different environments:

- **Urban environments**: Lower the confidence threshold to 0.1-0.2, as bird sounds may be mixed with noise
- **Quiet natural settings**: Raise the confidence threshold to 0.3-0.5 for higher accuracy
- **Dawn chorus (many birds)**: Set a shorter analysis interval (1-2 seconds) to catch more species
- **Single bird songs**: Increase the detection window to 5-10 seconds for better species identification

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BirdNET for the bird species identification algorithm
- PyQt5 for the GUI framework
- PyQtGraph for the visualization components
- The K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology
