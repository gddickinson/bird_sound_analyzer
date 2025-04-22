"""
Main window for Sound Analyzer application.
"""
import os
import sys
import logging
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QAction, QFileDialog,
    QSplitter, QTabWidget, QStatusBar, QToolBar, QDockWidget,
    QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QIcon

from core.audio_capture import AudioCapture
from core.audio_processor import AudioProcessor
from gui.widgets.spectrogram import SpectrogramWidget
from gui.widgets.waveform import WaveformWidget

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    Main application window for Sound Analyzer.
    """

    def __init__(self, config, plugin_manager):
        """
        Initialize the main window.

        Args:
            config: Application configuration
            plugin_manager: Plugin manager instance
        """
        super().__init__()

        self.config = config
        self.plugin_manager = plugin_manager

        # Initialize audio systems
        self.init_audio_systems()

        # Set up the UI
        self.init_ui()

        # Initialize plugins
        self.plugin_manager.initialize_plugins(self.audio_processor, self)

        # Start a timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(50)  # Update every 50ms

        logger.info("Main window initialized")

    def init_audio_systems(self):
        """
        Initialize audio capture and processing systems.
        """
        # Create audio capture and processor
        self.audio_capture = AudioCapture(self.config.get('audio', {}))
        self.audio_processor = AudioProcessor(self.config.get('audio', {}))

        # Connect audio capture to processor
        def process_audio(audio_data, sample_rate):
            self.audio_processor.process_audio(audio_data, sample_rate)

        self.audio_capture.register_callback(process_audio)

        logger.info("Audio systems initialized")

    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Set window properties
        self.setWindowTitle("Sound Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create plugins dock first (before menu creation)
        self.plugins_dock = QDockWidget("Plugins", self)
        self.plugins_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plugins_dock)

        # Create tab widget for plugins
        self.plugins_tabs = QTabWidget()
        self.plugins_dock.setWidget(self.plugins_tabs)

        # Create toolbar
        self.create_toolbar()

        # Create menu
        self.create_menu()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # Create visualization widgets
        self.waveform_widget = WaveformWidget(self.audio_processor)
        self.spectrogram_widget = SpectrogramWidget(self.audio_processor)

        # Add visualization widgets to splitter
        splitter.addWidget(self.waveform_widget)
        splitter.addWidget(self.spectrogram_widget)

        # Set initial splitter sizes
        splitter.setSizes([200, 600])

        # Create control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        logger.info("UI initialized")

    def create_toolbar(self):
        """
        Create the toolbar.
        """
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Add toolbar buttons
        self.action_record = QAction("Record", self)
        self.action_record.setCheckable(True)
        self.action_record.triggered.connect(self.toggle_recording)
        toolbar.addAction(self.action_record)

        self.action_open = QAction("Open File", self)
        self.action_open.triggered.connect(self.open_audio_file)
        toolbar.addAction(self.action_open)

        # Add device selection combobox
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Input Device:"))

        self.device_combo = QComboBox()
        self.update_device_list()
        self.device_combo.currentIndexChanged.connect(self.change_input_device)
        toolbar.addWidget(self.device_combo)

    def create_menu(self):
        """
        Create the menu bar.
        """
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        toggle_plugins_action = self.plugins_dock.toggleViewAction()
        toggle_plugins_action.setShortcut("Ctrl+P")
        view_menu.addAction(toggle_plugins_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def create_control_panel(self):
        """
        Create the control panel.

        Returns:
            QWidget containing the control panel
        """
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        # Open file button
        open_button = QPushButton("Open File")
        open_button.clicked.connect(self.open_audio_file)
        layout.addWidget(open_button)

        # Add status label
        layout.addStretch()
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        return panel

    def update_device_list(self):
        """
        Update the input device list in the combobox.
        """
        self.device_combo.clear()

        for device_index, device_name in self.audio_capture.get_device_list():
            self.device_combo.addItem(device_name, device_index)

        # Set current device
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == self.audio_capture.device_index:
                self.device_combo.setCurrentIndex(i)
                break

    def change_input_device(self, index):
        """
        Change the audio input device.

        Args:
            index: Index in the combobox
        """
        if index >= 0:
            device_index = self.device_combo.itemData(index)
            self.audio_capture.set_device(device_index)
            self.status_bar.showMessage(f"Input device changed to: {self.device_combo.itemText(index)}")

    def toggle_recording(self):
        """
        Start or stop recording based on current state.
        """
        if self.audio_capture.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """
        Start recording audio.
        """
        self.audio_capture.start_recording()

        if self.audio_capture.is_recording:
            self.record_button.setText("Stop Recording")
            self.action_record.setChecked(True)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Recording...")
            self.status_bar.showMessage("Recording audio...")
        else:
            QMessageBox.warning(self, "Recording Error",
                                "Could not start recording. Check the selected audio device.")

    def stop_recording(self):
        """
        Stop recording audio.
        """
        self.audio_capture.stop_recording()
        self.record_button.setText("Start Recording")
        self.action_record.setChecked(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Ready")
        self.status_bar.showMessage("Recording stopped")

    def stop_audio(self):
        """
        Stop all audio (recording or playback).
        """
        if self.audio_capture.is_recording:
            self.stop_recording()

        # Add code to stop playback when implemented

        self.status_bar.showMessage("Stopped")

    def open_audio_file(self):
        """
        Open an audio file for analysis.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )

        if file_path:
            # Stop current recording if active
            if self.audio_capture.is_recording:
                self.stop_recording()

            # Load and play the file
            success = self.audio_capture.play_audio_file(
                file_path,
                lambda progress: self.status_bar.showMessage(f"Playing: {int(progress * 100)}%")
            )

            if success:
                self.status_label.setText(f"Playing: {os.path.basename(file_path)}")
                self.stop_button.setEnabled(True)
                self.status_bar.showMessage(f"Playing: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "Error", f"Could not open the audio file: {file_path}")

    def add_plugin_tab(self, plugin_id, widget, name):
        """
        Add a tab for a plugin.

        Args:
            plugin_id: The plugin identifier
            widget: The plugin's widget
            name: The display name for the tab
        """
        if widget:
            self.plugins_tabs.addTab(widget, name)

    def update_ui(self):
        """
        Update UI components periodically.
        This is called by the timer.
        """
        # Nothing to do here for now, the visualization widgets update themselves
        pass

    def show_about_dialog(self):
        """
        Show the about dialog.
        """
        QMessageBox.about(
            self,
            "About Sound Analyzer",
            "Sound Analyzer\n"
            "A real-time sound analysis application with plugin support.\n"
            "Version: 0.1.0\n"
            "Created with Python and PyQt5."
        )

    def closeEvent(self, event):
        """
        Handle window close event.

        Args:
            event: Close event
        """
        # Stop audio processing
        if self.audio_capture.is_recording:
            self.audio_capture.stop_recording()

        # Clean up audio resources
        self.audio_capture.cleanup()

        # Accept the close event
        event.accept()
