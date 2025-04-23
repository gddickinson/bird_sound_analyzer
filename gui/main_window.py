"""
Main window for Sound Analyzer application with enhanced audio player and speaker output.
"""
import os
import sys
import logging
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QAction, QFileDialog,
    QSplitter, QTabWidget, QStatusBar, QToolBar, QDockWidget,
    QMessageBox, QApplication, QSlider, QListWidget, QListWidgetItem,
    QFrame, QTreeView, QFileSystemModel, QGroupBox, QStyle, QSizePolicy,
    QCheckBox
)
from PyQt5.QtCore import Qt, QSettings, QTimer, QDir, QModelIndex, QTime
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

        # Track current file being played
        self.current_file = None
        self.playback_progress = 0
        self.playback_duration = 0
        self.is_playing = False

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

        # Load settings
        self.load_settings()

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

        # Main layout with vertical orientation
        main_layout = QVBoxLayout(central_widget)

        # Create plugins dock
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

        # Create a horizontal splitter for file browser and main content
        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, 1)  # Give it a stretch factor of 1

        # Create file browser panel
        file_browser = self.create_file_browser()
        h_splitter.addWidget(file_browser)

        # Create right panel (visualization + audio player)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Create visualization splitter
        viz_splitter = QSplitter(Qt.Vertical)

        # Create visualization widgets
        self.waveform_widget = WaveformWidget(self.audio_processor)
        self.spectrogram_widget = SpectrogramWidget(self.audio_processor)

        # Add visualization widgets to splitter
        viz_splitter.addWidget(self.waveform_widget)
        viz_splitter.addWidget(self.spectrogram_widget)

        # Set initial splitter sizes
        viz_splitter.setSizes([200, 600])

        # Add visualization splitter to right panel
        right_layout.addWidget(viz_splitter, 3)  # Give it a stretch factor of 3

        # Create audio player panel
        audio_player = self.create_audio_player()
        right_layout.addWidget(audio_player, 1)  # Give it a stretch factor of 1

        # Add right panel to horizontal splitter
        h_splitter.addWidget(right_panel)

        # Set initial horizontal splitter sizes (30% file browser, 70% right panel)
        h_splitter.setSizes([300, 700])

        # Create control panel (with record button, etc.)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 0)  # No stretch

        # Create speaker control panel
        speaker_panel = self.create_speaker_controls()
        main_layout.addWidget(speaker_panel, 0)  # No stretch

        logger.info("UI initialized")

    def create_file_browser(self):
        """
        Create a file browser panel.

        Returns:
            QWidget containing the file browser
        """
        # Create a group box for the file browser
        group_box = QGroupBox("File Browser")
        layout = QVBoxLayout(group_box)

        # Create a file system model
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(QDir.homePath())

        # Filter for audio files
        self.file_model.setNameFilters(["*.wav", "*.mp3", "*.flac", "*.ogg"])
        self.file_model.setNameFilterDisables(False)  # Hide non-matching files

        # Create a tree view for the file system
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(QDir.homePath()))

        # Hide unnecessary columns
        self.file_tree.setColumnHidden(1, True)  # Size
        self.file_tree.setColumnHidden(2, True)  # Type

        # Connect signals
        self.file_tree.doubleClicked.connect(self.on_file_double_clicked)

        # Add to layout
        layout.addWidget(self.file_tree)

        # Add quick access buttons
        quick_access = QHBoxLayout()

        home_btn = QPushButton("Home")
        home_btn.clicked.connect(lambda: self.file_tree.setRootIndex(
            self.file_model.index(QDir.homePath())
        ))
        quick_access.addWidget(home_btn)

        music_btn = QPushButton("Music")
        music_btn.clicked.connect(lambda: self.file_tree.setRootIndex(
            self.file_model.index(QDir.homePath() + "/Music")
        ))
        quick_access.addWidget(music_btn)

        documents_btn = QPushButton("Documents")
        documents_btn.clicked.connect(lambda: self.file_tree.setRootIndex(
            self.file_model.index(QDir.homePath() + "/Documents")
        ))
        quick_access.addWidget(documents_btn)

        layout.addLayout(quick_access)

        return group_box

    def create_speaker_controls(self):
        """
        Create speaker control panel.

        Returns:
            QWidget containing speaker controls
        """
        # Create a group box for the speaker controls
        group_box = QGroupBox("Speaker Controls")
        group_box.setFixedHeight(120)  # Set fixed height to avoid taking too much space
        layout = QVBoxLayout(group_box)

        # Add speaker enable checkbox
        speaker_enable_layout = QHBoxLayout()
        self.speaker_enable_check = QCheckBox("Enable Speaker Output")
        self.speaker_enable_check.setChecked(True)  # Default to enabled
        self.speaker_enable_check.stateChanged.connect(self.toggle_speaker_output)
        speaker_enable_layout.addWidget(self.speaker_enable_check)

        # Add output device selector
        speaker_enable_layout.addWidget(QLabel("Output Device:"))
        self.output_device_combo = QComboBox()
        self.update_output_device_list()
        self.output_device_combo.currentIndexChanged.connect(self.change_output_device)
        speaker_enable_layout.addWidget(self.output_device_combo)

        layout.addLayout(speaker_enable_layout)

        # Add volume slider
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(100)  # Default to full volume
        self.volume_slider.valueChanged.connect(self.change_volume)
        volume_layout.addWidget(self.volume_slider)

        # Add volume percentage label
        self.volume_label = QLabel("100%")
        volume_layout.addWidget(self.volume_label)

        layout.addLayout(volume_layout)

        return group_box

    def create_audio_player(self):
        """
        Create an audio player panel.

        Returns:
            QWidget containing the audio player controls
        """
        # Create a group box for the audio player
        group_box = QGroupBox("Audio Player")
        layout = QVBoxLayout(group_box)

        # Add current file label
        self.file_label = QLabel("No file loaded")
        self.file_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.file_label)

        # Add time display
        time_layout = QHBoxLayout()
        self.time_elapsed = QLabel("00:00")
        self.time_total = QLabel("00:00")
        time_layout.addWidget(self.time_elapsed)
        time_layout.addStretch()
        time_layout.addWidget(self.time_total)
        layout.addLayout(time_layout)

        # Add progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        self.progress_slider.setTracking(False)  # Only update when released
        self.progress_slider.sliderReleased.connect(self.on_progress_slider_change)
        layout.addWidget(self.progress_slider)

        # Add playback controls
        controls_layout = QHBoxLayout()

        # Get standard icons from Qt style
        style = self.style()

        # Previous button
        self.prev_button = QPushButton()
        self.prev_button.setIcon(style.standardIcon(QStyle.SP_MediaSkipBackward))
        self.prev_button.clicked.connect(self.on_prev_clicked)
        controls_layout.addWidget(self.prev_button)

        # Play/Pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)

        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setIcon(style.standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        # Next button
        self.next_button = QPushButton()
        self.next_button.setIcon(style.standardIcon(QStyle.SP_MediaSkipForward))
        self.next_button.clicked.connect(self.on_next_clicked)
        controls_layout.addWidget(self.next_button)

        layout.addLayout(controls_layout)

        # Add playlist
        playlist_layout = QVBoxLayout()
        playlist_layout.addWidget(QLabel("Playlist:"))

        self.playlist = QListWidget()
        self.playlist.itemDoubleClicked.connect(self.on_playlist_item_double_clicked)
        playlist_layout.addWidget(self.playlist)

        # Add playlist controls
        playlist_controls = QHBoxLayout()

        add_to_playlist_btn = QPushButton("Add Files")
        add_to_playlist_btn.clicked.connect(self.add_files_to_playlist)
        playlist_controls.addWidget(add_to_playlist_btn)

        clear_playlist_btn = QPushButton("Clear")
        clear_playlist_btn.clicked.connect(self.clear_playlist)
        playlist_controls.addWidget(clear_playlist_btn)

        playlist_layout.addLayout(playlist_controls)
        layout.addLayout(playlist_layout)

        return group_box

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

        # Add speaker controls to toolbar
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Volume:"))

        volume_slider = QSlider(Qt.Horizontal)
        volume_slider.setMinimum(0)
        volume_slider.setMaximum(100)
        volume_slider.setValue(100)
        volume_slider.setFixedWidth(100)
        volume_slider.valueChanged.connect(self.change_volume)
        toolbar.addWidget(volume_slider)
        self.toolbar_volume_slider = volume_slider

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

        open_folder_action = QAction("Open F&older", self)
        open_folder_action.setShortcut("Ctrl+D")
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Playback menu
        playback_menu = menu_bar.addMenu("&Playback")

        play_action = QAction("&Play/Pause", self)
        play_action.setShortcut("Space")
        play_action.triggered.connect(self.toggle_playback)
        playback_menu.addAction(play_action)

        stop_action = QAction("&Stop", self)
        stop_action.setShortcut("Ctrl+S")
        stop_action.triggered.connect(self.stop_playback)
        playback_menu.addAction(stop_action)

        playback_menu.addSeparator()

        prev_action = QAction("P&revious", self)
        prev_action.setShortcut("Ctrl+Left")
        prev_action.triggered.connect(self.on_prev_clicked)
        playback_menu.addAction(prev_action)

        next_action = QAction("&Next", self)
        next_action.setShortcut("Ctrl+Right")
        next_action.triggered.connect(self.on_next_clicked)
        playback_menu.addAction(next_action)

        # Speaker menu
        speaker_menu = menu_bar.addMenu("&Speaker")

        enable_speaker_action = QAction("&Enable Speaker Output", self)
        enable_speaker_action.setCheckable(True)
        enable_speaker_action.setChecked(True)
        enable_speaker_action.triggered.connect(lambda checked: self.toggle_speaker_output(Qt.Checked if checked else Qt.Unchecked))
        speaker_menu.addAction(enable_speaker_action)
        self.enable_speaker_action = enable_speaker_action

        speaker_menu.addSeparator()

        volume_up_action = QAction("Volume &Up", self)
        volume_up_action.setShortcut("Ctrl+Up")
        volume_up_action.triggered.connect(self.volume_up)
        speaker_menu.addAction(volume_up_action)

        volume_down_action = QAction("Volume &Down", self)
        volume_down_action.setShortcut("Ctrl+Down")
        volume_down_action.triggered.connect(self.volume_down)
        speaker_menu.addAction(volume_down_action)

        mute_action = QAction("&Mute", self)
        mute_action.setShortcut("Ctrl+M")
        mute_action.triggered.connect(self.toggle_mute)
        speaker_menu.addAction(mute_action)

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
        layout.setContentsMargins(5, 5, 5, 5)

        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

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

        for device_index, device_name in self.audio_capture.get_input_device_list():
            self.device_combo.addItem(device_name, device_index)

        # Set current device
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == self.audio_capture.input_device_index:
                self.device_combo.setCurrentIndex(i)
                break

    def update_output_device_list(self):
        """
        Update the output device list in the combobox.
        """
        self.output_device_combo.clear()

        for device_index, device_name in self.audio_capture.get_output_device_list():
            self.output_device_combo.addItem(device_name, device_index)

        # Set current device
        for i in range(self.output_device_combo.count()):
            if self.output_device_combo.itemData(i) == self.audio_capture.output_device_index:
                self.output_device_combo.setCurrentIndex(i)
                break

    def change_input_device(self, index):
        """
        Change the audio input device.

        Args:
            index: Index in the combobox
        """
        if index >= 0:
            device_index = self.device_combo.itemData(index)
            self.audio_capture.set_input_device(device_index)
            self.status_bar.showMessage(f"Input device changed to: {self.device_combo.itemText(index)}")

    def change_output_device(self, index):
        """
        Change the audio output device.

        Args:
            index: Index in the combobox
        """
        if index >= 0:
            device_index = self.output_device_combo.itemData(index)
            self.audio_capture.set_output_device(device_index)
            self.status_bar.showMessage(f"Output device changed to: {self.output_device_combo.itemText(index)}")

    def toggle_speaker_output(self, state):
        """
        Toggle speaker output on/off.

        Args:
            state: Qt.Checked or Qt.Unchecked
        """
        enable = (state == Qt.Checked)
        self.audio_capture.set_speaker_output(enable)

        # Update menu action
        self.enable_speaker_action.setChecked(enable)

        # Update checkbox if called from menu
        if self.speaker_enable_check.isChecked() != enable:
            self.speaker_enable_check.setChecked(enable)

        # Update status
        self.status_bar.showMessage(f"Speaker output {'enabled' if enable else 'disabled'}")

    def change_volume(self, value):
        """
        Change the output volume.

        Args:
            value: Volume value from 0 to 100
        """
        # Convert to 0.0-1.0 range
        volume = value / 100.0
        self.audio_capture.set_output_volume(volume)

        # Update volume label
        self.volume_label.setText(f"{value}%")

        # Update toolbar slider if not the source
        if hasattr(self, 'toolbar_volume_slider') and self.toolbar_volume_slider.value() != value:
            self.toolbar_volume_slider.setValue(value)

        # Update slider if not the source
        if hasattr(self, 'volume_slider') and self.volume_slider.value() != value:
            self.volume_slider.setValue(value)

        self.status_bar.showMessage(f"Volume set to {value}%")

    def volume_up(self):
        """
        Increase volume by 10%.
        """
        current = self.volume_slider.value()
        new_value = min(100, current + 10)
        self.volume_slider.setValue(new_value)

    def volume_down(self):
        """
        Decrease volume by 10%.
        """
        current = self.volume_slider.value()
        new_value = max(0, current - 10)
        self.volume_slider.setValue(new_value)

    def toggle_mute(self):
        """
        Toggle mute state.
        """
        if self.volume_slider.value() > 0:
            # Store current volume and set to 0
            self._previous_volume = self.volume_slider.value()
            self.volume_slider.setValue(0)
        else:
            # Restore previous volume if available
            previous = getattr(self, '_previous_volume', 100)
            self.volume_slider.setValue(previous)

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
        # Stop playback if running to prevent feedback
        if self.is_playing:
            self.stop_playback()

        # Attempt to start recording
        if self.audio_capture.start_recording():
            self.record_button.setText("Stop Recording")
            self.action_record.setChecked(True)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Recording...")
            self.status_bar.showMessage("Recording audio...")

            # Disable playback controls while recording to prevent feedback
            self.play_button.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.progress_slider.setEnabled(False)
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

        # Re-enable playback controls
        self.play_button.setEnabled(self.playlist.count() > 0)
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.progress_slider.setEnabled(True)

    def stop_audio(self):
        """
        Stop all audio (recording or playback).
        """
        if self.audio_capture.is_recording:
            self.stop_recording()

        self.stop_playback()
        self.status_bar.showMessage("Stopped")

    def open_audio_file(self):
        """
        Open an audio file for analysis.
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Audio File(s)",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )

        if file_paths:
            # Add files to playlist
            for file_path in file_paths:
                self.add_to_playlist(file_path)

            # If not currently playing, start the first file
            if not self.is_playing and self.playlist.count() > 0:
                self.play_file(file_paths[0])

    def open_folder(self):
        """
        Open a folder and add all audio files to the playlist.
        """
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Open Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder_path:
            # Get all audio files in the folder
            audio_files = []
            for file in os.listdir(folder_path):
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    file_path = os.path.join(folder_path, file)
                    audio_files.append(file_path)

            # Add files to playlist
            for file_path in audio_files:
                self.add_to_playlist(file_path)

            # If not currently playing and we found files, start the first one
            if not self.is_playing and audio_files and self.playlist.count() > 0:
                self.play_file(audio_files[0])

            self.status_bar.showMessage(f"Added {len(audio_files)} audio files from {folder_path}")

    def play_file(self, file_path):
        """
        Play an audio file.

        Args:
            file_path: Path to the audio file
        """
        # Stop current recording if active to prevent feedback
        if self.audio_capture.is_recording:
            self.stop_recording()

        # Stop current playback if active
        if self.is_playing:
            self.stop_playback()

        # Clear the waveform and spectrogram
        self.waveform_widget.clear()
        self.spectrogram_widget.clear()

        # Update UI
        self.current_file = file_path
        self.file_label.setText(os.path.basename(file_path))
        self.status_label.setText(f"Playing: {os.path.basename(file_path)}")
        self.status_bar.showMessage(f"Playing: {os.path.basename(file_path)}")

        # Update playlist selection
        for i in range(self.playlist.count()):
            item = self.playlist.item(i)
            if item.data(Qt.UserRole) == file_path:
                self.playlist.setCurrentItem(item)
                break

        # Load and play the file
        success = self.audio_capture.play_audio_file(
            file_path,
            self.update_playback_progress
        )

        if success:
            self.is_playing = True
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)

            # Disable recording controls while playing to prevent feedback
            self.record_button.setEnabled(False)
            self.action_record.setEnabled(False)

            # Get audio duration
            audio_data, sample_rate = self.audio_capture.load_audio_file(file_path)
            if audio_data is not None and sample_rate > 0:
                self.playback_duration = len(audio_data) / sample_rate
                minutes = int(self.playback_duration // 60)
                seconds = int(self.playback_duration % 60)
                self.time_total.setText(f"{minutes:02}:{seconds:02}")
            else:
                self.playback_duration = 0
                self.time_total.setText("00:00")
        else:
            self.is_playing = False
            self.current_file = None
            self.file_label.setText("Error loading file")
            self.status_label.setText("Error loading file")
            QMessageBox.warning(self, "Error", f"Could not open the audio file: {file_path}")


    def update_playback_progress(self, progress):
        """
        Update playback progress.

        Args:
            progress: Progress value between 0.0 and 1.0
        """
        self.playback_progress = progress

        # Update progress slider
        self.progress_slider.setValue(int(progress * 1000))

        # Update time display
        if self.playback_duration > 0:
            current_time = progress * self.playback_duration
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            self.time_elapsed.setText(f"{minutes:02}:{seconds:02}")

        self.status_bar.showMessage(f"Playing: {os.path.basename(self.current_file)} - {int(progress * 100)}%")

        # Check if playback is complete
        if progress >= 0.99:
            self.on_playback_complete()

    def on_playback_complete(self):
        """
        Handle playback completion.
        """
        self.is_playing = False
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        # Play the next file in playlist if available
        current_row = self.playlist.currentRow()
        if current_row < self.playlist.count() - 1:
            next_item = self.playlist.item(current_row + 1)
            next_file = next_item.data(Qt.UserRole)
            self.play_file(next_file)

    def toggle_playback(self):
        """
        Toggle play/pause state.
        """
        if not self.current_file:
            # If no file is playing but playlist has items, play the first item
            if self.playlist.count() > 0:
                first_item = self.playlist.item(0)
                first_file = first_item.data(Qt.UserRole)
                self.play_file(first_file)
            return

        if self.is_playing:
            # Pause playback
            if self.audio_capture.pause_playback():
                self.is_playing = False
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                self.status_bar.showMessage(f"Paused: {os.path.basename(self.current_file)}")
                self.status_label.setText(f"Paused: {os.path.basename(self.current_file)}")
        else:
            # Check if playback is paused
            if self.audio_capture.is_playback_paused():
                # Resume playback
                if self.audio_capture.resume_playback():
                    self.is_playing = True
                    self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                    self.status_bar.showMessage(f"Playing: {os.path.basename(self.current_file)}")
                    self.status_label.setText(f"Playing: {os.path.basename(self.current_file)}")
            else:
                # Start playback from beginning
                self.play_file(self.current_file)

    def stop_playback(self):
        """
        Stop playback.
        """
        # Stop the audio
        self.audio_capture.stop_playback()

        # Reset UI
        self.is_playing = False
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setEnabled(self.playlist.count() > 0)
        self.stop_button.setEnabled(False)
        self.progress_slider.setValue(0)
        self.time_elapsed.setText("00:00")
        self.status_bar.showMessage("Playback stopped")
        self.status_label.setText("Playback stopped")

        # Re-enable recording controls
        self.record_button.setEnabled(True)
        self.action_record.setEnabled(True)

    def on_progress_slider_change(self):
        """
        Handle progress slider position change.
        """
        # Get new position (0.0 to 1.0)
        position = self.progress_slider.value() / 1000.0

        # Seek to the new position
        if self.audio_capture.seek_playback(position):
            logger.info(f"Seek to position: {position:.2f}")

            # Update time display
            if self.playback_duration > 0:
                current_time = position * self.playback_duration
                minutes = int(current_time // 60)
                seconds = int(current_time % 60)
                self.time_elapsed.setText(f"{minutes:02}:{seconds:02}")

                # Update status
                status = "Playing" if self.is_playing else "Paused"
                self.status_bar.showMessage(f"{status}: {os.path.basename(self.current_file)} - {int(position * 100)}%")

    def on_prev_clicked(self):
        """
        Play the previous file in the playlist.
        """
        current_row = self.playlist.currentRow()
        if current_row > 0:
            prev_item = self.playlist.item(current_row - 1)
            prev_file = prev_item.data(Qt.UserRole)
            self.play_file(prev_file)

    def on_next_clicked(self):
        """
        Play the next file in the playlist.
        """
        current_row = self.playlist.currentRow()
        if current_row < self.playlist.count() - 1:
            next_item = self.playlist.item(current_row + 1)
            next_file = next_item.data(Qt.UserRole)
            self.play_file(next_file)

    def add_to_playlist(self, file_path):
        """
        Add a file to the playlist.

        Args:
            file_path: Path to the audio file
        """
        # Check if file already exists in playlist
        for i in range(self.playlist.count()):
            if self.playlist.item(i).data(Qt.UserRole) == file_path:
                return

        # Create new item with file name
        item = QListWidgetItem(os.path.basename(file_path))
        item.setData(Qt.UserRole, file_path)
        self.playlist.addItem(item)

        # Enable play button if it was disabled
        self.play_button.setEnabled(True)

    def add_files_to_playlist(self):
        """
        Open a file dialog to add files to the playlist.
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Audio Files to Playlist",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )

        for file_path in file_paths:
            self.add_to_playlist(file_path)

    def clear_playlist(self):
        """
        Clear the playlist.
        """
        self.playlist.clear()
        self.play_button.setEnabled(False)

        if not self.is_playing:
            self.file_label.setText("No file loaded")

    def on_playlist_item_double_clicked(self, item):
        """
        Handle double-click on playlist item.

        Args:
            item: The clicked QListWidgetItem
        """
        file_path = item.data(Qt.UserRole)
        self.play_file(file_path)

    def on_file_double_clicked(self, index):
        """
        Handle double-click on file browser item.

        Args:
            index: QModelIndex of the clicked item
        """
        # Get the file path
        file_path = self.file_model.filePath(index)

        # Check if it's a directory
        if os.path.isdir(file_path):
            return

        # Check if it's an audio file
        if file_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            # Add to playlist
            self.add_to_playlist(file_path)

            # Play the file if not already playing
            if not self.is_playing:
                self.play_file(file_path)

    def update_ui(self):
        """
        Update UI components periodically.
        This is called by the timer.
        """
        # Update progress display if playing
        if self.is_playing:
            # Update visualization widgets if needed
            # (Most visualization is already handled by the processor callbacks)
            pass

        # Check if buffer data is available for the file browser
        if hasattr(self, 'file_tree') and not self.file_tree.isVisible() and self.file_tree.parent().isVisible():
            self.file_tree.setVisible(True)

    def show_about_dialog(self):
        """
        Show the about dialog.
        """
        QMessageBox.about(
            self,
            "About Sound Analyzer",
            "Sound Analyzer\n"
            "A real-time sound analysis application with plugin support.\n"
            "Version: 0.2.0\n"
            "Created with Python and PyQt5.\n\n"
            "Features:\n"
            "- Real-time audio visualization\n"
            "- Audio file playback and analysis\n"
            "- Plugin system for audio processing\n"
            "- Bird species identification with BirdNET\n"
            "- Speaker output for audio monitoring"
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

        # Stop playback if active
        if self.is_playing:
            self.stop_playback()

        # Clean up audio resources
        self.audio_capture.cleanup()

        # Save settings
        self.save_settings()

        # Accept the close event
        event.accept()

    def save_settings(self):
        """
        Save application settings.
        """
        settings = QSettings("SoundAnalyzer", "App")

        # Save window geometry
        settings.setValue("geometry", self.saveGeometry())

        # Save last directory
        if hasattr(self, 'file_tree') and self.file_model:
            current_path = self.file_model.filePath(self.file_tree.rootIndex())
            settings.setValue("lastDirectory", current_path)

        # Save playlist
        playlist_files = []
        for i in range(self.playlist.count()):
            item = self.playlist.item(i)
            playlist_files.append(item.data(Qt.UserRole))
        settings.setValue("playlist", playlist_files)

        # Save speaker settings
        settings.setValue("speakerEnabled", self.speaker_enable_check.isChecked())
        settings.setValue("volume", self.volume_slider.value())
        if hasattr(self, 'output_device_combo') and self.output_device_combo.currentData() is not None:
            settings.setValue("outputDevice", self.output_device_combo.currentData())

        logger.info("Settings saved")

    def load_settings(self):
        """
        Load application settings.
        """
        settings = QSettings("SoundAnalyzer", "App")

        # Restore window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Restore last directory
        last_dir = settings.value("lastDirectory")
        if last_dir and os.path.exists(last_dir) and hasattr(self, 'file_tree'):
            self.file_tree.setRootIndex(self.file_model.index(last_dir))

        # Restore playlist
        playlist_files = settings.value("playlist", [])
        if playlist_files:
            for file_path in playlist_files:
                if os.path.exists(file_path):
                    self.add_to_playlist(file_path)

        # Load speaker settings
        speaker_enabled = settings.value("speakerEnabled", True, type=bool)
        self.speaker_enable_check.setChecked(speaker_enabled)
        self.toggle_speaker_output(Qt.Checked if speaker_enabled else Qt.Unchecked)

        volume = settings.value("volume", 100, type=int)
        self.volume_slider.setValue(volume)

        output_device = settings.value("outputDevice")
        if output_device is not None:
            # Find the index of the device in the combo box
            for i in range(self.output_device_combo.count()):
                if self.output_device_combo.itemData(i) == output_device:
                    self.output_device_combo.setCurrentIndex(i)
                    break

        logger.info("Settings loaded")

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

