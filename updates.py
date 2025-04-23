"""
Integration code for the BirdNET Plugin to work with the detection database.
Add this to your birdnet_plugin.py file.
"""

# Import the database module
from utils.detection_db import get_db_instance

# Add these functions to the BirdNETPlugin class

def _init_database(self):
    """
    Initialize the detection database connection.
    """
    # Get database instance
    self.db = get_db_instance()

    # Start a new session when the plugin initializes
    if not hasattr(self, 'session_id') or self.session_id is None:
        notes = f"Session started with {BIRDNET_TYPE}. "
        notes += f"Confidence threshold: {self.confidence_threshold}, "
        notes += f"Analysis interval: {self.analysis_interval}s, "
        notes += f"Detection window: {self.detection_window}s"

        # Add location if enabled
        if self.use_location and self.latitude is not None and self.longitude is not None:
            self.session_id = self.db.start_session(
                latitude=self.latitude,
                longitude=self.longitude,
                notes=notes
            )
        else:
            self.session_id = self.db.start_session(notes=notes)

        logger.info(f"Started detection database session with ID {self.session_id}")

def _save_detection_to_db(self, scientific_name, common_name, confidence, audio_data):
    """
    Save a detection to the database.

    Args:
        scientific_name: Scientific name of the bird species
        common_name: Common name of the bird species
        confidence: Detection confidence score (0.0 to 1.0)
        audio_data: Audio data as numpy array
    """
    # Make sure we have a session
    if not hasattr(self, 'session_id') or self.session_id is None:
        self._init_database()

    try:
        # Get latitude and longitude if available
        lat = self.latitude if self.use_location else None
        lon = self.longitude if self.use_location else None

        # Save to database
        detection_id = self.db.add_detection(
            session_id=self.session_id,
            scientific_name=scientific_name,
            common_name=common_name,
            confidence=confidence,
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            latitude=lat,
            longitude=lon
        )

        logger.info(f"Saved detection to database with ID {detection_id}")

        # Update status message if we have a UI
        if self.status_label:
            self.status_label.setText(f"Status: Saved detection ID {detection_id} to database")

    except Exception as e:
        logger.error(f"Error saving detection to database: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Show error in UI
        if self.status_label:
            self.status_label.setText(f"Status: Error saving to database")

def shutdown(self):
    """
    Perform cleanup when shutting down the plugin.

    Note: This replaces the original shutdown method to add database session closing.
    """
    # End the database session if we have one
    if hasattr(self, 'session_id') and self.session_id is not None and hasattr(self, 'db'):
        try:
            self.db.end_session(self.session_id, notes="Session ended normally")
            logger.info(f"Ended database session {self.session_id}")
        except Exception as e:
            logger.error(f"Error ending database session: {e}")

    # First stop the auto-analysis timer
    if self.auto_analysis_timer and self.auto_analysis_timer.isActive():
        logger.info("Stopping auto-analysis timer")
        self.auto_analysis_timer.stop()

    # Stop any ongoing analysis
    if self.analysis_thread and self.analysis_thread.isRunning():
        logger.info("Stopping analysis thread")
        self.analysis_thread.quit()
        self.analysis_thread.wait()

    # Stop other timers
    if hasattr(self, 'buffer_update_timer') and self.buffer_update_timer and self.buffer_update_timer.isActive():
        self.buffer_update_timer.stop()
    if hasattr(self, 'debug_timer') and self.debug_timer and self.debug_timer.isActive():
        self.debug_timer.stop()

    # Remove log observer
    plugin_log_handler.remove_observer(self._on_log_update)

    logger.info("BirdNET plugin shutdown")
    super().shutdown()

def handle_analysis_results(self, results):
    """
    Handle the results from BirdNET analysis.

    Note: This is a replacement for the original handle_analysis_results
    method to add database storage.

    Args:
        results: List of (scientific_name, common_name, confidence, time) tuples
    """
    logger.info(f"Received analysis results: {len(results)} detections")

    # Make sure database is initialized
    if not hasattr(self, 'db') or self.db is None:
        self._init_database()

    # Update results list with new species
    for scientific, common, confidence, time_start in results:
        # Save to database first - use a copy of the buffer for the specific detection
        # This gives us the audio segment that triggered this detection
        buffer_copy = self.buffer.copy()
        self._save_detection_to_db(scientific, common, confidence, buffer_copy)

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
            self.status_label.setText(f"Status: Detected and saved {len(results)} species")
        else:
            self.status_label.setText("Status: No birds detected")

def __init__(self, audio_processor, config):
    """
    Initialize the BirdNET plugin.

    Note: This is an extension to the original __init__ method to add database initialization.
    Add these lines at the end of your existing __init__ method.

    Args:
        audio_processor: The audio processor instance
        config: Plugin configuration
    """
    # Add after all other initialization

    # Initialize detection database
    self.db = None
    self.session_id = None

    # Add UI elements for database controls - this will be done in initialize_ui

def initialize_ui(self, main_window):
    """
    Note: This is an extension to the original initialize_ui method.
    Add the following code at the end of your initialize_ui method,
    just before starting the buffer_update_timer.
    """
    # Add after all other UI initialization, but before starting timers

    # Add database controls section
    db_group = QGroupBox("Detection Database")
    db_layout = QVBoxLayout(db_group)

    # Database status
    db_status_label = QLabel("Database: Not initialized")
    db_layout.addWidget(db_status_label)
    self.db_status_label = db_status_label

    # Add controls
    db_buttons_layout = QHBoxLayout()

    # Start new session button
    new_session_btn = QPushButton("Start New Session")
    new_session_btn.clicked.connect(self._start_new_db_session)
    db_buttons_layout.addWidget(new_session_btn)

    # View detections button
    view_detections_btn = QPushButton("View Detections")
    view_detections_btn.clicked.connect(self._show_detections)
    db_buttons_layout.addWidget(view_detections_btn)

    # Export detections button
    export_btn = QPushButton("Export to CSV")
    export_btn.clicked.connect(self._export_detections)
    db_buttons_layout.addWidget(export_btn)

    db_layout.addLayout(db_buttons_layout)

    # Add to main layout - find where to add it
    if main_content is not None and isinstance(main_content, QWidget):
        # Try to add it to the main_layout
        for child in main_content.children():
            if isinstance(child, QVBoxLayout):
                child.addWidget(db_group)
                break
    else:
        # Fallback to adding it to the main splitter
        for i in range(main_splitter.count()):
            widget = main_splitter.widget(i)
            if isinstance(widget, QWidget):
                # Find a suitable layout to add to
                for child in widget.children():
                    if isinstance(child, QVBoxLayout):
                        child.addWidget(db_group)
                        break

    # Initialize database and update status
    try:
        self._init_database()
        if self.db and self.session_id:
            self.db_status_label.setText(f"Database: Active (Session ID: {self.session_id})")
            self.db_status_label.setStyleSheet("color: green;")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        self.db_status_label.setText(f"Database: Error initializing")
        self.db_status_label.setStyleSheet("color: red;")

def _start_new_db_session(self):
    """
    Start a new database session.
    """
    # End the current session if any
    if hasattr(self, 'session_id') and self.session_id is not None and hasattr(self, 'db') and self.db is not None:
        try:
            self.db.end_session(self.session_id, notes="Session ended by user")
            logger.info(f"Ended database session {self.session_id}")
        except Exception as e:
            logger.error(f"Error ending database session: {e}")

    # Start a new session
    try:
        notes = f"Session started with {BIRDNET_TYPE}. "
        notes += f"Confidence threshold: {self.confidence_threshold}, "
        notes += f"Analysis interval: {self.analysis_interval}s, "
        notes += f"Detection window: {self.detection_window}s"

        # Add location if enabled
        if self.use_location and self.latitude is not None and self.longitude is not None:
            self.session_id = self.db.start_session(
                latitude=self.latitude,
                longitude=self.longitude,
                notes=notes
            )
        else:
            self.session_id = self.db.start_session(notes=notes)

        logger.info(f"Started detection database session with ID {self.session_id}")

        # Update UI
        if hasattr(self, 'db_status_label') and self.db_status_label is not None:
            self.db_status_label.setText(f"Database: Active (Session ID: {self.session_id})")
            self.db_status_label.setStyleSheet("color: green;")

        if self.status_label:
            self.status_label.setText(f"Status: Started new database session {self.session_id}")

    except Exception as e:
        logger.error(f"Error starting new database session: {e}")

        # Update UI
        if hasattr(self, 'db_status_label') and self.db_status_label is not None:
            self.db_status_label.setText(f"Database: Error starting session")
            self.db_status_label.setStyleSheet("color: red;")

def _show_detections(self):
    """
    Show a dialog with all detections in the current session.
    """
    if not hasattr(self, 'db') or self.db is None:
        logger.error("Cannot show detections - database not initialized")
        if self.status_label:
            self.status_label.setText("Status: Database not initialized")
        return

    try:
        # Create a dialog to show detections
        dialog = QDialog(None)
        dialog.setWindowTitle("Bird Detections")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Add session selector
        session_layout = QHBoxLayout()
        session_layout.addWidget(QLabel("Session:"))

        session_combo = QComboBox()
        session_combo.addItem("All Sessions", None)

        # Get all sessions
        sessions = self.db.get_sessions()
        for session in sessions:
            session_id = session['session_id']
            start_time = session['start_time']
            detection_count = session['detection_count']
            label = f"Session {session_id} - {start_time} ({detection_count} detections)"
            session_combo.addItem(label, session_id)

            # Select current session
            if hasattr(self, 'session_id') and self.session_id == session_id:
                session_combo.setCurrentIndex(session_combo.count() - 1)

        session_layout.addWidget(session_combo)

        # Add confidence filter
        session_layout.addWidget(QLabel("Min Confidence:"))
        confidence_spin = QDoubleSpinBox()
        confidence_spin.setMinimum(0.0)
        confidence_spin.setMaximum(1.0)
        confidence_spin.setSingleStep(0.05)
        confidence_spin.setValue(0.0)
        confidence_spin.setDecimals(2)
        session_layout.addWidget(confidence_spin)

        # Add filter button
        filter_btn = QPushButton("Filter")
        session_layout.addWidget(filter_btn)

        layout.addLayout(session_layout)

        # Add table of detections
        detections_table = QTableWidget()
        detections_table.setColumnCount(6)
        detections_table.setHorizontalHeaderLabels([
            "ID", "Time", "Scientific Name", "Common Name", "Confidence", "Audio"
        ])

        # Set column widths
        header = detections_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

        layout.addWidget(detections_table)

        # Add play controls
        play_layout = QHBoxLayout()
        play_btn = QPushButton("Play Selected")
        play_layout.addWidget(play_btn)

        delete_btn = QPushButton("Delete Selected")
        play_layout.addWidget(delete_btn)

        export_selected_btn = QPushButton("Export Selected")
        play_layout.addWidget(export_selected_btn)

        layout.addLayout(play_layout)

        # Function to update the table
        def update_table():
            # Get selected session ID
            session_id = session_combo.currentData()

            # Get min confidence
            min_confidence = confidence_spin.value()

            # Get detections
            detections = self.db.get_detections(
                session_id=session_id,
                min_confidence=min_confidence
            )

            # Update table
            detections_table.setRowCount(len(detections))

            for row, detection in enumerate(detections):
                # ID
                id_item = QTableWidgetItem(str(detection['detection_id']))
                detections_table.setItem(row, 0, id_item)

                # Time
                time_str = detection['timestamp'].split(' ')[1] if ' ' in detection['timestamp'] else detection['timestamp']
                time_item = QTableWidgetItem(time_str)
                detections_table.setItem(row, 1, time_item)

                # Scientific Name
                scientific_item = QTableWidgetItem(detection['scientific_name'])
                detections_table.setItem(row, 2, scientific_item)

                # Common Name
                common_item = QTableWidgetItem(detection['common_name'])
                detections_table.setItem(row, 3, common_item)

                # Confidence
                conf_str = f"{detection['confidence']:.2f}"
                conf_item = QTableWidgetItem(conf_str)
                detections_table.setItem(row, 4, conf_item)

                # Audio button
                audio_btn = QPushButton("Play")
                audio_btn.setProperty("detection_id", detection['detection_id'])
                audio_btn.clicked.connect(lambda checked, d_id=detection['detection_id']: play_audio(d_id))
                detections_table.setCellWidget(row, 5, audio_btn)

                # Color code by confidence
                confidence = detection['confidence']
                color = self.get_confidence_color(confidence)
                for col in range(5):  # Don't color the button
                    detections_table.item(row, col).setBackground(color)

        # Function to play audio
        def play_audio(detection_id):
            try:
                audio_path = self.db.get_audio_path(detection_id)
                if audio_path and os.path.exists(audio_path):
                    # Use system default audio player
                    import platform
                    import subprocess

                    if platform.system() == 'Windows':
                        os.startfile(audio_path)
                    elif platform.system() == 'Darwin':  # macOS
                        subprocess.call(['open', audio_path])
                    elif platform.system() == 'Linux':
                        subprocess.call(['xdg-open', audio_path])

                    logger.info(f"Playing audio clip: {audio_path}")
                else:
                    logger.warning(f"Audio file not found for detection ID {detection_id}")
                    QMessageBox.warning(dialog, "Audio Not Found", f"Audio file for detection ID {detection_id} not found")
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
                QMessageBox.critical(dialog, "Error", f"Error playing audio: {e}")

        # Function to delete selected detections
        def delete_selected():
            selected_rows = set()
            for index in detections_table.selectedIndexes():
                selected_rows.add(index.row())

            if not selected_rows:
                QMessageBox.information(dialog, "No Selection", "Please select one or more detections to delete")
                return

            # Confirm deletion
            confirm = QMessageBox.question(
                dialog,
                "Confirm Deletion",
                f"Are you sure you want to delete {len(selected_rows)} detection(s)?",
                QMessageBox.Yes | QMessageBox.No
            )

            if confirm != QMessageBox.Yes:
                return

            # Delete detections
            deleted_count = 0
            for row in sorted(selected_rows, reverse=True):  # Delete in reverse order
                detection_id = int(detections_table.item(row, 0).text())
                if self.db.delete_detection(detection_id):
                    deleted_count += 1

            # Update table
            update_table()

            # Show result
            QMessageBox.information(dialog, "Deletion Complete", f"Deleted {deleted_count} detection(s)")

        # Function to export selected detections
        def export_selected():
            selected_rows = set()
            for index in detections_table.selectedIndexes():
                selected_rows.add(index.row())

            if not selected_rows:
                QMessageBox.information(dialog, "No Selection", "Please select one or more detections to export")
                return

            # Get detection IDs
            detection_ids = []
            for row in selected_rows:
                detection_id = int(detections_table.item(row, 0).text())
                detection_ids.append(detection_id)

            # Export to CSV
            filepath, _ = QFileDialog.getSaveFileName(
                dialog,
                "Save Detections to CSV",
                "",
                "CSV files (*.csv);;All files (*)"
            )

            if not filepath:
                return

            # Ensure .csv extension
            if not filepath.lower().endswith('.csv'):
                filepath += '.csv'

            try:
                # Create temporary table with selected detections
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()

                # Create temp view
                cursor.execute('''
                CREATE TEMP VIEW selected_detections AS
                SELECT * FROM bird_detections
                WHERE detection_id IN ({})
                '''.format(','.join(['?'] * len(detection_ids))), detection_ids)

                # Export to CSV
                import csv

                cursor.execute('''
                SELECT d.detection_id, d.session_id, d.timestamp,
                       d.scientific_name, d.common_name, d.confidence,
                       d.audio_file, d.latitude, d.longitude,
                       s.start_time as session_start
                FROM selected_detections d
                JOIN detection_sessions s ON d.session_id = s.session_id
                ORDER BY d.timestamp DESC
                ''')

                # Get column names from cursor description
                columns = [desc[0] for desc in cursor.description]

                # Fetch all rows
                rows = cursor.fetchall()

                # Write to CSV
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    writer.writerows(rows)

                # Close connection
                conn.close()

                # Show result
                QMessageBox.information(
                    dialog,
                    "Export Complete",
                    f"Exported {len(rows)} detection(s) to {filepath}"
                )

            except Exception as e:
                logger.error(f"Error exporting detections: {e}")
                QMessageBox.critical(dialog, "Error", f"Error exporting detections: {e}")

        # Connect signals
        filter_btn.clicked.connect(update_table)
        delete_btn.clicked.connect(delete_selected)
        export_selected_btn.clicked.connect(export_selected)

        # Initial update
        update_table()

        # Show dialog
        dialog.exec_()

    except Exception as e:
        logger.error(f"Error showing detections: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def _export_detections(self):
    """
    Export all detections in the current session to a CSV file.
    """
    if not hasattr(self, 'db') or self.db is None:
        logger.error("Cannot export detections - database not initialized")
        if self.status_label:
            self.status_label.setText("Status: Database not initialized")
        return

    if not hasattr(self, 'session_id') or self.session_id is None:
        logger.error("Cannot export detections - no active session")
        if self.status_label:
            self.status_label.setText("Status: No active session")
        return

    try:
        # Get file path from dialog
        filepath, _ = QFileDialog.getSaveFileName(
            None,
            "Save Detections to CSV",
            "",
            "CSV files (*.csv);;All files (*)"
        )

        if not filepath:
            return

        # Ensure .csv extension
        if not filepath.lower().endswith('.csv'):
            filepath += '.csv'

        # Export to CSV
        count = self.db.export_detections_csv(filepath, self.session_id)

        if count > 0:
            logger.info(f"Exported {count} detections to {filepath}")
            if self.status_label:
                self.status_label.setText(f"Status: Exported {count} detections to CSV")

            # Show success message
            QMessageBox.information(
                None,
                "Export Complete",
                f"Exported {count} detections to {filepath}"
            )
        else:
            logger.warning("No detections to export")
            if self.status_label:
                self.status_label.setText("Status: No detections to export")

            # Show warning message
            QMessageBox.warning(
                None,
                "Export Warning",
                "No detections found to export"
            )

    except Exception as e:
        logger.error(f"Error exporting detections: {e}")
        QMessageBox.critical(
            None,
            "Export Error",
            f"Error exporting detections: {e}"
        )

"""
Integration steps:

1. Add the detection_db.py file to your utils directory
2. Import the database module at the top of birdnet_plugin.py:
   from utils.detection_db import get_db_instance
3. Add all the methods above to your BirdNETPlugin class
4. Replace the original handle_analysis_results method with the version above
5. Add the database initialization code to your __init__ method
6. Add the database UI code to your initialize_ui method
7. Replace your original shutdown method with the version above

If done correctly, your BirdNET Plugin will now automatically store detections
in a SQLite database and save audio clips for each detection.
"""
