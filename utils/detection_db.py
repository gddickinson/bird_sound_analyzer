"""
Database module for Sound Analyzer detections.
Handles storage and retrieval of bird species detections and audio clips.
"""
import os
import sqlite3
import logging
import time
from datetime import datetime
import wave
import shutil
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

class DetectionDatabase:
    """
    Database for bird species detections.
    Stores detection metadata in SQLite and audio clips on disk.
    """
    
    def __init__(self, db_path: str = None, audio_dir: str = None):
        """
        Initialize the detection database.
        
        Args:
            db_path: Path to SQLite database file, or None to use default
            audio_dir: Directory for storing audio clips, or None to use default
        """
        # Set up default paths if not provided
        if db_path is None:
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(app_dir, 'data', 'detections.db')
        
        if audio_dir is None:
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            audio_dir = os.path.join(app_dir, 'data', 'audio_clips')
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        self.db_path = db_path
        self.audio_dir = audio_dir
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Detection database initialized at {db_path}")
        logger.info(f"Audio clips will be stored in {audio_dir}")
    
    def _init_database(self):
        """
        Initialize the database schema if it doesn't exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            latitude REAL,
            longitude REAL,
            notes TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bird_detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scientific_name TEXT,
            common_name TEXT,
            confidence REAL,
            audio_file TEXT,
            latitude REAL,
            longitude REAL,
            FOREIGN KEY (session_id) REFERENCES detection_sessions (session_id)
        )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_session ON bird_detections (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_species ON bird_detections (scientific_name)')
        
        conn.commit()
        conn.close()
        
        logger.debug("Database schema initialized")
    
    def start_session(self, latitude: float = None, longitude: float = None, notes: str = None) -> int:
        """
        Start a new detection session.
        
        Args:
            latitude: Optional latitude for the session
            longitude: Optional longitude for the session
            notes: Optional notes about the session
        
        Returns:
            The session ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO detection_sessions (start_time, latitude, longitude, notes) VALUES (?, ?, ?, ?)',
            (datetime.now(), latitude, longitude, notes)
        )
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Started new detection session {session_id}")
        return session_id
    
    def end_session(self, session_id: int, notes: str = None):
        """
        End a detection session, optionally adding notes.
        
        Args:
            session_id: The session ID to end
            notes: Optional notes to add to the session
        """
        if notes:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing notes
            cursor.execute('SELECT notes FROM detection_sessions WHERE session_id = ?', (session_id,))
            result = cursor.fetchone()
            
            if result:
                existing_notes = result[0] or ""
                # Add new notes with timestamp
                updated_notes = f"{existing_notes}\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {notes}"
                
                cursor.execute(
                    'UPDATE detection_sessions SET notes = ? WHERE session_id = ?',
                    (updated_notes, session_id)
                )
                
                conn.commit()
            
            conn.close()
        
        logger.info(f"Ended detection session {session_id}")
    
    def add_detection(self, 
                     session_id: int,
                     scientific_name: str,
                     common_name: str,
                     confidence: float,
                     audio_data: np.ndarray,
                     sample_rate: int,
                     latitude: float = None,
                     longitude: float = None) -> int:
        """
        Add a bird detection to the database and save the audio clip.
        
        Args:
            session_id: The session ID
            scientific_name: Scientific name of the bird species
            common_name: Common name of the bird species
            confidence: Detection confidence score (0.0 to 1.0)
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate in Hz
            latitude: Optional latitude for the detection
            longitude: Optional longitude for the detection
            
        Returns:
            The detection ID
        """
        timestamp = datetime.now()
        
        # Create a unique filename for the audio clip
        formatted_name = scientific_name.replace(' ', '_').lower()
        filename = f"{formatted_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{int(confidence*100)}.wav"
        filepath = os.path.join(self.audio_dir, filename)
        
        # Save the audio clip
        self._save_audio_clip(filepath, audio_data, sample_rate)
        
        # Add to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''INSERT INTO bird_detections 
               (session_id, timestamp, scientific_name, common_name, confidence, 
                audio_file, latitude, longitude)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (session_id, timestamp, scientific_name, common_name, confidence, 
             filename, latitude, longitude)
        )
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Added detection ID {detection_id}: {scientific_name} ({common_name}) with confidence {confidence:.2f}")
        return detection_id
    
    def _save_audio_clip(self, filepath: str, audio_data: np.ndarray, sample_rate: int):
        """
        Save audio data to a WAV file.
        
        Args:
            filepath: Path to save the WAV file
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate in Hz
        """
        try:
            # Convert float32 to int16
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            elif audio_data.dtype == np.int16:
                audio_int16 = audio_data
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            logger.debug(f"Saved audio clip to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving audio clip to {filepath}: {e}")
    
    def get_detections(self, 
                      session_id: Optional[int] = None,
                      species: Optional[str] = None,
                      min_confidence: float = 0.0,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get bird detections matching the specified criteria.
        
        Args:
            session_id: Optional session ID to filter by
            species: Optional scientific or common name to filter by
            min_confidence: Minimum confidence threshold
            start_date: Optional start date in format 'YYYY-MM-DD'
            end_date: Optional end date in format 'YYYY-MM-DD'
            limit: Maximum number of results to return
            
        Returns:
            List of detection dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        query = '''
        SELECT * FROM bird_detections
        WHERE confidence >= ?
        '''
        
        params = [min_confidence]
        
        if session_id is not None:
            query += ' AND session_id = ?'
            params.append(session_id)
        
        if species is not None:
            query += ' AND (scientific_name LIKE ? OR common_name LIKE ?)'
            params.extend([f'%{species}%', f'%{species}%'])
        
        if start_date is not None:
            query += ' AND timestamp >= ?'
            params.append(f'{start_date} 00:00:00')
        
        if end_date is not None:
            query += ' AND timestamp <= ?'
            params.append(f'{end_date} 23:59:59')
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        # Convert result to list of dictionaries
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        logger.debug(f"Retrieved {len(results)} detections matching criteria")
        return results
    
    def get_species_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all detected species and their counts.
        
        Returns:
            List of species summary dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
        SELECT scientific_name, common_name, 
               COUNT(*) as detection_count,
               AVG(confidence) as avg_confidence,
               MAX(confidence) as max_confidence,
               MIN(timestamp) as first_detection,
               MAX(timestamp) as last_detection
        FROM bird_detections
        GROUP BY scientific_name
        ORDER BY detection_count DESC
        '''
        
        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        logger.debug(f"Retrieved summary for {len(results)} species")
        return results
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific detection by ID.
        
        Args:
            detection_id: The detection ID to retrieve
            
        Returns:
            Detection dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bird_detections WHERE detection_id = ?', (detection_id,))
        result = cursor.fetchone()
        
        if result:
            result = dict(result)
        
        conn.close()
        return result
    
    def get_audio_path(self, detection_id: int) -> Optional[str]:
        """
        Get the full path to an audio clip for a detection.
        
        Args:
            detection_id: The detection ID
            
        Returns:
            Full path to the audio file or None if not found
        """
        detection = self.get_detection_by_id(detection_id)
        
        if detection and detection['audio_file']:
            return os.path.join(self.audio_dir, detection['audio_file'])
        
        return None
    
    def get_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get a list of detection sessions.
        
        Args:
            limit: Maximum number of sessions to retrieve
            
        Returns:
            List of session dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
        SELECT s.*, COUNT(d.detection_id) as detection_count
        FROM detection_sessions s
        LEFT JOIN bird_detections d ON s.session_id = d.session_id
        GROUP BY s.session_id
        ORDER BY s.start_time DESC
        LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        logger.debug(f"Retrieved {len(results)} detection sessions")
        return results
    
    def delete_detection(self, detection_id: int) -> bool:
        """
        Delete a detection and its associated audio file.
        
        Args:
            detection_id: The detection ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Get the audio file path before deleting from database
        audio_path = self.get_audio_path(detection_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM bird_detections WHERE detection_id = ?', (detection_id,))
            conn.commit()
            
            # Delete the audio file if it exists
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"Deleted audio file: {audio_path}")
            
            logger.info(f"Deleted detection ID {detection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting detection ID {detection_id}: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def export_detections_csv(self, filepath: str, session_id: Optional[int] = None) -> int:
        """
        Export detections to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
            session_id: Optional session ID to filter by
            
        Returns:
            Number of exported records
        """
        import csv
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
        SELECT d.detection_id, d.session_id, d.timestamp, 
               d.scientific_name, d.common_name, d.confidence,
               d.audio_file, d.latitude, d.longitude,
               s.start_time as session_start
        FROM bird_detections d
        JOIN detection_sessions s ON d.session_id = s.session_id
        '''
        
        params = []
        if session_id is not None:
            query += ' WHERE d.session_id = ?'
            params.append(session_id)
        
        query += ' ORDER BY d.timestamp DESC'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            conn.close()
            logger.warning(f"No detections found to export")
            return 0
        
        # Write to CSV
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([key for key in dict(results[0]).keys()])
                
                # Write data
                for row in results:
                    writer.writerow([value for value in dict(row).values()])
            
            logger.info(f"Exported {len(results)} detections to {filepath}")
            return len(results)
            
        except Exception as e:
            logger.error(f"Error exporting detections to CSV: {e}")
            return 0
            
        finally:
            conn.close()
    
    def vacuum(self):
        """
        Optimize the database by running VACUUM.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("VACUUM")
        conn.close()
        logger.info("Database optimized with VACUUM")


# Singleton instance for easy access
_instance = None

def get_db_instance(db_path=None, audio_dir=None):
    """
    Get or create the singleton database instance.
    
    Args:
        db_path: Optional path to database file
        audio_dir: Optional path to audio directory
        
    Returns:
        DetectionDatabase instance
    """
    global _instance
    if _instance is None:
        _instance = DetectionDatabase(db_path, audio_dir)
    return _instance
