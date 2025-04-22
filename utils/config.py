"""
Configuration utilities for Sound Analyzer.
"""
import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file, or None to use default
        
    Returns:
        Dict containing configuration
    """
    # Default configuration
    config = {
        'audio': {
            'sample_rate': 44100,
            'chunk_size': 1024,
            'channels': 1,
            'n_fft': 2048,
            'hop_length': 512,
            'window': 'hann',
            'min_freq': 0,
            'max_freq': 22050,  # Nyquist frequency
            'history_seconds': 5,
        },
        'plugins': {
            'enabled': ['birdnet'],
            'birdnet': {
                'confidence_threshold': 0.5,
                'analysis_interval': 3.0,
                'detection_window': 3.0,
                'overlap': 0.0,
                'max_results': 10,
                'use_location': False,
                'latitude': None,
                'longitude': None,
            }
        },
        'ui': {
            'theme': 'light',
            'window_size': [1200, 800],
            'auto_start': False,
        }
    }
    
    # If config path is not specified, look for config.yaml in the current directory
    if config_path is None:
        config_path = 'config.yaml'
        
        # If not found, look in the app directory
        if not os.path.exists(config_path):
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(app_dir, 'config.yaml')
    
    # Load configuration from file if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Update default config with values from file
            if file_config:
                _update_config_recursive(config, file_config)
                
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    else:
        logger.warning(f"Configuration file not found at {config_path}, using defaults")
        
        # Save default config for reference
        try:
            save_config(config, config_path)
        except Exception as e:
            logger.error(f"Error saving default configuration to {config_path}: {e}")
    
    return config

def save_config(config, config_path='config.yaml'):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dict to save
        config_path: Path to save the config file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")

def _update_config_recursive(base_config, new_config):
    """
    Recursively update a nested configuration dictionary.
    
    Args:
        base_config: Base configuration to update
        new_config: New configuration to apply
    """
    for key, value in new_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            _update_config_recursive(base_config[key], value)
        else:
            base_config[key] = value
