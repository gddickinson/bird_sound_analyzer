"""
Plugin management system for Sound Analyzer.
Handles loading, configuring, and managing plugins.
"""
import os
import sys
import logging
import importlib
from typing import Dict, List, Optional, Any, Type

logger = logging.getLogger(__name__)

class PluginManager:
    """Manages the loading and execution of plugins."""

    def __init__(self, plugin_config: Dict[str, Any]):
        """
        Initialize the plugin manager.

        Args:
            plugin_config: Configuration data for plugins
        """
        self.plugins = {}
        self.plugin_config = plugin_config
        self.plugin_classes = {}

        # Auto-discover and register plugins
        self._discover_plugins()

    def _discover_plugins(self):
        """
        Discover and register available plugins.
        Scans the plugins directory for plugin modules.
        """
        logger.info("Discovering plugins...")

        # Get the plugins directory
        plugins_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plugins')
        if not os.path.exists(plugins_dir):
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return

        # Add plugins directory to path to enable imports
        if plugins_dir not in sys.path:
            sys.path.append(plugins_dir)

        # Import base plugin class for type checking
        from plugins.base_plugin import BasePlugin

        # Scan for plugin modules
        for item in os.listdir(plugins_dir):
            # Skip private modules and non-directories
            if item.startswith('_') or item == 'base_plugin.py':
                continue

            plugin_path = os.path.join(plugins_dir, item)

            # Check if it's a plugin package (directory with __init__.py)
            if os.path.isdir(plugin_path) and os.path.exists(os.path.join(plugin_path, '__init__.py')):
                try:
                    # Try to import the plugin module
                    plugin_module_name = f"plugins.{item}"
                    plugin_module = importlib.import_module(plugin_module_name)

                    # Look for plugin classes
                    for attr_name in dir(plugin_module):
                        attr = getattr(plugin_module, attr_name)

                        # Check if it's a plugin class (subclass of BasePlugin)
                        if (isinstance(attr, type) and
                            issubclass(attr, BasePlugin) and
                            attr is not BasePlugin):

                            plugin_id = item
                            self.plugin_classes[plugin_id] = attr
                            logger.info(f"Registered plugin: {plugin_id} ({attr.__name__})")

                except (ImportError, AttributeError) as e:
                    logger.error(f"Error loading plugin {item}: {e}")

    def initialize_plugins(self, audio_processor, main_window):
        """
        Initialize all enabled plugins.

        Args:
            audio_processor: The audio processor instance
            main_window: The main application window
        """
        enabled_plugins = self.plugin_config.get('enabled', [])

        for plugin_id, plugin_class in self.plugin_classes.items():
            # Check if plugin is enabled
            if plugin_id in enabled_plugins or 'all' in enabled_plugins:
                try:
                    # Get plugin-specific config
                    plugin_specific_config = self.plugin_config.get(plugin_id, {})

                    # Initialize the plugin
                    plugin_instance = plugin_class(audio_processor, plugin_specific_config)

                    # Register plugin instance
                    self.plugins[plugin_id] = plugin_instance

                    # Initialize plugin UI components
                    plugin_instance.initialize_ui(main_window)

                    logger.info(f"Initialized plugin: {plugin_id}")

                except Exception as e:
                    logger.error(f"Error initializing plugin {plugin_id}: {e}")

    def get_plugin(self, plugin_id: str):
        """
        Get a plugin instance by ID.

        Args:
            plugin_id: The ID of the plugin to retrieve

        Returns:
            The plugin instance, or None if not found
        """
        return self.plugins.get(plugin_id)

    def process_audio_chunk(self, audio_chunk, sample_rate):
        """
        Process an audio chunk with all active plugins.

        Args:
            audio_chunk: The audio data chunk
            sample_rate: The audio sample rate

        Returns:
            Dict of plugin_id -> results from each plugin
        """
        results = {}

        for plugin_id, plugin in self.plugins.items():
            try:
                results[plugin_id] = plugin.process_audio(audio_chunk, sample_rate)
            except Exception as e:
                logger.error(f"Error in plugin {plugin_id}: {e}")

        return results
