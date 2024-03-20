Configuring your RIME runs


process_config.py
=================

This file is designed to configure settings and working directories for the project. It acts as a central configuration module to be imported across other scripts in the project, ensuring consistent configuration.

Key Features
------------

- **Central Configuration**: Stores and manages settings and directory paths that are used throughout the project.
- **Easy Import**: Can be easily imported with ``from process_config import *``, making all configurations readily available in other scripts.

Dependencies
------------

- ``os``: For interacting with the operating system's file system, likely used to manage file paths and directories.

Usage
-----

This script is not meant to be run directly. Instead, it should be imported at the beginning of other project scripts to ensure they have access to shared configurations, settings, and directory paths.

