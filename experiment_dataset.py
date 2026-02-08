#!/usr/bin/env python3
"""Compatibility exports for dataset helpers moved under src/."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cot_uncertainty.experiment_dataset import *  # noqa: F401,F403
