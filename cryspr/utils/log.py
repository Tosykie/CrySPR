"""Utilities for log"""

from datetime import datetime
def now():
    return datetime.now().strftime("%Y-%b-%d %H:%M:%S")