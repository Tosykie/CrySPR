"""Analysis tools"""

import pandas as pd
try:
    from pandarallel import pandarallel
    pandarallel.initialize()
except:
    print("Warning: Package pandarallel is not found")

def read_predict_output():
    """To be done"""
    pass
