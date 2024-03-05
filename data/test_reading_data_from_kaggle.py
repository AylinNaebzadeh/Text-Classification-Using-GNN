import numpy as np 
import pandas as pd
import os


if __name__ == "__main__":
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))