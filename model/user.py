import numpy as np
import pandas as pd

class User:
    
    def __init__(self, uid, history):

        self.uid = uid #userID
        self.history = history #history of venues 
