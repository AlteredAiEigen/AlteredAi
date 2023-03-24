import sys
import os

# Add the parent directory to the Python path

from AlteredAi.core.DataLoader import DataLoader
#import tensorflow as tf

class tensorflow(DataLoader):
    def __init__(self):
        print("AlteredAiTensorflow Loaded Successfully")

    def funTensorflow(self):
        print("This is function from Tensorflow Class")

print("test") 
