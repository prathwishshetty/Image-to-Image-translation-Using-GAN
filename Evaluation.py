import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2
import math
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_similarity_score


def calculate_iou(component1,component2):
    overlap = component1*component2 # Logical AND
    union = component1 + component2 # Logical OR

    IOU = overlap.sum()/float(union.sum())
    
    return(IOU)
    
def dice(component1,component2):
    eps = 1e-6
    pred = np.asarray(component1).astype(np.bool)
    gtruth = np.asarray(component2).astype(np.bool)

    if not np.sum(gtruth) + np.sum(pred):
        return 1.0

    return ((2. * np.sum(gtruth * pred)) /(np.sum(gtruth) + np.sum(pred) + eps))
                
               
 
