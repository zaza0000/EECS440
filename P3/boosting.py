import numpy as np
import pandas as pd

def boosting(pre1,pre2,pre3):
    weight[3] = []

    prediction = pre1* weight[0] + pre2 * weight[1] + pre3 * weight[2]
    prediction_renew = sign(prediction)

    return 0




