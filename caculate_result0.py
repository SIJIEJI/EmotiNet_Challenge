
import requests
import time
import json
import os
import numpy as np
import pandas as pd




def main():

    results_dict = np.load('107k_result_dict.npy',allow_pickle='TRUE').item()
    df = pd.DataFrame([results_dict], columns=results_dict.keys())
    import pdb; pdb.set_trace()
    print (df)
    #


if __name__ == "__main__":
    main()