
import requests
import time
import json
import os
import numpy as np


results_dict = np.load('107k_result_dict.npy',allow_pickle='TRUE').item()

results_thres_dict = np.load('107k_result_dict_thres.npy',allow_pickle='TRUE').item()