import numpy as np
import os
import json
from time import time

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE

from ..utils.log_utils import logger
from ..utils.config_utils import config
from ..utils.helper_utils import json_load_data, json_save_data
from ..utils.helper_utils import pickle_save_data, pickle_load_data, check_dir
from ..database_utils.data import Data


class BackendModel(object):
    def __init__(self, dataname=None, step=0, case_mode=True):
        self.dataname = dataname
        self.step = step
        self.case_mode = case_mode
        # if config:
        #     self.config = config
        # else:
        #     self.config = {
        #         "step": 1,
        #         "text_k": 9,
        #         "image_k": 9,
        #         "pre_k": 100,
        #         "weight": 1
        #     }
        self.nearest_actions = 3
        self.nearest_frames = 4
        self.window_size = 5
        if dataname is None:
            return 
        self._init()
    
    def update_data_root(self, dataname, step):
        self.step = step
        suffix = "step" + str(step)
        self.common_data_root = os.path.join(config.data_root, dataname)
        self.step_data_root = os.path.join(config.data_root, self.dataname, suffix)
        self.buffer_path = os.path.join(self.step_data_root, "model.pkl")

    def _init(self):
        None

    def reset(self, dataname, step):
        self.dataname = dataname
        self.step = step
        self._init()

    def run(self):
        None


    def save_model(self, path=None):
        logger.info("save model buffer")
        buffer_path = self.buffer_path
        if path:
            buffer_path = path
        tmp_data = self.data
        tmp_video = self.data.video
        self.data.video = None
        self.data = None
        pickle_save_data(buffer_path, self)
        self.data = tmp_data

    def load_model(self, path=None):
        buffer_path = self.buffer_path
        if path:
            buffer_path = path
        self = pickle_load_data(buffer_path)
        self.data = Data(self.dataname, self.step)
    
    def buffer_exist(self, path=None):
        buffer_path = self.buffer_path
        if path:
            buffer_path = path
        logger.info(buffer_path)
        if os.path.exists(buffer_path):
            return True
        else:
            return False
    
    