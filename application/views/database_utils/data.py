from asyncio.log import logger
import io
import os
from scipy import io as sio
from scipy import sparse
import sqlite3

from ..utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from ..utils.config_utils import config
from ..utils.log_utils import logger
from .video import VideoHelper
from time import time
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np


def encoding_pair(i, j):
    return str(i) + "-" + str(j)

def decoding_pair(pair):
    a, b = pair.split("-")
    return int(a), int(b)

def feature_norm(features):
    norms = (features**2).sum(axis=1)
    norms = norms ** 0.5
    features = features / norms[:, np.newaxis]
    return features


class DatabaseLoader():
    '''
    base operation of database
    '''

    def __init__(self, dataname, step=0):
        # open database
        self.dataname = dataname
        self.step = step 
        self.common_data_root = os.path.join(config.data_root, dataname)
        self.step_data_root = os.path.join(self.common_data_root, "step" + str(step), )
        # self.save_path = 'data/' + dataname + '/step_' + str(step)
        # self.meta_data = pickle_load_data(self.save_path + '/meta_info.pkl')
        # self.conn = sqlite3.connect(self.save_path + '/database.db', detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        # sqlite3.register_adapter(np.ndarray, self._adapt_array)
        # sqlite3.register_converter("array", self._convert_array)

    def get_feature_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select feature
                            from data 
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        fea_index = cursor.fetchone()[0]
        cursor.execute('''  select feature
                            from features
                            where id = {}   '''.format(fea_index))
        return cursor.fetchone()[0]
    
    def get_classification_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select classification
                            from data
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        return cursor.fetchone()[0]

    def get_action_score_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select action_score
                            from data
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        return cursor.fetchone()[0]
    
    def get_ground_truth_of_single_frame(self, video_id, frame_id):
        cursor = self.conn.cursor()
        cursor.execute('''  select groundtruth
                            from data
                            where video_id = {}
                            and frame_id = {}  '''.format(video_id, frame_id))
        return cursor.fetchone()[0]

    def _adapt_array(self, arr: np.ndarray):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return bytes(sqlite3.Binary(out.read()))

    def _convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

class Data(DatabaseLoader):
    def __init__(self, dataname, step=0, mode="pred"):
        # load some meta information
        super(Data, self).__init__(dataname, step)
        self.debug = False
        t0 = time()