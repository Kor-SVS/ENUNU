# import json
# from pathlib import Path
# from warnings import warn

# import pysptk
# import pyworld
# from hydra.utils import instantiate
# from nnmnkwii.io import hts
# from nnmnkwii.postfilters import merlin_post_filter
# from nnsvs.dsp import bandpass_filter
# from nnsvs.gen import (
#     gen_spsvs_static_features,
#     gen_world_params,
#     postprocess_duration,
#     predict_acoustic,
#     predict_duration,
#     predict_timelag,
# )
# from nnsvs.io.hts import segment_labels
# from nnsvs.multistream import get_static_stream_sizes
# from nnsvs.pitch import lowpass_filter
# from nnsvs.postfilters import variance_scaling

import os
from typing import Tuple
import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nnsvs.base import BaseModel
from nnsvs import util

from enulib.common import set_checkpoint, set_normalization_stat

try:
    from parallel_wavegan.utils import load_model

    _pwg_available = True
except ImportError:
    _pwg_available = False


class ModelManager(object):
    def __init__(self):
        self.model_dict = {}

    def load_scaler(self, typ: str, config: DictConfig):
        # maybe_set_normalization_stats_(config) のかわり
        set_normalization_stat(config, typ)

        in_scaler: MinMaxScaler = joblib.load(config[typ].in_scaler_path)
        out_scaler: StandardScaler = joblib.load(config[typ].out_scaler_path)

        return in_scaler, out_scaler

    def load_config(self, typ: str, config: DictConfig):
        # maybe_set_checkpoints_(config) のかわり
        set_checkpoint(config, typ)

        # 各種設定を読み込む
        model_config = OmegaConf.load(to_absolute_path(config[typ].model_yaml))

        return model_config

    def load_model(self, typ: str, config: DictConfig, device: str):
        # 各種設定を読み込む
        model_config = self.load_config(typ, config)
        model: BaseModel = hydra.utils.instantiate(model_config.netG).to(device)
        checkpoint = torch.load(config[typ].checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model_config, model

    def __load_model_package(self, typ: str, config: DictConfig, device: str):
        return *self.load_model(typ, config, device), *self.load_scaler(typ, config)

    def get_timelag_model(self, config: DictConfig, device: str):
        return self.__load_model_package("timelag", config, device)

    def get_duration_model(self, config: DictConfig, device: str):
        return self.__load_model_package("duration", config, device)

    def get_acoustic_model(self, config: DictConfig, device: str):
        return self.__load_model_package("acoustic", config, device)

    def get_post_filter_model(self, config: DictConfig, device: str, post_filter_type: str) -> Tuple[DictConfig, BaseModel, StandardScaler]:
        postfilter_model_config, postfilter_model, postfilter_out_scaler = None, None, None

        if post_filter_type in ["nnsvs", "gv"]:
            postfilter_out_scaler = joblib.load(config["postfilter"].out_scaler_path)

            if post_filter_type == "nnsvs":
                postfilter_model_config = OmegaConf.load(to_absolute_path(config["postfilter"].model_yaml))
                postfilter_model = hydra.utils.instantiate(postfilter_model_config.netG).to(device)

        return postfilter_model_config, postfilter_model, postfilter_out_scaler

    def get_vocoder_model(self, config: DictConfig, device: str) -> Tuple[DictConfig, BaseModel, util.StandardScaler]:
        if not _pwg_available:
            raise ValueError('Unable to load "parallel_wavegan" library')

        typ = "vocoder"

        # setup vocoder model path
        if config.model_dir is None:
            raise ValueError('"model_dir" config is required')

        model_dir = to_absolute_path(config.model_dir)
        config[typ].model_yaml = os.path.join(model_dir, typ, "config.yml")
        config[typ].checkpoint = os.path.join(model_dir, typ, config[typ].checkpoint)

        # setup vocoder scaler path
        if config.stats_dir is None:
            raise ValueError('"stats_dir" config is required')

        stats_dir = to_absolute_path(config.stats_dir)
        in_vocoder_scaler_mean = os.path.join(stats_dir, f"in_vocoder_scaler_mean.npy")
        in_vocoder_scaler_var = os.path.join(stats_dir, f"in_vocoder_scaler_var.npy")
        in_vocoder_scaler_scale = os.path.join(stats_dir, f"in_vocoder_scaler_scale.npy")

        vocoder_config = OmegaConf.load(to_absolute_path(config[typ].model_yaml))
        vocoder = load_model(config[typ].checkpoint, config=vocoder_config).to(device)
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder_in_scaler = util.StandardScaler(
            np.load(in_vocoder_scaler_mean),
            np.load(in_vocoder_scaler_var),
            np.load(in_vocoder_scaler_scale),
        )

        return vocoder_config, vocoder, vocoder_in_scaler


__GLOBAL_MODEL_MANAGER = ModelManager()


def get_global_model_manager():
    global __GLOBAL_MODEL_MANAGER
    return __GLOBAL_MODEL_MANAGER
