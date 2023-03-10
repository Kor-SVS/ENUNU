#!/usr/bin/env python3
# Copyright (c) 2022 oatsu
# ---------------------------------------------------------------------------------
#
# MIT License
#
# Copyright (c) 2020 Ryuichi Yamamoto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---------------------------------------------------------------------------------

"""
発声タイミングの情報を持ったフルラベルから、WORLD用の音響特長量を推定する。
"""
from typing import List
import numpy as np
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from nnsvs.gen import predict_acoustic
from nnsvs.logger import getLogger
from omegaconf import DictConfig, OmegaConf

from nnsvs.io.hts import segment_labels, get_pitch_index, get_pitch_indices

from enulib.common import get_device
from enulib.model_manager import get_global_model_manager

logger = None


def timing2acoustic(config: DictConfig, timing_path, acoustic_path, use_segment_label=False):
    """
    フルラベルを読み取って、音響特長量のファイルを出力する。
    """
    # -----------------------------------------------------
    # ここから nnsvs.bin.synthesis.my_app() の内容 --------
    # -----------------------------------------------------
    # loggerの設定
    global logger  # pylint: disable=global-statement
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    # CUDAが使えるかどうか
    device = get_device()

    model_config, model, in_scaler, out_scaler = get_global_model_manager().get_acoustic_model(config, device)
    # -----------------------------------------------------
    # ここまで nnsvs.bin.synthesis.my_app() の内容 --------
    # -----------------------------------------------------

    # -----------------------------------------------------
    # ここから nnsvs.bin.synthesis.synthesis() の内容 -----
    # -----------------------------------------------------
    # full_score_lab を読み取る。
    duration_modified_labels = hts.load(timing_path).round_()

    # hedファイルを読み取る。
    question_path = to_absolute_path(config.question_path)
    # hts2wav.pyだとこう↓-----------------
    # これだと各モデルに別個のhedを適用できる。
    # if config[typ].question_path is None:
    #     config[typ].question_path = config.question_path
    # --------------------------------------
    # hedファイルを辞書として読み取る。
    binary_dict, numeric_dict = hts.load_question_set(question_path, append_hat_for_LL=False)
    # pitch indices in the input features
    # pitch_idx = get_pitch_index(binary_dict, numeric_dict)
    pitch_indices = get_pitch_indices(binary_dict, numeric_dict)

    # check force_clip_input_features (for backward compatibility)
    force_clip_input_features = True
    try:
        force_clip_input_features = config.acoustic.force_clip_input_features
    except:
        logger.info(f"force_clip_input_features of acoustic is not set so enabled as default")

    if use_segment_label:
        segmented_labels: List[hts.HTSLabelFile] = segment_labels(duration_modified_labels)
        from tqdm.auto import tqdm

        acoustic_features_list = []
        for seg_labels in tqdm(segmented_labels):
            acoustic_features = predict_acoustic(
                device,
                seg_labels,
                model,
                model_config,
                in_scaler,
                out_scaler,
                binary_dict,
                numeric_dict,
                config.acoustic.subphone_features,
                pitch_indices,
                config.log_f0_conditioning,
                force_clip_input_features,
            )
            acoustic_features_list.append(acoustic_features)

        acoustic_features = np.concatenate(acoustic_features_list, axis=0)

    else:
        acoustic_features = predict_acoustic(
            device,
            duration_modified_labels,
            model,
            model_config,
            in_scaler,
            out_scaler,
            binary_dict,
            numeric_dict,
            config.acoustic.subphone_features,
            pitch_indices,
            config.log_f0_conditioning,
            force_clip_input_features,
        )

    # csvファイルとしてAcousticの行列を出力
    np.savetxt(acoustic_path, acoustic_features, delimiter=",")
