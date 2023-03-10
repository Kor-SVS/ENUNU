#!/usr/bin/env python3
# Copyright (c) 2022 oatsu
"""
timelagとdurationをまとめて実行する。

MDN系のdurationが確率分布を持って生成されるため、フルラベルにしづらい。
そのため、timelagとdurationをファイル出力せずにtimingまで一気にやる。
"""
import numpy as np
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from nnsvs.gen import postprocess_duration, predict_duration, predict_timelag
from nnsvs.logger import getLogger
from nnsvs.io.hts import get_pitch_index, get_pitch_indices
from omegaconf import DictConfig, OmegaConf

from enulib.common import get_device
from enulib.model_manager import get_global_model_manager

logger = None


def _score2timelag(config: DictConfig, labels):
    """
    全体の処理を実行する。
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

    model_config, model, in_scaler, out_scaler = get_global_model_manager().get_timelag_model(config, device)
    # -----------------------------------------------------
    # ここまで nnsvs.bin.synthesis.my_app() の内容 --------
    # -----------------------------------------------------

    # -----------------------------------------------------
    # ここから nnsvs.bin.synthesis.synthesis() の内容 -----
    # -----------------------------------------------------
    # full_score_lab を読み取る。
    # labels = hts.load(score_path).round_()

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
        force_clip_input_features = config.timelag.force_clip_input_features
    except:
        logger.info(f"force_clip_input_features of timelag is not set so enabled as default")

    # timelagモデルを適用
    # Time-lag
    lag = predict_timelag(
        device,
        labels,
        model,
        model_config,
        in_scaler,
        out_scaler,
        binary_dict,
        numeric_dict,
        pitch_indices,
        config.log_f0_conditioning,
        config.timelag.allowed_range,
        config.timelag.allowed_range_rest,
        force_clip_input_features,
    )
    # -----------------------------------------------------
    # ここまで nnsvs.bin.synthesis.synthesis() の内容 -----
    # -----------------------------------------------------

    # フルラベルとして出力する
    # save_timelag_label_file(lag, score_path, timelag_path)
    return lag


def _score2duration(config: DictConfig, labels):
    """
    full_score と timelag ラベルから durationラベルを生成する。
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

    model_config, model, in_scaler, out_scaler = get_global_model_manager().get_duration_model(config, device)
    # -----------------------------------------------------
    # ここまで nnsvs.bin.synthesis.my_app() の内容 --------
    # -----------------------------------------------------

    # -----------------------------------------------------
    # ここから nnsvs.bin.synthesis.synthesis() の内容 -----
    # -----------------------------------------------------
    # full_score_lab を読み取る。
    # labels = hts.load(score_path).round_()
    # いまのduraitonモデルだと使わない
    # timelag = hts.load(timelag_path).round_()

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
        force_clip_input_features = config.duration.force_clip_input_features
    except:
        logger.info(f"force_clip_input_features of duration is not set so enabled as default")

    # durationモデルを適用
    duration = predict_duration(
        device,
        labels,
        model,
        model_config,
        in_scaler,
        out_scaler,
        binary_dict,
        numeric_dict,
        pitch_indices,
        config.log_f0_conditioning,
        force_clip_input_features,
    )
    # durationのタプルまたはndarrayを返す
    return duration


def score2timing(config: DictConfig, path_score, path_timing):
    """
    full_score から full_timing ラベルを生成する。
    """
    # full_score を読む
    score = hts.load(path_score).round_()
    # timelag
    timelag = _score2timelag(config, score)
    # duration
    duration = _score2duration(config, score)
    # timing
    timing = postprocess_duration(score, duration, timelag)

    # timingファイルを出力する
    with open(path_timing, "w", encoding="utf-8") as f:
        f.write(str(timing))
