#!/usr/bin/env python3
# Copyright (c) 2022 oatsu
"""
acousticのファイルをWAVファイルにするまでの処理を行う。
"""
from typing import List
import torch
import numpy as np
import pysptk
import pyworld
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile

from nnmnkwii.io import hts
from nnmnkwii.postfilters import merlin_post_filter

from nnsvs.dsp import bandpass_filter
from nnsvs.gen import gen_spsvs_static_features, gen_world_params, postprocess_acoustic, predict_waveform, postprocess_waveform
from nnsvs.logger import getLogger
from nnsvs.multistream import get_static_stream_sizes
from nnsvs.pitch import lowpass_filter
from nnsvs.postfilters import variance_scaling
from nnsvs.io.hts import segment_labels, get_pitch_index, get_pitch_indices

from enulib.common import set_checkpoint, set_normalization_stat, get_device
from enulib.model_manager import get_global_model_manager

logger = None


def estimate_bit_depth(wav: np.ndarray) -> str:
    """
    wavformのビット深度を判定する。
    16bitか32bit
    16bitの最大値: 32767
    32bitの最大値: 2147483647
    """
    # 音量の最大値を取得
    max_gain = np.nanmax(np.abs(wav))
    # 学習データのビット深度を推定(8388608=2^24)
    if max_gain > 8388608:
        return "int32"
    if max_gain > 8:
        return "int16"
    return "float"


def generate_wav_file(config: DictConfig, wav, out_wav_path):
    """
    ビット深度を指定してファイル出力(32bit float)
    """
    # 出力された音量をもとに、学習に使ったビット深度を推定
    training_data_bit_depth = estimate_bit_depth(wav)
    # print(training_data_bit_depth)

    # 16bitで学習したモデルの時
    if training_data_bit_depth == "int16":
        wav = wav / 32767
    # 32bitで学習したモデルの時
    elif training_data_bit_depth == "int32":
        wav = wav / 2147483647
    elif training_data_bit_depth == "float":
        pass
    # なぜか16bitでも32bitでもないとき
    else:
        raise ValueError("WAVのbit深度がよくわかりませんでした。")

    # 音量ノーマライズする場合
    if config.gain_normalize:
        wav = wav / np.max(np.abs(wav))

    # ファイル出力
    wav = wav.astype(np.float32)
    wavfile.write(out_wav_path, rate=config.sample_rate, data=wav)


# def acoustic2wav(config: DictConfig, path_timing, path_acoustic, path_wav):
#     """
#     Acousticの行列のCSVを読んで、WAVファイルとして出力する。
#     """
#     # loggerの設定
#     global logger  # pylint: disable=global-statement
#     logger = getLogger(config.verbose)
#     logger.info(OmegaConf.to_yaml(config))

#     # load labels and question
#     duration_modified_labels = hts.load(path_timing).round_()

#     # CUDAが使えるかどうか
#     # device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # 各種設定を読み込む
#     typ = 'acoustic'
#     model_config = OmegaConf.load(to_absolute_path(config[typ].model_yaml))

#     # hedファイルを読み取る。
#     question_path = to_absolute_path(config.question_path)
#     # hts2wav.pyだとこう↓-----------------
#     # これだと各モデルに別個のhedを適用できる。
#     # if config[typ].question_path is None:
#     #     config[typ].question_path = config.question_path
#     # --------------------------------------

#     # hedファイルを辞書として読み取る。
#     binary_dict, numeric_dict = hts.load_question_set(
#         question_path, append_hat_for_LL=False
#     )

#     # pitch indices in the input features
#     pitch_idx = len(binary_dict) + 1
#     # pitch_indices = np.arange(len(binary_dict), len(binary_dict)+3)

#     # pylint: disable=no-member
#     # Acousticの数値を読み取る
#     acoustic_features = np.loadtxt(
#         path_acoustic, delimiter=',', dtype=np.float64
#     )

#     # 設定の一部を取り出す

#     generated_waveform = gen_waveform(
#         duration_modified_labels,
#         acoustic_features,
#         binary_dict,
#         numeric_dict,
#         model_config.stream_sizes,
#         model_config.has_dynamic_features,
#         subphone_features=config.acoustic.subphone_features,
#         log_f0_conditioning=config.log_f0_conditioning,
#         pitch_idx=pitch_idx,
#         num_windows=model_config.num_windows,
#         post_filter=config.acoustic.post_filter,
#         sample_rate=config.sample_rate,
#         frame_period=config.frame_period,
#         relative_f0=config.acoustic.relative_f0
#     )

#     # 音量を調整して 32bit float でファイル出力
#     generate_wav_file(config, generated_waveform, path_wav)


def get_acoustic_feature(
    config: DictConfig,
    path_timing,
    path_acoustic,
    trajectory_smoothing=True,
    trajectory_smoothing_cutoff=50,
    trajectory_smoothing_cutoff_f0=20,
    vibrato_scale=1.0,
    vuv_threshold=0.1,
):
    # loggerの設定
    global logger  # pylint: disable=global-statement
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    device = get_device()
    post_filter_type = "gv"

    # load labels and question
    duration_modified_labels = hts.load(path_timing).round_()

    # hedファイルを読み取る。
    question_path = to_absolute_path(config.question_path)
    # hts2wav.pyだとこう↓-----------------
    # これだと各モデルに別個のhedを適用できる。
    # if config[typ].question_path is None:
    #     config[typ].question_path = config.question_path
    # --------------------------------------

    model_manager = get_global_model_manager()
    acoustic_model_config, model, in_scaler, acoustic_out_static_scaler = get_global_model_manager().get_acoustic_model(config, device)
    postfilter_config, postfilter_model, postfilter_out_scaler = None, None, None
    # postfilter_config, postfilter_model, postfilter_out_scaler = get_global_model_manager().get_post_filter_model(
    #     config,
    #     device,
    #     post_filter_type,)

    # hedファイルを辞書として読み取る。
    binary_dict, numeric_dict = hts.load_question_set(question_path, append_hat_for_LL=False)

    # pitch indices in the input features
    pitch_idx = get_pitch_index(binary_dict, numeric_dict)
    # pitch_indices = get_pitch_indices(binary_dict, numeric_dict)

    # pylint: disable=no-member
    # Acousticの数値を読み取る
    acoustic_features = np.loadtxt(path_acoustic, delimiter=",", dtype=np.float64)

    multistream_features = postprocess_acoustic(
        device=device,
        duration_modified_labels=duration_modified_labels,
        acoustic_features=acoustic_features,
        binary_dict=binary_dict,
        numeric_dict=numeric_dict,
        acoustic_config=acoustic_model_config,
        acoustic_out_static_scaler=acoustic_out_static_scaler,
        postfilter_model=postfilter_model,
        postfilter_config=postfilter_config,
        postfilter_out_scaler=postfilter_out_scaler,
        sample_rate=config.sample_rate,
        frame_period=config.frame_period,
        relative_f0=config.acoustic.relative_f0,
        feature_type="world",
        post_filter_type=post_filter_type,
        trajectory_smoothing=trajectory_smoothing,
        trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
        trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
        vuv_threshold=vuv_threshold,
        vibrato_scale=vibrato_scale,  # only valid for Sinsy-like models
    )

    return multistream_features


def acoustic2world(
    config: DictConfig,
    path_timing,
    path_acoustic,
    path_f0,
    path_spcetrogram,
    path_aperiodicity,
    trajectory_smoothing=True,
    trajectory_smoothing_cutoff=50,
    trajectory_smoothing_cutoff_f0=20,
    vibrato_scale=1.0,
    vuv_threshold=0.1,
):
    """
    Acousticの行列のCSVを読んで、WAVファイルとして出力する。
    """

    mgc, lf0, vuv, bap = get_acoustic_feature(
        config, path_timing, path_acoustic, trajectory_smoothing, trajectory_smoothing_cutoff, trajectory_smoothing_cutoff_f0, vibrato_scale, vuv_threshold)

    # Generate WORLD parameters
    f0, spectrogram, aperiodicity = gen_world_params(
        mgc, lf0, vuv, bap, config.sample_rate, vuv_threshold=vuv_threshold, use_world_codec=config.use_world_codec)

    # csvファイルとしてf0の行列を出力
    for path, array in ((path_f0, f0), (path_spcetrogram, spectrogram), (path_aperiodicity, aperiodicity)):
        np.savetxt(path, array, fmt="%.16f", delimiter=",")


def world2wav(
    config: DictConfig,
    path_f0,
    path_spectrogram,
    path_aperiodicity,
    path_wav,
    vuv_threshold=0.1,
):
    """WORLD用のパラメータからWAVファイルを生成する。"""
    f0 = np.loadtxt(path_f0, delimiter=",", dtype=np.float64)
    spectrogram = np.loadtxt(path_spectrogram, delimiter=",", dtype=np.float64)
    aperiodicity = np.loadtxt(path_aperiodicity, delimiter=",", dtype=np.float64)

    spectrogram[spectrogram == 0] = 1e-16  # 0.0000000000000001

    wav = predict_waveform(
        device="cpu",
        multistream_features=(f0, spectrogram, aperiodicity),
        feature_type="world_org",
        sample_rate=config.sample_rate,
        frame_period=config.frame_period,
        use_world_codec=config.get("use_world_codec", False),
        vuv_threshold=vuv_threshold,
    )

    wav = postprocess_waveform(
        wav=wav,
        dtype=wav.dtype,
        sample_rate=config.sample_rate,
    )

    # 音量を調整して 32bit float でファイル出力
    generate_wav_file(config, wav, path_wav)


def acoustic2vocoder_wav(
    config: DictConfig,
    path_timing,
    path_acoustic,
    path_wav,
    trajectory_smoothing=True,
    trajectory_smoothing_cutoff=50,
    trajectory_smoothing_cutoff_f0=20,
    vibrato_scale=1.0,
    vuv_threshold=0.1,
    use_segment_label=False,
):
    device = get_device()

    duration_modified_labels = hts.load(path_timing).round_()

    mgc, lf0, vuv, bap = get_acoustic_feature(
        config, path_timing, path_acoustic, trajectory_smoothing, trajectory_smoothing_cutoff, trajectory_smoothing_cutoff_f0, vibrato_scale, vuv_threshold)

    model_config, model, in_scaler = get_global_model_manager().get_vocoder_model(config, device)

    vuv = (vuv > vuv_threshold).astype(np.float32)
    voc_inp = torch.from_numpy(in_scaler.transform(np.concatenate([mgc, lf0, vuv, bap], axis=-1))).float()

    if use_segment_label:
        segmented_labels: List[hts.HTSLabelFile] = segment_labels(duration_modified_labels)
        from tqdm.auto import tqdm

        overlap = 100
        overlap_wav = overlap * model_config.hop_size

        wavs = []
        with torch.no_grad():
            end_time_store = 0
            for idx, seg_labels in enumerate(tqdm(segmented_labels)):
                start_time = seg_labels.start_times[0] // 50000 + end_time_store
                end_time = seg_labels.end_times[-1] // 50000 + end_time_store
                end_time_store = end_time

                if idx > 0:
                    start_time -= overlap
                if idx < len(segmented_labels) - 1:
                    end_time += overlap

                wav = model.inference(voc_inp[start_time:end_time].to(device)).view(-1).to("cpu").numpy()

                if idx > 0:
                    wav = wav[overlap_wav:]
                if idx < len(segmented_labels) - 1:
                    wav = wav[:-overlap_wav]

                wavs.append(wav)
                print(f"infer: {start_time} ~ {end_time}")

        # Concatenate segmented wavs
        wav = np.concatenate(wavs, axis=0).reshape(-1)
    else:
        with torch.no_grad():
            wav = model.inference(voc_inp.to(device)).view(-1).to("cpu").numpy()

    # post-processing
    wav = bandpass_filter(wav, model_config.sampling_rate)

    if np.max(wav) > 10:
        # data is likely already in [-32768, 32767]
        wav = wav.astype(np.int16)
    else:
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767.0).astype(np.int16)

    generate_wav_file(config, wav, path_wav)
