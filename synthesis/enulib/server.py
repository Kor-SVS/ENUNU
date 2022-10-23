#! /usr/bin/env python3
# coding: utf-8
# fmt: off

print("> CP6 Patched Version ver.0.6")
from enunu_kor_tool import version as ekt_ver
print(f"> enunu_kor_tool ver.{ekt_ver.version}")

import json
import os
import subprocess
import traceback

import numpy as np
import enulib
from enulib import enu_logic
try:
    import zmq
except ModuleNotFoundError:
    python_exe = os.path.join('.', 'python-3.8.10-embed-amd64', 'python.exe')
    command = [python_exe, '-m', 'pip', 'install', 'pyzmq']
    print('command:', command)
    subprocess.run(command, check=True)
    import zmq
# fmt: on


def timing(path_ust: str):
    config, temp_dir, _ = enu_logic.setup(path_ust)
    path_full_timing, path_mono_timing = enu_logic.run_timing(config, temp_dir)
    return {
        "path_full_timing": path_full_timing,
        "path_mono_timing": path_mono_timing,
    }


def acoustic(path_ust: str):
    config, temp_dir, _ = enu_logic.setup(path_ust)
    (
        path_temp_ust,
        path_temp_table,
        path_full_score,
        path_mono_score,
        path_full_timing,
        path_mono_timing,
        path_acoustic,
        path_f0,
        path_spectrogram,
        path_aperiodicity,
    ) = enu_logic.get_paths(temp_dir)
    enulib.utauplugin2score.utauplugin2score(path_temp_ust, path_temp_table, path_full_timing, strict_sinsy_style=False, lang_mode=config.get("lang_mode", "jpn"))
    path_acoustic, path_f0, path_spectrogram, path_aperiodicity = enu_logic.run_acoustic(config, temp_dir)
    for path in (path_f0, path_spectrogram, path_aperiodicity):
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
        np.save(path[:-4] + ".npy", arr)
        if os.path.isfile(path):
            os.remove(path)
    return {
        "path_acoustic": path_acoustic,
        "path_f0": path_f0,
        "path_spectrogram": path_spectrogram,
        "path_aperiodicity": path_aperiodicity,
    }


def vocoder_synthesis(path_ust: str, path_wav: str):
    config, temp_dir, _ = enu_logic.setup(path_ust)
    (
        path_temp_ust,
        path_temp_table,
        path_full_score,
        path_mono_score,
        path_full_timing,
        path_mono_timing,
        path_acoustic,
        path_f0,
        path_spectrogram,
        path_aperiodicity,
    ) = enu_logic.get_paths(temp_dir)
    enulib.utauplugin2score.utauplugin2score(path_temp_ust, path_temp_table, path_full_timing, strict_sinsy_style=False, lang_mode=config.get("lang_mode", "jpn"))
    path_acoustic, _, _, _ = enu_logic.run_acoustic(config, temp_dir)

    enu_logic.run_synthesizer(config, temp_dir, path_wav)

    return {"path_wav": path_wav}


def poll_socket(socket, timetick=100):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 100msec
    try:
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv()
    except KeyboardInterrupt:
        print("done.")
    # Escape while loop if there's a keyboard interrupt.


def start_server(port=15555):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print("Started enunu server")

    for message in poll_socket(socket):
        request = json.loads(message)
        print("Received request: %s" % request)

        response = {}
        try:
            if request[0] == "timing":
                response["result"] = timing(request[1])
            elif request[0] == "acoustic":
                response["result"] = acoustic(request[1])
            elif request[0] == "vocoder":
                response["result"] = vocoder_synthesis(request[1], request[2])
            else:
                raise NotImplementedError("unexpected command %s" % request[0])
        except Exception as e:
            response["error"] = str(e)
            traceback.print_exc()

        print("Sending response: %s" % response)
        socket.send_string(json.dumps(response))
