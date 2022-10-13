#! /usr/bin/env python3
# coding: utf-8
# Copyright (c) 2020 oatsu
"""
1. UTAUプラグインのテキストファイルを読み取る。
  - 音源のフォルダを特定する。
  - プロジェクトもしくはUSTファイルのパスを特定する。
2. LABファイルを(一時的に)生成する
  - キャッシュフォルダでいいと思う。
3. LABファイル→WAVファイル
"""
import sys
import warnings
from os.path import dirname
from sys import argv
from typing import Union

import colored_traceback.always  # pylint: disable=unused-import

# ENUNUのフォルダ直下にあるenulibフォルダをimportできるようにする
sys.path.append(dirname(__file__))
warnings.simplefilter("ignore")

from enulib import enu_logic

# try:
#     import enulib
# except ModuleNotFoundError:
#     print("----------------------------------------------------------")
#     print("初回起動ですね。")
#     print("PC環境に合わせてPyTorchを自動インストールします。")
#     print("インストール完了までしばらくお待ちください。")
#     print("----------------------------------------------------------")
#     from install_torch import pip_install_torch

#     pip_install_torch(join(".", "python-3.8.10-embed-amd64", "python.exe"))
#     print("----------------------------------------------------------")
#     print("インストール成功しました。歌声合成を始めます。")
#     print("----------------------------------------------------------\n")
#     import enulib


def main(path_plugin: str, path_wav_out: Union[str, None] = None):
    """
    入力ファイルによって処理を分岐する。
    """

    # logging.basicConfig(level=logging.INFO)
    if path_plugin.endswith(".tmp"):
        enu_logic.main_as_plugin(path_plugin, path_wav_out)
    else:
        raise ValueError("Input file must be TMP(plugin).")


if __name__ == "__main__":
    print("______ξ ・ヮ・)ξ < ENUNU v0.6.0 ________")
    print("______ [ CP6 Patched Version ] _________")

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", dest="mode", choices=["legacy", "server"], default="legacy", help="Plugin start mode.")
    parser.add_argument("--port", dest="port", default=15555, help="Server port.")

    args, unknown_args = parser.parse_known_args()
    args = vars(args)

    print(f"args: {args}")
    print(f"unknown_args: {unknown_args}")

    if args["mode"].lower() == "legacy":
        if len(unknown_args) > 0:
            main(*(unknown_args[:2]))
        elif len(argv) == 1:
            main(input("Input file path of TMP(plugin)\n>>> ").strip('"'), None)
        else:
            raise Exception("引数が多すぎます。/ Too many arguments.")
    elif args["mode"] == "server":
        print("Starting enunu server...")

        from enulib import server

        server.start_server(args["port"])
