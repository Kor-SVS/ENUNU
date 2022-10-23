#!/usr/bin/env python3
# Copyright (c) 2022 oatsu
"""
TMPファイル(UTAUプラグインに渡されるUST似のファイル) を
フルラベル(full_score)とモノラベル(mono_score)に変換する。
"""
import utaupy


from enunu_kor_tool.utaupyk import _ust2hts as utaupyk_ust2hts

g2p_converter = None


def utauplugin2score(path_plugin_in, path_table, path_full_out, lang_mode="jpn", strict_sinsy_style=False):
    """
    UTAUプラグイン用のファイルをフルラベルファイルに変換する。
    """
    # プラグイン用一時ファイルを読み取る
    plugin = utaupy.utauplugin.load(path_plugin_in)
    # 変換テーブルを読み取る
    table = utaupy.table.load(path_table, encoding="utf-8")

    # 2ノート以上選択されているかチェックする
    if len(plugin.notes) < 2:
        raise Exception("ENUNU requires at least 2 notes. / ENUNUを使うときは2ノート以上選択してください。")

    # 歌詞が無いか空白のノートを休符にする。
    for note in plugin.notes:
        if note.lyric.strip(" 　") == "":
            note.lyric = "R"
        # フルラベルの区切り文字と干渉しないように符号を置換する
        if note.flags != "":
            note.flags = note.flags.replace("-", "n")
            note.flags = note.flags.replace("+", "p")

    # classを変更
    ust = plugin.as_ust()
    # フルラベル用のclassに変換
    if lang_mode == "kor":
        from enunu_kor_tool import g2pk4utau

        global g2p_converter

        if g2p_converter == None:
            print("> g2pk4utau Init")
            g2p_converter = g2pk4utau.g2pk4utau()

        song = utaupyk_ust2hts.ustobj2songobj(ust, table, g2p_converter)
    else:
        song = utaupy.utils.ustobj2songobj(ust, table)
    # ファイル出力
    song.write(path_full_out, strict_sinsy_style)
