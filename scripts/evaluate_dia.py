import os
import glob
import numpy as np
import pandas as pd


def dia_summary():

    dia_dir = "results/%s/dia/*.csv"

    results = []
    for sid in [625, 676, 7170, 798]:
        dia_files = glob.glob(dia_dir % sid)
        for dia_file in dia_files:
            dia_df = pd.read_csv(dia_file)
            dia_df["duration"] = dia_df.end - dia_df.start
            for speaker in dia_df.speaker.unique():
                dia_df_speaker = dia_df[dia_df.speaker == speaker]
                result = [
                    dia_df_speaker.sid.unique()[0],
                    dia_df_speaker.conv.unique()[0],
                    speaker,
                    dia_df_speaker.duration.sum(),
                    dia_df_speaker.len.unique()[0],
                ]
                results.append(result)
    results = pd.DataFrame(results)
    results.columns = ["sid", "conv_idx", "speaker", "speaker_len", "audio_len"]
    results.sort_values(by=["sid", "conv_idx", "speaker"], inplace=True)
    results.to_csv("summary_dia.csv", index=False)

    return results


def main():

    # dia_df = dia_summary()
    dia_df = pd.read_csv("summary_dia.csv")

    speaker_dir = "results/%s/speaker/*.csv"
    results = pd.DataFrame()
    for sid in [625, 676, 7170, 798]:
        speaker_files = glob.glob(speaker_dir % sid)
        for speaker_file in speaker_files:
            speaker_df = pd.read_csv(speaker_file)
            results = pd.concat((results, speaker_df))
    results.columns = ["speaker", "emb", "conv_idx", "sid"]
    results.reset_index(drop=True, inplace=True)

    dia_df = dia_df.merge(results, how="inner", on=["sid", "conv_idx", "speaker"])
    dia_df.to_csv("summary_speaker.csv", index=False)
    breakpoint()

    return


if __name__ == "__main__":
    main()
