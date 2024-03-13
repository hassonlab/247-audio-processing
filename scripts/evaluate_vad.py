import os
import glob
import numpy as np
import pandas as pd


def main():

    vad_dir = "results/%s/vad/*.csv"

    results = []
    results_all = pd.DataFrame()
    for sid in [625, 676, 7170, 798]:
        vad_files = glob.glob(vad_dir % sid)
        for vad_file in vad_files:
            vad_df = pd.read_csv(vad_file)
            vad_df["duration"] = vad_df.end - vad_df.start
            vad = vad_df.loc[vad_df.speaker == "SPEECH", :]
            osd = vad_df.loc[vad_df.speaker == "OVERLAP", :]
            result = [
                vad.sid.unique()[0],
                vad.conv.unique()[0],
                vad.duration.sum(),
                osd.duration.sum(),
                vad.len.unique()[0],
            ]
            results.append(result)
            results_all = pd.concat((results_all, vad_df))
    results = pd.DataFrame(results)
    results.columns = ["sid", "conv_idx", "vad", "osd", "audio_len"]
    results.sort_values(by=["sid", "conv_idx"], inplace=True)
    results.to_csv("summary_vad.csv", index=False)
    results_all.to_csv("all_vad.csv", index=False)

    return


if __name__ == "__main__":
    main()
