import os
import glob
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns


def plot_tsne(df, x, y, freq, color, title):
    colors_distinct = [
        "#000000",
        "#00FF00",
        "#0000FF",
        "#FF0000",
        "#01FFFE",
        "#FFA6FE",
        "#774D00",
        "#006401",
        "#010067",
        "#95003A",
        "#007DB5",
        "#FF00F6",
        "#FFEEE8",
        "#FFDB66",
        "#90FB92",
        "#0076FF",
        "#D5FF00",
        "#FF937E",
        "#6A826C",
        "#FF029D",
        "#FE8900",
        "#7A4782",
        "#7E2DD2",
        "#85A900",
        "#FF0056",
        "#A42400",
        "#00AE7E",
        "#683D3B",
        "#BDC6FF",
        "#263400",
        "#BDD393",
        "#00B917",
        "#9E008E",
        "#001544",
        "#C28C9F",
        "#FF74A3",
        "#01D0FF",
        "#004754",
        "#E56FFE",
        "#788231",
        "#0E4CA1",
        "#91D0CB",
        "#BE9970",
        "#968AE8",
        "#BB8800",
        "#43002C",
        "#DEFF74",
        "#00FFC6",
        "#FFE502",
        "#620E00",
        "#008F9C",
        "#98FF52",
        "#7544B1",
        "#B500FF",
        "#00FF78",
        "#FF6E41",
        "#005F39",
        "#6B6882",
        "#5FAD4E",
        "#A75740",
        "#A5FFD2",
        "#FFB167",
        "#009BFF",
        "#E85EBE",
    ]
    df2 = df.copy()
    g = df.groupby(df[color])
    df2 = g.filter(lambda x: len(x) >= freq)
    df2["freq"] = df2.groupby(df2[color])[color].transform("count")
    df2.sort_values("freq", inplace=True, ascending=False)
    df2["marker"] = 1
    # plt.style.use("/scratch/gpfs/ln1144/247-plotting/scripts/paper.mlpstyle")
    sns.scatterplot(
        data=df2,
        x=df2[x],
        y=df2[y],
        hue=df2[color],
        palette=colors_distinct[0 : len(df2[color].unique())],
        linewidth=0,
        style=df2["marker"],
        s=5,
        markers=["o"],
    )
    plt.title(f"{title}")
    # plt.show()
    plt.savefig(f"{title}.jpeg")
    plt.close()
    return


def main():
    return


if __name__ == "__main__":
    main(0)
