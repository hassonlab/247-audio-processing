import os
import glob
import evaluate
import pandas as pd
import numpy as np

# import whisper


def evaluate_preds(transcript, transcript_pred):
    # evaluate
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=transcript_pred, references=transcript)
    metric = evaluate.load("cer")
    cer = metric.compute(predictions=transcript_pred, references=transcript)
    print(f"WER: {wer}, CER: {cer}")
    return (wer, cer)


def evaluate_preds_chunk(grtr, pred):
    pred_len = max(pred.start.max(), pred.end.max())
    bins = np.arange(0, pred_len, step=300)
    bins = np.append(bins, pred_len)
    grtr["chunk"] = pd.cut(grtr.start, bins)
    pred["chunk"] = pd.cut(pred.start, bins)
    grtr_grouped = grtr.groupby("chunk")
    pred_grouped = pred.groupby("chunk")

    results = []
    for (chunk1, grtr_group), (chunk2, pred_group) in zip(grtr_grouped, pred_grouped):
        result = []

        grtr_group, pred_group = delete_inaudible(grtr_group, pred_group)
        result.append(chunk1)
        result.append(len(grtr_group))
        result.append(len(pred_group))
        result.append(len(grtr_group.speaker.unique()))
        result.append(len(pred_group.speaker.unique()))
        result.append(  # utterance num in gt
            grtr_group.speaker.ne(grtr_group.speaker.shift()).cumsum().max()
        )
        result.append(  # utterance num in pred
            pred_group.speaker.ne(pred_group.speaker.shift()).cumsum().max()
        )
        grtr_trans = " ".join(grtr_group.word.astype(str).tolist())
        pred_trans = " ".join(pred_group.word.astype(str).tolist())
        if len(grtr_group) == 0 or len(pred_group) == 0:  # silence
            continue
        wer, cer = evaluate_preds([grtr_trans], [pred_trans])
        # if wer >= 1.5:
        #     breakpoint()
        result.append(wer)
        result.append(len(grtr_group) - len(pred_group))

        result.append(pred_group.score.min())
        result.append(pred_group.score.quantile(0.25))
        result.append(pred_group.score.quantile(0.5))
        result.append(pred_group.score.quantile(0.75))
        result.append(pred_group.score.max())
        result.append(pred_group.score.mean())
        result.append(pred_group.score.std())
        result.append(pred_group.score.sem())

        results.append(result)

    return results


def delete_inaudible(grtr_group, pred_group):
    inaud_tags = ["{inaudible}", "inaudible"]
    if sum(grtr_group.word.isin(inaud_tags)) > 0:
        inaud_grtr_group = grtr_group.sort_values(by="start")
        inaud_grtr_group["start"] = inaud_grtr_group.start.shift(-1)
        inaud_grtr_group["end"] = inaud_grtr_group.end.shift()
        inaud_grtr_group = inaud_grtr_group[inaud_grtr_group.word.isin(inaud_tags)]
        for idx in inaud_grtr_group.index:
            inaud_start = inaud_grtr_group.loc[idx, "end"]  # end of previous word
            inaud_end = inaud_grtr_group.loc[idx, "start"]  # start of next word
            # pred_group_len = len(pred_group)
            pred_group = pred_group[
                (pred_group.start <= inaud_start) | (pred_group.start >= inaud_end)
            ]
            # if len(pred_group) < pred_group_len:
            #     print(f"Deleted inaudible chunks: {pred_group_len - len(pred_group)}")
        grtr_group = grtr_group[~grtr_group.word.isin(inaud_tags)]
        return grtr_group, pred_group
    else:
        return grtr_group, pred_group

    return


def get_pred(predict_file):
    # get prediction
    df_pred = pd.read_csv(predict_file)
    df_pred["onset"] = df_pred.start * 512 - 3000
    df_pred["offset"] = df_pred.end * 512 - 3000
    df_pred["word"] = df_pred["word"].str.strip()
    df_pred["word"] = df_pred["word"].apply(  # getting rid of punctuations
        lambda x: str(x).translate(
            str.maketrans("", "", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
        )
    )

    return df_pred


def get_transcript(datum_file):
    # original datum
    df = pd.read_csv(
        datum_file,
        sep=" ",
        header=None,
        names=["word", "onset", "offset", "accuracy", "speaker"],
    )
    df["word"] = df["word"].str.strip()
    df["start"] = (df.onset + 3000) / 512
    df["end"] = (df.offset + 3000) / 512
    # exclude_words = ["sp", "{lg}", "{ns}", "{LG}", "{NS}", "SP", "{inaudible}"]
    exclude_words = ["sp", "{lg}", "{ns}", "{LG}", "{NS}", "SP"]
    non_words = ["hm", "huh", "mhm", "mm", "oh", "uh", "uhuh", "um"]
    df = df[~df.word.isin(exclude_words)]
    df = df[~df.word.isin(non_words)]

    return df


def get_speakers_utts(pred):
    pred["utt"] = pred.speaker.ne(pred.speaker.shift()).cumsum()

    def get_speaker_utts(groupdf):
        return len(groupdf.utt.unique())

    speakers_utts = pred.groupby(pred.speaker).apply(get_speaker_utts).tolist()
    return speakers_utts


def main():
    # models = ["large", "large-v2"]
    # models = ["large-v1-x", "large-v2-x", "medium.en-x"]
    models = ["large-v2-x"]

    sid = "676"
    # conv_dir = f"data/tfs/{sid}/*"
    # conv_files = sorted(glob.glob(conv_dir))

    for model in models:
        print(f"Running {model}")
        total_result = pd.DataFrame()
        conv_dir = f"results/{sid}/{model}/*"
        conv_files = sorted(glob.glob(conv_dir))
        for conv in conv_files:
            print(f"\tRunning {conv}")
            gt_file_string = os.path.basename(conv).replace(".csv", "")
            gt_file_path = glob.glob(f"data/tfs/{sid}/{gt_file_string}_*")
            if len(gt_file_path) != 1:
                print("WRONG WRONG WRONG")
                breakpoint()
            transcript = get_transcript(gt_file_path[0])
            pred = get_pred(conv)
            results = evaluate_preds_chunk(transcript, pred)
            results = pd.DataFrame(results)
            results.columns = [
                "chunk",
                "gt_word_num",
                "pr_word_num",
                "gt_speaker",
                "pr_speaker",
                "gt_utt_num",
                "pr_utt_num",
                "wer",
                "word_num_diff",
                "score_min",
                "score_25",
                "score_50",
                "score_75",
                "score_max",
                "score_mean",
                "score_std",
                "score_sem",
            ]
            results["conversation"] = gt_file_string

            speakers_utts = get_speakers_utts(pred)
            results["speakers_utts"] = np.tile(
                speakers_utts, (len(results), 1)
            ).tolist()
            total_result = pd.concat((total_result, results))
        total_result.to_csv(f"{sid}-{model}.csv", index=False)

    return


if __name__ == "__main__":
    main()
