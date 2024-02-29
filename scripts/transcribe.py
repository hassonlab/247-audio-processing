import argparse
import os
import glob
import pandas as pd
from datetime import datetime

from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import (
    VoiceActivityDetection,
    OverlappedSpeechDetection,
)
import torch
import torchaudio
import whisperx
import whisper
import gc

HF_TOKEN = os.environ["HF_TOKEN"]


def arg_parser():
    """Argument Parser

    Args:

    Returns:
        args (namespace): commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--conv-idx", type=str, required=True)
    parser.add_argument("--sid", type=str, required=True)

    args = parser.parse_args()

    if args.conv_idx.isdigit():  # conv idx
        conv_dir = f"data/tfs/{args.sid}/*"
        conv_list = sorted(glob.glob(conv_dir))
        conv_name = os.path.basename(conv_list[int(args.conv_idx)])
        args.audio_filename = (
            f"data/tfs/{args.sid}/{conv_name}/audio/{conv_name}_deid.wav"
        )
    else:  # conv name for testing
        args.audio_filename = f"data/podcast/{args.conv_idx}.wav"  # short test

    result_dir1 = os.path.join("results", args.sid, f"vad")
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    result_dir2 = os.path.join("results", args.sid, f"dia")
    if not os.path.exists(result_dir2):
        os.makedirs(result_dir2)
    result_dir3 = os.path.join("results", args.sid, f"speaker")
    if not os.path.exists(result_dir3):
        os.makedirs(result_dir3)

    args.vad_filename = os.path.join(result_dir1, f"{conv_name}.csv")
    args.dia_filename = os.path.join(result_dir2, f"{conv_name}.csv")
    args.speaker_filename = os.path.join(result_dir3, f"{conv_name}.csv")
    args.device = "cuda"

    return args


def transcribe_whisper(args, audio):
    print("Transcribe with original whisper")
    start_time = datetime.now()

    model = whisper.load_model("tiny.en")
    result = model.transcribe(audio, language="en")
    # result = model.transcribe(args.audio_filename)

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return result


def align_whisperx(args, audio, result):
    print("Align whisper output")
    start_time = datetime.now()

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=args.device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        args.device,
        return_char_alignments=True,
    )

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return result


def vad_pyannote(args):
    start_time = datetime.now()

    # new code, loading from huggingface
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
    }
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=HF_TOKEN)

    # VAD
    print("VAD with pyannote")
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(args.audio_filename)
    vad_df = pd.DataFrame(
        vad.itertracks(yield_label=True), columns=["segment", "label", "speaker"]
    )
    vad_df["start"] = vad_df.segment.apply(lambda x: x.start)
    vad_df["end"] = vad_df.segment.apply(lambda x: x.end)

    # OSD
    print("OSD with pyannote")
    pipeline2 = OverlappedSpeechDetection(segmentation=model)
    pipeline2.instantiate(HYPER_PARAMETERS)
    osd = pipeline2(args.audio_filename)
    osd_df = pd.DataFrame(
        osd.itertracks(yield_label=True), columns=["segment", "label", "speaker"]
    )
    osd_df["start"] = osd_df.segment.apply(lambda x: x.start)
    osd_df["end"] = osd_df.segment.apply(lambda x: x.end)

    waveform, sample_rate = torchaudio.load(args.audio_filename)
    result_df = pd.concat((vad_df, osd_df)).reset_index(drop=True)
    result_df["len"] = waveform.shape[1] / sample_rate
    result_df["sid"] = args.sid
    result_df["conv"] = args.conv_idx
    result_df = result_df.loc[:, ("sid", "conv", "speaker", "start", "end", "len")]

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return result_df


def diarization_pyannote(args):
    start_time = datetime.now()
    print("Diarization with Pyannote")
    # Diarization
    pipeline3 = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
    )
    pipeline3.to(torch.device("cuda"))
    waveform, sample_rate = torchaudio.load(args.audio_filename)
    dia, speaker_embs = pipeline3(
        {"waveform": waveform, "sample_rate": sample_rate}, return_embeddings=True
    )
    dia_df = pd.DataFrame(
        dia.itertracks(yield_label=True), columns=["segment", "label", "speaker"]
    )
    dia_df["start"] = dia_df.segment.apply(lambda x: x.start)
    dia_df["end"] = dia_df.segment.apply(lambda x: x.end)
    dia_df["len"] = waveform.shape[1] / sample_rate
    dia_df["sid"] = args.sid
    dia_df["conv"] = args.conv_idx
    dia_df = dia_df.loc[:, ("sid", "conv", "speaker", "start", "end", "len")]

    speaker_df = pd.DataFrame()
    speaker_df["speaker"] = dia.labels()
    speaker_df["embs"] = speaker_embs.tolist()
    speaker_df["conv"] = args.conv_idx
    speaker_df["sid"] = args.sid

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return dia_df, speaker_df


def get_datum(result):
    print("Getting Datum")
    start_time = datetime.now()

    data = []
    word_idx = 0
    for segment in result["segments"]:
        for word in segment["words"]:
            data.append(pd.DataFrame(word, index=[word_idx]))
            word_idx += 1
    df = pd.concat(data)

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return df


def main():
    args = arg_parser()

    # vad_df = vad_pyannote(args)
    # dia_df, speaker_df = diarization_pyannote(args)

    # vad_df.to_csv(args.vad_filename, index=False)
    # dia_df.to_csv(args.dia_filename, index=False)
    # speaker_df.to_csv(args.speaker_filename, index=False)

    # load audio
    audio = whisper.load_audio(args.audio_filename)
    result = transcribe_whisper(args, audio)
    breakpoint()
    # result = align_whisperx(args, audio, result)

    # saving results
    # df = get_datum(result)
    # df.to_csv(args.out_filename, index=False)

    return


if __name__ == "__main__":
    main()
