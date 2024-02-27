import argparse
import os
import glob
import pandas as pd
from datetime import datetime

from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import (
    VoiceActivityDetection,
    OverlappedSpeechDetection,
    SpeakerDiarization,
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

    result_dir = os.path.join("results", args.sid, f"{args.model}-x")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    args.out_filename = os.path.join(result_dir, f"{conv_name}.csv")
    args.device = "cuda"

    return args


def transcribe_whisperx(args, audio):
    print("Transcribe with whisperx (batched)")
    start_time = datetime.now()
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

    model = whisperx.load_model(args.model, args.device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

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
        return_char_alignments=False,
    )

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return result


def diarization_whisperx(args, result):
    print("Assign Speaker Labels")
    start_time = datetime.now()

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=HF_TOKEN, device=args.device
    )
    diarize_segments = diarize_model(args.audio_filename)
    # diarize_model(args.audio_filename, min_speakers=2, max_speakers=3)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return result


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

    # load audio
    audio = whisperx.load_audio(args.audio_filename)
    result = transcribe_whisperx(args, audio)
    result = align_whisperx(args, audio, result)
    result = diarization_whisperx(args, result)

    # saving results
    df = get_datum(result)
    df.to_csv(args.out_filename, index=False)

    return


if __name__ == "__main__":
    main()
