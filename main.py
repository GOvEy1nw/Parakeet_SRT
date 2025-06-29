import nemo.collections.asr as nemo_asr
import torch
import gc
import io
import itertools
from typing import BinaryIO, Union
import av
import numpy as np
import os


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
):
    """Decodes the audio.
    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
    Returns:
      A float32 Numpy array.
    """

    def _ignore_invalid_frames(frames):
        iterator = iter(frames)

        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break
            except av.error.InvalidDataError:
                continue

    def _group_frames(frames, num_samples=None):
        fifo = av.audio.fifo.AudioFifo()

        for frame in frames:
            frame.pts = None  # Ignore timestamp check.
            fifo.write(frame)

            if num_samples is not None and fifo.samples >= num_samples:
                yield fifo.read()

        if fifo.samples > 0:
            yield fifo.read()

    def _resample_frames(frames, resampler):
        # Add None to flush the resampler.
        for frame in itertools.chain(frames, [None]):
            yield from resampler.resample(frame)

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sampling_rate,
    )

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    # It appears that some objects related to the resampler are not freed
    # unless the garbage collector is manually run.
    # https://github.com/SYSTRAN/faster-whisper/issues/390
    # note that this slows down loading the audio a little bit
    # if that is a concern, please use ffmpeg directly as in here:
    # https://github.com/openai/whisper/blob/25639fc/whisper/audio.py#L25-L62
    del resampler
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    return audio


def format_srt_timestamp(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,ms."""
    total_milliseconds = int(seconds * 1000)

    hours = total_milliseconds // 3600000
    minutes = (total_milliseconds % 3600000) // 60000
    seconds = (total_milliseconds % 60000) // 1000
    milliseconds = total_milliseconds % 1000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


device = torch.device("cuda")

asr_model = nemo_asr.models.ASRModel.restore_from(
    restore_path="models/parakeet-tdt-0.6b-v2.nemo", map_location=device
)
asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
asr_model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

file_path = "files/sample2.mkv"
output_srt_path = os.path.splitext(file_path)[0] + ".srt"

audio = decode_audio(
    file_path, sampling_rate=asr_model.preprocessor._cfg["sample_rate"]
)
output = asr_model.transcribe(audio, timestamps=True)
# by default, timestamps are enabled for char, word and segment level
word_timestamps = output[0].timestamp["word"]  # word level timestamps for first sample
segment_timestamps = output[0].timestamp["segment"]  # segment level timestamps
char_timestamps = output[0].timestamp["char"]  # char level timestamps

with open(output_srt_path, "w", encoding="utf-8") as srt_file:
    for i, stamp in enumerate(segment_timestamps, 1):
        start_time = format_srt_timestamp(stamp["start"])
        end_time = format_srt_timestamp(stamp["end"])
        text = stamp["segment"].strip()
        srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

print(f"Subtitle file saved to {output_srt_path}")
