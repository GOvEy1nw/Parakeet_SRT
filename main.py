import nemo.collections.asr as nemo_asr
import torch
import gc
import io
import itertools
from typing import BinaryIO, Union
import av
import numpy as np
import os
from flask import Flask, request, jsonify

# --- Global variable to hold the ASR model ---
asr_model = None


def process_file(input_path, output_path):
    """Processes a single media file to generate an SRT."""
    print(f"Processing request: {input_path} -> {output_path}")
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        audio = decode_audio(
            input_path, sampling_rate=asr_model.preprocessor._cfg["sample_rate"]
        )
        output = asr_model.transcribe(audio, timestamps=True)

        segment_timestamps = output[0].timestamp["segment"]

        with open(output_path, "w", encoding="utf-8") as srt_file:
            for i, stamp in enumerate(segment_timestamps, 1):
                start_time = format_srt_timestamp(stamp["start"])
                end_time = format_srt_timestamp(stamp["end"])
                text = stamp["segment"].strip()
                srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        print(f"Successfully saved subtitle file to {output_path}")
        return True, None
    except Exception as e:
        print(f"Failed to process {input_path}. Error: {e}")
        return False, str(e)


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


app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    input_path = data.get("input_path")
    output_path = data.get("output_path")

    if not input_path or not output_path:
        return jsonify(
            {"status": "error", "message": "input_path and output_path are required"}
        ), 400

    success, error_message = process_file(input_path, output_path)

    if success:
        return jsonify({"status": "success", "output_path": output_path})
    else:
        return jsonify({"status": "error", "message": error_message}), 500


def main():
    global asr_model

    print("Loading ASR model... (this may take a moment)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = nemo_asr.models.ASRModel.restore_from(
        restore_path="models/parakeet-tdt-0.6b-v2.nemo", map_location=device
    )
    asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
    asr_model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
    print("Model loaded.")

    print("Starting transcription service...")
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
