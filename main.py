import nemo.collections.asr as nemo_asr
import torch

device = torch.device("cuda")

asr_model = nemo_asr.models.ASRModel.restore_from(restore_path="parakeet-tdt-0.6b-v2.nemo", map_location=device)
asr_model.change_attention_model("rel_pos_local_attn", [256,256])
asr_model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

file_path = "sample2.mkv"

output = asr_model.transcribe([file_path], timestamps=True)
# by default, timestamps are enabled for char, word and segment level
word_timestamps = output[0].timestamp['word'] # word level timestamps for first sample
segment_timestamps = output[0].timestamp['segment'] # segment level timestamps
char_timestamps = output[0].timestamp['char'] # char level timestamps

for stamp in segment_timestamps:
    print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")