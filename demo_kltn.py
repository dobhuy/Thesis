import tempfile
import streamlit as st
import librosa
import torch
import pandas as pd
from transformers import (
    AutoConfig,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoTokenizer, AutoModelForTokenClassification,
    AutoProcessor, AutoModelForSpeechSeq2Seq
)

# --- Configuration: model paths ---
ASR_MODELS = {
    "PhoWhisper": "Huydb/phowhisper-toxic",
    "Whisper": "Huydb/whisper-toxic",
    "Wav2Vec2_vlsp": "Huydb/wav2vec-vlsp-toxic",
    "Wav2Vec2_950h": "Huydb/wav2vec-950h-toxic",
}
TSD_MODELS = {
    "PhoBERT": "Huydb/PhoBERT-toxic",
    "ViSoBERT": "Huydb/ViSoBERT-toxic",
    "CafeBERT": "Huydb/CafeBERT-toxic",
    "XLMR": "Huydb/XLMR-toxic",
    "BERT": "Huydb/BERT-toxic",
    "DistilBERT": "Huydb/DistilBERT-toxic",
}

# --- Load ASR processors & models using config ---
asr_processors = {}
asr_models = {}
for name, path in ASR_MODELS.items():
    config = AutoConfig.from_pretrained(path)
    if config.model_type in ["wav2vec2_vlsp", "wav2vec2_950h"]:
        proc = Wav2Vec2Processor.from_pretrained(path)
        mod = Wav2Vec2ForCTC.from_pretrained(path, ignore_mismatched_sizes=True)
        # st.write(f"Loaded Wav2Vec2 ASR for {name}")
    elif config.model_type == "phowhisper":
        proc = AutoProcessor.from_pretrained(path)
        mod = AutoModelForSpeechSeq2Seq.from_pretrained(path)
        # st.write(f"Loaded Whisper ASR for {name}")
    elif config.model_type == "whisper":
        proc = WhisperProcessor.from_pretrained(path)
        mod = WhisperForConditionalGeneration.from_pretrained(path)
        mod.generation_config.forced_decoder_ids = None
        mod.generation_config.suppress_tokens = None
        
        mod.generation_config.language = "vi"
        mod.generation_config.task = "transcribe"
    else:
        st.error(f"Không hỗ trợ loại ASR model {config.model_type} tại {name}")
        continue
    asr_processors[name] = proc
    asr_models[name] = mod

# --- Load TSD tokenizers & models ---
tsd_tokenizers = {}
tsd_models = {}
for name, path in TSD_MODELS.items():
    tok = AutoTokenizer.from_pretrained(path)
    mod = AutoModelForTokenClassification.from_pretrained(path, num_labels=2)
    tsd_tokenizers[name] = tok
    tsd_models[name] = mod

# --- Streamlit UI ---
# CSS for animated background gradient and styled button
st.markdown(
    """
    <style>
    @keyframes bgfade {
        0% { background-color: white; }
        100% { background-color: white; }
        50% { background-color: #889ECE; }
    }
    html, body, .reportview-container, .main {
        height: 100% !important;
        margin: 0;
        padding: 0;
        animation: bgfade 10s ease infinite;
    }
    div.stButton > button:first-child {
        background-color: red !important;
        color: white !important;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Toxic Span Detection from Audio Transcript")

# Step 1: Upload audio
uploaded_audio = st.file_uploader("1. Upload a WAV audio file", type=["wav"])
if not uploaded_audio:
    st.info("Please upload a WAV audio file to begin.")
    st.stop()
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tfile:
    tfile.write(uploaded_audio.read())
    audio_path = tfile.name
st.success("Audio uploaded.")
st.audio(audio_path, format='audio/wav')

# Step 2: Upload ground-truth transcript
uploaded_txt = st.file_uploader("2. Upload ground-truth transcript (TXT)", type=["txt"])
if not uploaded_txt:
    st.info("Please upload the transcript TXT file.")
    st.stop()
transcript_text = uploaded_txt.read().decode('utf-8').strip()

# Step 3: Run ASR and TSD
if st.button("Transcript and Detect Toxic Spans Now"):
    # Section 1: ASR transcriptions table
    st.subheader("ASR Transcriptions")
    rows = [{"Model": "Groundtruth", "Transcript": transcript_text}]
    for name, proc in asr_processors.items():
        mod = asr_models[name]
        waveform, _ = librosa.load(audio_path, sr=16000)
        if isinstance(proc, Wav2Vec2Processor):
            input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values
            # Dự đoán logits
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            # Decode thành văn bản
            text = processor.decode(predicted_ids[0], skip_special_tokens=True)
        elif name == "PhoWhisper":
            input_features = proc(waveform, return_tensors="pt", sampling_rate=16000).input_features
            predicted_ids = mod.generate(input_features)
            text = proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        elif name == "Whisper":
            pipe = pipeline(
                    "automatic-speech-recognition",
                    model=mod,
                    tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor,
                    chunk_length_s=30,
                )
            result = pipe(audio_path, return_timestamps=True)
            text = result["text"]
        rows.append({"Model": name, "Transcript": text})
    st.table(pd.DataFrame(rows))

    # Section 2: Toxic span detection table
    st.subheader("Toxic Span Detection Results")
    html = [
        "<table style='width:100%; border-collapse: collapse;'>",
        "<tr><th style='border:1px solid #ddd; padding:8px;'>Model</th>",
        "<th style='border:1px solid #ddd; padding:8px;'>Result</th></tr>"
    ]
    def highlight_toxic_span(text, labels):
        out = ""
        for c, l in zip(text, labels):
            if l:
                out += f"<mark style='background-color:#ff6b6b'>{c}</mark>"
            else:
                out += c
        return out
    for name, tok in tsd_tokenizers.items():
        mod = tsd_models[name]
        enc = tok([transcript_text], is_split_into_words=True,
                  padding='max_length', truncation=True,
                  max_length=len(transcript_text), return_tensors="pt")
        with torch.no_grad(): logits = mod(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
        labels = logits.argmax(-1)[0].cpu().tolist()
        highlighted = highlight_toxic_span(transcript_text, labels)
        html.append(
            f"<tr><td style='border:1px solid #ddd; padding:8px;'>{name}</td>"
            f"<td style='border:1px solid #ddd; padding:8px;'>{highlighted}</td></tr>"
        )
    html.append("</table>")
    st.markdown('\n'.join(html), unsafe_allow_html=True)
