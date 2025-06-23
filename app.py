import streamlit as st
import os
import tempfile
import subprocess
import json
import base64
from moviepy.editor import VideoFileClip
from google.cloud import speech
from pathlib import Path
import whisper

# UI
st.title("🎬 AI Subtitle Generator")

input_mode = st.selectbox("Choose your input type:", ["🎥 Video for Transcription", "📄 SRT for Translation"])

# Ask user to choose transcription type
transcription_type = st.radio(
    "Is the video spoken in a single language or multiple languages?",
    ["Single Language", "Multilingual"],
    horizontal=True
)

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })
    return segments

def transcribe_audio_google(audio_path):
    import streamlit as st
    import json
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_json:
        json.dump(dict(st.secrets["GOOGLE_CREDENTIALS"]), temp_json)
        temp_json_path = temp_json.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_json_path

    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        audio_channel_count=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        language_code="en-US",
        alternative_language_codes=["fr-FR", "ar-EG", "de-DE", "es-ES", "hi-IN", "zh"],
        model="latest_long"
    )

    response = client.recognize(config=config, audio=audio)

    segments = []
    for result in response.results:
        alt = result.alternatives[0]
        start = alt.words[0].start_time.total_seconds() if alt.words else 0
        end = alt.words[-1].end_time.total_seconds() if alt.words else start + 2
        segments.append({"start": start, "end": end, "text": alt.transcript})

    return segments

def format_srt(segments):
    def srt_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt = ""
    for i, seg in enumerate(segments):
        start = srt_timestamp(seg["start"])
        end = srt_timestamp(seg["end"])
        text = seg["text"]
        srt += f"{i+1}\n{start} --> {end}\n{text}\n\n"
    return srt

if input_mode == "🎥 Video for Transcription":
    video_file = st.file_uploader("Upload your video file (MP4, MOV, MPEG4)", type=["mp4", "mov", "mpeg4"])
    output_name = st.text_input("Enter desired name for output subtitle file:", value="transcription")

    if st.button("Transcribe Video"):
        if video_file:
            with st.spinner("Processing video and generating transcription..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    temp_video.write(video_file.read())
                    temp_video_path = temp_video.name

                audio_output_path = temp_video_path.replace(".mp4", ".mp3")
                video = VideoFileClip(temp_video_path)
                video.audio.write_audiofile(audio_output_path)

                if transcription_type == "Single Language":
                    segments = transcribe_audio_whisper(audio_output_path)
                else:
                    segments = transcribe_audio_google(audio_output_path)

                srt_output = format_srt(segments)
                srt_file_path = f"{output_name}.srt"
                with open(srt_file_path, "w", encoding="utf-8") as f:
                    f.write(srt_output)

                st.success("✅ Transcription complete!")
                st.download_button("📥 Download SRT File", srt_output, file_name=srt_file_path, mime="text/plain")

elif input_mode == "📄 SRT for Translation":
    uploaded_srt = st.file_uploader("Upload your SRT file", type=["srt"])
    target_lang = st.text_input("Translate to which language? (e.g. 'fr', 'ar', 'de')", value="fr")

    if st.button("Translate SRT") and uploaded_srt:
        srt_text = uploaded_srt.read().decode("utf-8")
        import openai
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        prompt = f"Translate the following subtitles to {target_lang}, keeping timestamps unchanged:\n\n" + srt_text
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        translated_srt = response.choices[0].message.content
        st.success("✅ Translation complete!")
        st.download_button("📥 Download Translated SRT", translated_srt, file_name=f"translated_{target_lang}.srt", mime="text/plain")
