# Subtitle Translator App - Streamlit + OpenAI + Google STT + NLLB

from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
import tempfile
import datetime
import re
from moviepy.editor import VideoFileClip, AudioFileClip
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import requests
import json
import base64

# Load OpenAI API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Language options
LANGUAGES = [
    "Arabic", "Egyptian Arabic", "French", "Spanish",
    "German", "Japanese", "English", "Chinese", "Hindi"
]

# Transcribe a chunk

def transcribe_audio_whisper(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    return transcript.model_dump()

def transcribe_audio_google(file_path):
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    with open(file_path, "rb") as f:
        audio = speech.RecognitionAudio(content=f.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ar-EG",
        enable_word_time_offsets=True
    )
    response = client.recognize(config=config, audio=audio)
    results = []
    for result in response.results:
        for alt in result.alternatives:
            words = alt.words
            for word in words:
                results.append({
                    "start": word.start_time.total_seconds(),
                    "end": word.end_time.total_seconds(),
                    "text": word.word
                })
    return {"segments": results}

# Format timestamp
def format_timestamp(seconds):
    td = datetime.timedelta(seconds=round(seconds))
    return str(td) + ",000"

# Adjust gaps
def adjust_timestamps(segments):
    min_gap = 0.04
    adjusted = []
    for i, (start, end, text) in enumerate(segments):
        if i > 0:
            prev_end = adjusted[-1][1]
            if start - prev_end < min_gap:
                start = prev_end + min_gap
        adjusted.append((start, end, text))
    return adjusted

# Split video
def split_video(file_path, chunk_duration=600):
    try:
        video = VideoFileClip(file_path)
    except Exception as e:
        raise RuntimeError(f"Could not read video file. Error: {e}")

    duration = int(video.duration)
    chunks = []
    for start in range(0, duration, chunk_duration):
        end = min(start + chunk_duration, duration)
        if end - start <= 0:
            continue
        try:
            subclip = video.subclip(start, end - 0.1)
            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            subclip.audio.write_audiofile(audio_temp.name, logger=None)
            chunks.append((audio_temp.name, start))
        except Exception as e:
            st.error(f"Error processing chunk {start}-{end}: {e}")
    return chunks

# Enforce line limits on translation
def enforce_line_formatting(text, max_chars_per_line, max_lines):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= max_chars_per_line:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)

# Translate in batch
def batch_translate_lines(lines, language, chunk_size=30, lines_per_sub=2, chars_per_line=42):
    prompt_lang = "Egyptian Arabic dialect" if language == "Egyptian Arabic" else language
    translated_all = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i+chunk_size]
        numbered_lines = [f"{j+1}. {line}" for j, line in enumerate(chunk)]
        joined = "\n".join(numbered_lines)

        prompt = (
            f"Translate the following subtitle lines into {prompt_lang}. Rephrase if necessary to meet a maximum of {chars_per_line} characters per line and {lines_per_sub} lines per subtitle, while keeping the original meaning."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a subtitle translator and rephraser."},
                    {"role": "user", "content": f"{prompt}\n\n{joined}"}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            translated_lines = re.findall(r"\d+\.\s*(.+)", content)
        except:
            translated_lines = ["?"] * len(chunk)

        if len(translated_lines) < len(chunk):
            translated_lines += ["?"] * (len(chunk) - len(translated_lines))

        formatted = [enforce_line_formatting(t, chars_per_line, lines_per_sub) for t in translated_lines]
        translated_all.extend(formatted)
    return translated_all

# Parse SRT

def parse_srt(srt_content):
    srt_content = srt_content.replace('\ufeff', '').replace('\u202b', '')
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    parsed_entries = []
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            try:
                num = int(lines[0])
                start, end = lines[1].split(' --> ')
                text = ' '.join(lines[2:])
                parsed_entries.append((num, start, end, text))
            except:
                continue
    return parsed_entries

# Translate SRT

def translate_srt(srt_content, target_language, lines_per_sub=2, chars_per_line=42):
    entries = parse_srt(srt_content)
    original_lines = [entry[3] for entry in entries]
    translated_lines = batch_translate_lines(original_lines, target_language, lines_per_sub=lines_per_sub, chars_per_line=chars_per_line)
    translated = [f"{num}\n{start} --> {end}\n{translated}\n" for (num, start, end, _), translated in zip(entries, translated_lines)]
    return "\n".join(translated)

# === STREAMLIT APP ===
st.set_page_config(page_title="🎬 Subtitle App")
st.title("🎬 Subtitle App")
st.write("Choose an input method below. Upload a video for transcription, or an SRT file for translation.")

input_mode = st.radio("Select input type:", ["🎥 Video for Transcription", "📄 SRT for Translation"])

if input_mode == "🎥 Video for Transcription":
    video_file = st.file_uploader("Upload your video file (MP4, MOV, MPEG4)", type=["mp4", "mov", "mpeg4"])
    language_from = st.selectbox("Select language spoken in the video:", LANGUAGES)
    output_name = st.text_input("Enter desired name for output subtitle file:", value="transcription")

    if st.button("Transcribe Video"):
        if video_file and language_from:
            with st.spinner("Processing video and generating transcription..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    temp_video.write(video_file.read())
                    temp_video_path = temp_video.name

                chunks = split_video(temp_video_path)

                def process_chunk(chunk_path, offset):
                    if language_from == "Egyptian Arabic":
                        result = transcribe_audio_google(chunk_path)
                    else:
                        result = transcribe_audio_whisper(chunk_path)
                    return [(seg["start"] + offset, seg["end"] + offset, seg["text"]) for seg in result["segments"]]

                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(lambda c: process_chunk(c[0], c[1]), chunks))

                segments = [seg for group in results for seg in group]
                segments = adjust_timestamps(segments)

                srt_lines = []
                for i, (start, end, text) in enumerate(segments, 1):
                    srt_lines.append(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text.strip()}\n")

                srt_output = "\n".join(srt_lines)
                with open("transcription.srt", "w", encoding="utf-8") as f:
                    f.write(srt_output)

            st.success("✅ Transcription complete!")
            st.download_button("Download Transcription SRT", srt_output, file_name=f"{output_name}.srt")
        else:
            st.warning("Please upload a video and select language.")

elif input_mode == "📄 SRT for Translation":
    srt_file = st.file_uploader("Upload an SRT file", type=["srt"])
    target_language = st.selectbox("Translate subtitles into:", LANGUAGES)
    lines_per_sub = st.slider("Lines per subtitle", 1, 2, value=2)
    chars_per_line = st.slider("Max characters per line", 20, 60, value=42)
    output_name = st.text_input("Enter desired name for translated file:", value="translated")

    if st.button("Translate SRT"):
        if srt_file and target_language:
            srt_content = srt_file.read().decode("utf-8")
            with st.spinner("Translating SRT file..."):
                translated_srt = translate_srt(srt_content, target_language, lines_per_sub=lines_per_sub, chars_per_line=chars_per_line)
                st.success("✅ Translation complete!")
                st.download_button("Download Translated SRT", translated_srt, file_name=f"{output_name}.srt")
        else:
            st.warning("Please upload an SRT file and choose a target language.")
