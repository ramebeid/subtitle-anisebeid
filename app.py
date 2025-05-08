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

# Load OpenAI API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Language options
LANGUAGES = [
    "Arabic", "Egyptian Arabic", "French", "Spanish",
    "German", "Japanese", "English", "Chinese", "Hindi"
]

# Function to transcribe a chunk
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    return transcript.model_dump()

# Enforce subtitle formatting rules
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

# Translate multiple lines using GPT

def batch_translate_lines(lines, language, chunk_size=30, lines_per_sub=2, chars_per_line=42):
    if language == "Egyptian Arabic":
        prompt_lang = "Egyptian Arabic dialect"
    else:
        prompt_lang = language

    translated_all = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i+chunk_size]
        numbered_lines = [f"{j+1}. {line}" for j, line in enumerate(chunk)]
        joined = "\n".join(numbered_lines)

        prompt = (
            f"Translate the following subtitle lines into {prompt_lang}. Rephrase if necessary to meet a maximum of {chars_per_line} characters per line and {lines_per_sub} lines per subtitle, while keeping the original meaning."
        )
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
        formatted = [enforce_line_formatting(t, chars_per_line, lines_per_sub) for t in translated_lines]
        translated_all.extend(formatted if formatted else chunk)

    return translated_all

# Format timestamp
def format_timestamp(seconds):
    td = datetime.timedelta(seconds=round(seconds))
    return str(td) + ",000"

# Ensure safe subtitle gap
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
    video = VideoFileClip(file_path)
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

# UI
st.set_page_config(page_title="Subtitle Translator App")
st.title("\U0001F3AC Subtitle Translator")
st.write("Upload a video *or* subtitle file, choose options, and get translated subtitles.")

input_mode = st.radio("Choose input type:", ["Video", "SRT file"])
target_language = st.selectbox("Translate subtitles into:", LANGUAGES)
lines_per_sub = st.radio("Number of lines per subtitle:", [1, 2])
chars_per_line = st.number_input("Maximum characters per line:", min_value=20, max_value=80, value=42)
output_filename = st.text_input("Name your output file (without extension):", "subtitles")

if input_mode == "Video":
    st.markdown("**Note:** For Dropbox, use a link ending with `?dl=1`. For Google Drive, format as `https://drive.google.com/uc?export=download&id=FILE_ID`")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mpeg4"], key="video")
    video_url = st.text_input("Or paste a video URL (Dropbox or Google Drive):")
    if st.button("Generate Subtitles from Video"):
        if (video_file or video_url) and target_language:
            with st.spinner("Processing video and generating subtitles..."):
                if video_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                        temp_video.write(video_file.read())
                        temp_video_path = temp_video.name
                else:
                    response = requests.get(video_url)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                        temp_video.write(response.content)
                        temp_video_path = temp_video.name

                chunks = split_video(temp_video_path)

                def process_chunk(chunk_path, offset):
                    result = transcribe_audio(chunk_path)
                    return [(seg["start"] + offset, seg["end"] + offset, seg["text"]) for seg in result["segments"]]

                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(lambda c: process_chunk(c[0], c[1]), chunks))

                segments = [seg for group in results for seg in group if seg[2].strip()]
                segments = adjust_timestamps(segments)
                original_texts = [text.strip() for (_, _, text) in segments]
                translated_texts = batch_translate_lines(original_texts, target_language, lines_per_sub=lines_per_sub, chars_per_line=chars_per_line)

                txt_lines = []
                srt_lines = []

                for i, ((start_sec, end_sec, original), translated) in enumerate(zip(segments, translated_texts), 1):
                    start = format_timestamp(start_sec)
                    end = format_timestamp(end_sec)
                    txt_lines.append(f"{start} --> {end}\n{original}\n{translated}\n")
                    srt_lines.append(f"{i}\n{start} --> {end}\n{translated}\n")

                txt_output = "\n".join(txt_lines)
                srt_output = "\n".join(srt_lines)

                with open(f"{output_filename}.txt", "w", encoding="utf-8") as f:
                    f.write(txt_output)
                with open(f"{output_filename}.srt", "w", encoding="utf-8") as f:
                    f.write(srt_output)

            st.success("\u2705 Subtitles generated!")
            st.download_button("Download .txt", txt_output, file_name=f"{output_filename}.txt")
            st.download_button("Download .srt", srt_output, file_name=f"{output_filename}.srt")
        else:
            st.warning("Please upload or link a video and select a language.")

elif input_mode == "SRT file":
    srt_file = st.file_uploader("Upload an SRT file", type=["srt"], key="srt")
    if st.button("Translate SRT File"):
        if srt_file and target_language:
            srt_content = srt_file.read().decode("utf-8-sig")
            with st.spinner("Translating SRT file..."):
                translated_srt = translate_srt(srt_content, target_language, lines_per_sub=lines_per_sub, chars_per_line=chars_per_line)
                st.success("\u2705 Translation complete!")
                st.download_button("Download Translated .srt", translated_srt, file_name=f"{output_filename}.srt")
        else:
            st.warning("Please upload an SRT file and select a language.")
