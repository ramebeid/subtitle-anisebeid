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
    return transcript.model_dump()  # Convert Pydantic object to dictionary

# Function to translate multiple lines using GPT-3.5 in one call
# with batching to avoid context length limits
def batch_translate_lines(lines, language, chunk_size=30):
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
            f"Translate the following subtitle lines into {prompt_lang}.\n"
            "Return only the translated lines in the same order and numbering:\n\n"
            f"{joined}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a subtitle translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        translated_lines = re.findall(r"\d+\.\s*(.+)", content)
        if translated_lines:
            translated_all.extend(translated_lines)
        else:
            translated_all.extend(chunk)  # fallback if parsing fails

    return translated_all

# Format SRT timestamps
def format_timestamp(seconds):
    return str(datetime.timedelta(seconds=int(seconds))) + ",000"

# Split video into audio chunks for transcription
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

# Parse SRT file content
def parse_srt(srt_content):
    srt_content = srt_content.replace('\ufeff', '')
    srt_content = srt_content.replace('\u202b', '')
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    parsed_entries = []
    for block in blocks:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if len(lines) >= 3:
            try:
                num = int(lines[0])
                times = lines[1]
                start, end = times.split(' --> ')
                text = ' '.join(lines[2:])
                parsed_entries.append((num, start, end, text))
            except Exception:
                continue
    return parsed_entries

# Translate SRT segments
def translate_srt(srt_content, target_language):
    entries = parse_srt(srt_content)
    original_lines = [entry[3] for entry in entries]
    translated_lines = batch_translate_lines(original_lines, target_language)
    translated = []
    for (num, start, end, _), translated_text in zip(entries, translated_lines):
        translated.append(f"{num}\n{start} --> {end}\n{translated_text}\n")
    return "\n".join(translated)

# Streamlit app UI
st.set_page_config(page_title="Subtitle Translator App")
st.title("\U0001F3AC Subtitle Translator")
st.write("Upload a video *or* subtitle file, choose a language, and get translated subtitles.")

input_mode = st.radio("Choose input type:", ["Video", "SRT file"])
target_language = st.selectbox("Translate subtitles into:", LANGUAGES)

if input_mode == "Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mpeg4"], key="video")
    if st.button("Generate Subtitles from Video"):
        if video_file and target_language:
            with st.spinner("Processing video and generating subtitles..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    temp_video.write(video_file.read())
                    temp_video_path = temp_video.name

                chunks = split_video(temp_video_path)

                def process_chunk(chunk_path, offset):
                    result = transcribe_audio(chunk_path)
                    return [(seg["start"] + offset, seg["end"] + offset, seg["text"]) for seg in result["segments"]]

                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(lambda c: process_chunk(c[0], c[1]), chunks))

                segments = [seg for group in results for seg in group if seg[2].strip()]
                original_texts = [text.strip() for (_, _, text) in segments]
                translated_texts = batch_translate_lines(original_texts, target_language)

                txt_lines = []
                srt_lines = []

                for i, ((start_sec, end_sec, original), translated) in enumerate(zip(segments, translated_texts), 1):
                    start = format_timestamp(start_sec)
                    end = format_timestamp(end_sec)
                    txt_lines.append(f"{start} --> {end}\n{original}\n{translated}\n")
                    srt_lines.append(f"{i}\n{start} --> {end}\n{translated}\n")

                txt_output = "\n".join(txt_lines)
                srt_output = "\n".join(srt_lines)

                with open("subtitles.txt", "w", encoding="utf-8") as f:
                    f.write(txt_output)
                with open("subtitles.srt", "w", encoding="utf-8") as f:
                    f.write(srt_output)

            st.success("\u2705 Subtitles generated!")
            st.download_button("Download .txt", txt_output, file_name="subtitles.txt")
            st.download_button("Download .srt", srt_output, file_name="subtitles.srt")
        else:
            st.warning("Please upload a video and select a language.")

elif input_mode == "SRT file":
    srt_file = st.file_uploader("Upload an SRT file", type=["srt"], key="srt")
    if st.button("Translate SRT File"):
        if srt_file and target_language:
            srt_content = srt_file.read().decode("utf-8-sig")
            with st.spinner("Translating SRT file..."):
                translated_srt = translate_srt(srt_content, target_language)
                st.success("\u2705 Translation complete!")
                st.download_button("Download Translated .srt", translated_srt, file_name="translated.srt")
        else:
            st.warning("Please upload an SRT file and select a language.")
