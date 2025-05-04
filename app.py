import os
os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.expanduser("~/bin/ffmpeg")
import openai
import streamlit as st
import tempfile
import datetime



from moviepy.editor import AudioFileClip

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Language options
LANGUAGES = ["Arabic", "French", "Spanish", "German", "Japanese", "English"]

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
    return transcript

# Function to translate text using GPT-3.5
def translate_line(text, language):
    messages = [
        {"role": "system", "content": f"Translate this sentence into {language}. Return only the translation."},
        {"role": "user", "content": text}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Function to format SRT timestamps
def format_timestamp(seconds):
    return str(datetime.timedelta(seconds=int(seconds))) + ",000"

# Streamlit app UI
st.set_page_config(page_title="Subtitle Translator App")
st.title("🎬 Subtitle Translator")
st.write("Upload a video file, choose a language, and get translated subtitles.")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])
target_language = st.selectbox("Translate subtitles into:", LANGUAGES)

if st.button("Generate Subtitles"):
    if video_file and target_language:
        with st.spinner("Processing video and generating subtitles..."):
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(video_file.read())
                temp_video_path = temp_video.name

            # Transcribe
            transcript = transcribe_audio(temp_video_path)
            segments = transcript["segments"]

            # Translate and build output strings
            txt_lines = []
            srt_lines = []

            for i, segment in enumerate(segments, 1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                original = segment["text"].strip()
                translation = translate_line(original, target_language)

                # TXT
                txt_lines.append(f"{start} --> {end}\n{original}\n{translation}\n")

                # SRT
                srt_lines.append(f"{i}\n{start} --> {end}\n{translation}\n")

            # Save files
            txt_output = "\n".join(txt_lines)
            srt_output = "\n".join(srt_lines)

            with open("subtitles.txt", "w", encoding="utf-8") as f:
                f.write(txt_output)
            with open("subtitles.srt", "w", encoding="utf-8") as f:
                f.write(srt_output)

        st.success("✅ Subtitles generated!")
        st.download_button("Download .txt", txt_output, file_name="subtitles.txt")
        st.download_button("Download .srt", srt_output, file_name="subtitles.srt")
    else:
        st.warning("Please upload a video and select a language.")
