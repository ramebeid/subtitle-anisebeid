import os
os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.expanduser("~/bin/ffmpeg")

import moviepy.editor
print("✅ moviepy.editor imported successfully with ffmpeg set.")

