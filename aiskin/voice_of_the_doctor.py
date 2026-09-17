# Step1: Create API keys

# Step2: Create Client and send request

from deepgram import DeepgramClient
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCTOR_AUDIO = BASE_DIR / "doctor_response.mp3"


def convert_text_to_doctor_audio(
    text,
    output_filepath=DEFAULT_DOCTOR_AUDIO
):

    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")

    deepgram = DeepgramClient(
        api_key=deepgram_api_key
    )

    # Flux Priya = Indian English female voice
    audio = deepgram.speak.v2.audio.generate(
        text=text,
        model="flux-priya-en",
        encoding="mp3",
    )

    output_filepath = Path(output_filepath)

    with output_filepath.open("wb") as file:
        for chunk in audio:
            file.write(chunk)

    return output_filepath


# ============================================================
# PLAY AUDIO
# ============================================================

import subprocess
import platform


def play_audio(audio_filepath):

    audio_filepath = str(audio_filepath)

    if platform.system() == "Darwin":

        subprocess.run(
            ["afplay", audio_filepath],
            check=False
        )

    elif platform.system() == "Windows":

        os.startfile(audio_filepath)

    else:

        subprocess.run(
            ["xdg-open", audio_filepath],
            check=False
        )


