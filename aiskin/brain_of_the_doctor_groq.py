import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from groq import Groq
from PIL import Image


load_dotenv()


def encode_image_for_groq(filepath):
    image = Image.open(filepath)

    image.thumbnail((1024, 1024))

    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=75
    )

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def brain_of_the_doctor(
    patient_text,
    image_filepath=None,
    video_filepath=None
):

    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "Missing GROQ_API_KEY in .env or environment"
        )

    if not image_filepath:
        raise ValueError(
            "Please upload a skin image for analysis."
        )

    # Convert the uploaded skin image to base64
    image_data = encode_image_for_groq(image_filepath)

    # Prompt for the doctor response
    prompt = (
        "Act as a professional and empathetic dermatologist "
        "during a virtual consultation. "

        "Carefully examine the provided visual information and "
        "consider the patient's description. "

        "Describe the visible skin findings in clear, natural "
        "language and provide helpful general guidance. "

        "Speak directly to the patient as a doctor would. "

        "Do not mention AI, artificial intelligence, models, "
        "technical limitations, processing limitations, files, "
        "uploads, or how the information was provided. "

        "Do not say that you cannot see or analyze anything. "

        "Do not claim certainty or provide a definitive diagnosis. "

        "If the appearance suggests a possible condition, "
        "describe it as a possibility rather than a confirmed diagnosis. "

        "Give one or two practical recommendations and advise "
        "professional medical evaluation when appropriate. "

        "Keep the entire response to two or three sentences maximum. "

        "Use plain conversational language suitable for spoken audio. "

        "Do not use markdown, bullet points, asterisks, emojis, "
        "or special formatting. "

        f"\n\nPatient description: {patient_text}"
    )

    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_completion_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful skin care assistant. "
                    "Give general information and guidance, "
                    "not a definitive diagnosis."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,{image_data}"
                            ),
                        },
                    },
                ],
            },
        ],
    )

    return response.choices[0].message.content