"""Voice recorder module: microphone capture and Whisper API transcription.

Two-Enter UX:
    1. User presses Enter → recording starts (sounddevice InputStream opens).
    2. User presses Enter again → recording stops (InputStream closes).

Audio is captured via a queue-based sounddevice InputStream callback pattern.
The queue is unbounded so it never drops frames while `input()` blocks the main
thread. After the stream closes the queue is fully drained before WAV encoding.

The resulting BytesIO buffer has `.name = "recording.wav"` set — this is
required by the OpenAI Whisper API for audio format detection.
"""

import io
import queue
import sys

from openai import OpenAI

from src.config import Settings


class VoiceInputError(Exception):
    """Raised when voice recording or transcription fails."""
    pass


def record_until_enter(sample_rate: int = 16000) -> io.BytesIO:
    """Record from the default microphone until user presses Enter.

    Returns a BytesIO WAV buffer with `.name = "recording.wav"` set.
    The buffer is ready to pass directly to the OpenAI Whisper API.

    Args:
        sample_rate: Audio sample rate in Hz. Whisper prefers 16000 Hz mono.

    Returns:
        io.BytesIO: WAV-encoded audio buffer with .name attribute set.

    Raises:
        VoiceInputError: If no audio was captured (e.g. microphone not connected).
    """
    q: queue.Queue = queue.Queue()  # Unbounded — do NOT set maxsize

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    import numpy as np  # noqa: PLC0415
    import sounddevice as sd  # noqa: PLC0415
    from scipy.io import wavfile  # noqa: PLC0415

    print("Press Enter to start recording...")
    input()  # First Enter starts recording
    print("Recording... press Enter to stop.")

    chunks = []
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        input()  # Second Enter stops recording (InputStream closes on exit)

    # Drain queue AFTER stream closes to avoid missing late callbacks
    while not q.empty():
        chunks.append(q.get())

    if not chunks:
        raise VoiceInputError(
            "No audio captured — microphone may not be connected or accessible."
        )

    audio = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio)
    buf.seek(0)
    buf.name = "recording.wav"  # CRITICAL: OpenAI Whisper uses this for format detection
    return buf


def transcribe_audio(wav_buffer: io.BytesIO, settings: Settings | None = None) -> str:
    """Transcribe WAV bytes via OpenAI Whisper API.

    Args:
        wav_buffer: BytesIO WAV buffer. Should have `.name = "recording.wav"`.
            If `.name` is missing it will be set automatically.
        settings: Settings instance. If None, a new Settings() is instantiated.

    Returns:
        str: Raw transcribed text from Whisper.
    """
    if settings is None:
        settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Ensure buffer is at start and has .name attribute
    wav_buffer.seek(0)
    if not hasattr(wav_buffer, "name"):
        wav_buffer.name = "recording.wav"

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_buffer,
        response_format="text",  # Returns plain str, not object
    )
    return transcript if isinstance(transcript, str) else str(transcript)


def validate_transcription(text: str) -> str:
    """Validate that Whisper returned usable text.

    Rejects empty transcriptions and clearly garbled output (where less than
    30% of characters are alphabetic or spaces).

    Args:
        text: Raw transcription string from Whisper.

    Returns:
        str: Stripped text if valid.

    Raises:
        VoiceInputError: If text is empty or appears garbled.
    """
    stripped = text.strip()
    if not stripped:
        raise VoiceInputError(
            "Couldn't understand audio — please type your query instead."
        )
    # Detect clearly garbled: all non-alphabetic/space characters
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / len(stripped)
    if alpha_ratio < 0.3:
        raise VoiceInputError(
            "Couldn't understand audio — please type your query instead."
        )
    return stripped


def record_and_transcribe(settings: Settings | None = None) -> str:
    """Record voice input, transcribe via Whisper, and validate result.

    Convenience function combining record_until_enter + transcribe_audio +
    validate_transcription. Used by the LangGraph node in Plan 05.

    Args:
        settings: Settings instance. If None, a new Settings() is instantiated.

    Returns:
        str: Validated transcribed text ready for use as a text query.

    Raises:
        VoiceInputError: If recording captured no audio or transcription fails
            validation (empty or garbled output).
    """
    wav_buffer = record_until_enter()
    raw_text = transcribe_audio(wav_buffer, settings=settings)
    return validate_transcription(raw_text)
