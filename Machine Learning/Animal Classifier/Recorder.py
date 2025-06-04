import os
import soundfile as sf
import sounddevice as sd
from datetime import datetime

def record_audio_clips(folder_name="recordings", segment_length=5):
    """Record audio from microphone in fixed-length segments"""
    os.makedirs(folder_name, exist_ok=True)
    sample_rate = 44100  # Standard sample rate
    clip_counter = 1

    print(f"Recording {segment_length}-second clips to folder '{folder_name}'...")
    print("Press Ctrl+C to stop recording")

    try:
        while True:
            # Record one segment
            print(f"Recording clip {clip_counter}...")
            audio = sd.rec(int(segment_length * sample_rate),
                          samplerate=sample_rate,
                          channels=1)
            sd.wait()  # Wait until recording is finished

            # Save with timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{folder_name}/{folder_name}_{timestamp}_{clip_counter:03d}.wav"
            sf.write(filename, audio, sample_rate)
            
            clip_counter += 1

    except KeyboardInterrupt:
        print("\nRecording stopped")
        return clip_counter - 1  # Return number of clips recorded

# Configuration
FOLDER_NAME = "Dogs"  # Customize this for your session
SEGMENT_LENGTH = 6 # Seconds

if __name__ == "__main__":
    try:
        input("Press Enter to start recording...")
        num_clips = record_audio_clips(FOLDER_NAME, SEGMENT_LENGTH)
        print(f"Successfully recorded {num_clips} clips in '{FOLDER_NAME}' folder!")
        
    except Exception as e:
        print(f"Error: {str(e)}")