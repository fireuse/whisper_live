import requests
import whisper

audio = whisper.load_audio("sample-4.mp3")
print(audio.shape)
resp = requests.post("http://127.0.0.1:8000/", data=audio.tobytes())
print(resp.text)
whisper.transcribe()