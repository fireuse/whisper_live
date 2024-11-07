import requests
import soundcard as sc

fs = 16000  # THIS
print(sc.all_microphones())
default_mic = sc.all_microphones(exclude_monitors=False)[1]#sc.default_microphone()
print(default_mic)

data = None
with default_mic.recorder(fs, channels=1, blocksize=fs*5) as rec:
    while True:
        data = rec.record(int(3 * fs)).flatten()
        resp = requests.post("http://127.0.0.1:8000/", data=data.tobytes())
        print(resp.text)

