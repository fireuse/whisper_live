import numpy as np
import requests
import soundcard as sc
import pyloudnorm as pyln


fs = 16000  # THIS
print(sc.all_microphones(exclude_monitors=False))
default_mic = sc.all_microphones(exclude_monitors=False)[1]#sc.default_microphone()#
print(default_mic)

data = None
meter = pyln.Meter(fs)
with default_mic.recorder(fs, channels=1, blocksize=fs*4) as rec:
    while True:
        data = rec.record(int(4 * fs)).flatten()
        if meter.integrated_loudness(data) > -np.inf:
            resp = requests.post("http://127.0.0.1:8000/", data=data.tobytes())
            print(resp.text)

