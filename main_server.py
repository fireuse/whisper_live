from typing import Dict

import numpy as np
import whisper
import ray
from ray import serve
from starlette.requests import Request


@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 1})
class MyModelDeployment:
    def __init__(self):
        self.model = whisper.load_model("small").cuda()
        print(self.model.device)

    async def __call__(self, request: Request) -> Dict:

        request = await request.body()
        audio = np.frombuffer(request, np.float32)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        options = whisper.DecodingOptions()
        result = self.model.decode(mel, options)
        return result.text


ray.init()

my_model_deployment = MyModelDeployment.bind()

