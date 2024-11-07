from typing import Dict

import numpy as np
import whisper
import ray
from ray import serve
from starlette.requests import Request

from context_tracker import ContextTracker


# serve run main_server:my_model_deployment


@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 1})
class MyModelDeployment:
    def __init__(self):
        self.ctx_tracker = ContextTracker.remote()
        self.model = whisper.load_model("turbo").cuda()
        print(self.model.device)

    async def __call__(self, request: Request) -> Dict:
        context = await self.ctx_tracker.get_tokens.remote()
        request = await request.body()
        audio = np.frombuffer(request, np.float32)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.model.device)
        options = whisper.DecodingOptions(prompt=context, prefix=context[-20:])
        result = self.model.decode(mel, options)
        if result.no_speech_prob > 0.4:
            return "[EMPTY]"
        self.ctx_tracker.add_tokens.remote(context)
        return result.text


ray.init()

my_model_deployment = MyModelDeployment.bind()
