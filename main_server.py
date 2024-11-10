from typing import Dict

import numpy as np
#import whisper
import ray
from ray import serve
import stable_whisper
from starlette.requests import Request

from context_tracker import ContextTracker


# serve run main_server:my_model_deployment


@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 1})
class MyModelDeployment:
    def __init__(self):
        self.ctx_tracker = ContextTracker.remote()
        self.model = stable_whisper.load_model("turbo").cuda()
        stable_whisper.modify_model(self.model)
        print(self.model.device)

    async def __call__(self, request: Request) -> Dict:
        context = await self.ctx_tracker.get_tokens.remote()
        request = await request.body()
        carry = (await self.ctx_tracker.get_last.remote()).astype(np.float32)
        preprocess = np.concatenate([carry, np.frombuffer(request, np.float32)])
        #audio = stable_whisper.pad_or_trim(preprocess)
        result = self.model.transcribe(preprocess, prompt=context, language='pl', vad=True, only_voice_freq=True)
        result.remove_repetition(5)
        if len(result.to_dict()["segments"]) == 0 or len(result.to_dict()["segments"][-1]["words"]) == 0:
            return " "
        last_word = result.to_dict()["segments"][-1]["words"][-1]
        last_confidence = last_word["probability"]
        if last_confidence < 0.75 and len(result.to_dict()["segments"][-1]["words"]) > 1:
            word = result.to_dict()["segments"][-1]["words"][-2]
            last_timestamp = word["end"]
            token_remove = len(word["tokens"])
        else:
            last_timestamp = last_word["end"]
            token_remove = 0
        print(result.to_dict()["segments"][-1]["words"][-1])
        self.ctx_tracker.last_sample.remote(preprocess[int(last_timestamp*16000):])
        self.ctx_tracker.add_tokens.remote(context[:-token_remove])
        return result.text[:-len(last_word["word"]) if token_remove else len(result.text)]


ray.init()

my_model_deployment = MyModelDeployment.bind()
