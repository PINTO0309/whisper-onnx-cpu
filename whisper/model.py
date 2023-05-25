import io
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import requests
import psutil
import onnx
import onnxruntime as ort
from whisper.decoding import detect_language as detect_language_function, decode as decode_function
from whisper.utils import onnx_dtype_to_np_dtype_convert


_MODELS = {
    "tiny.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.en.pt",
    "tiny": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.pt",
    "base.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.en.pt",
    "base": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.pt",
    "small.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.en.pt",
    "small": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.pt",
    "medium.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.en.pt",
    "medium": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.pt",
}

def model_download(name: str, onnx_file_save_path: str='.') -> onnx.ModelProto:
    onnx_file_path = f'{onnx_file_save_path}/{name}_11.onnx'
    onnx_serialized_graph = None
    if not os.path.exists(onnx_file_path):
        url = f'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/{name}_11.onnx'
        onnx_serialized_graph = requests.get(url).content
        with io.BytesIO(onnx_serialized_graph) as f:
            onnx_graph: onnx.ModelProto = onnx.load(f)
            onnx.save(onnx_graph, f'{onnx_file_save_path}/{name}_11.onnx')
    else:
        onnx_graph: onnx.ModelProto = onnx.load(onnx_file_path)
        onnx_serialized_graph = onnx._serialize(onnx_graph)
    return onnx_serialized_graph

def load_model(name: str):
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if name == "tiny":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "tiny.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "base":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "base.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "small":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "small.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "medium":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    elif name == "medium.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    else:
        raise ValueError(f"model type {name} not supported")

    dims = ModelDimensions(**dims_config)
    model = Whisper(dims=dims, model_name=name)
    return model

def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class OnnxAudioEncoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_encoder'),
                sess_options=sess_options,
                providers=[
                    'CPUExecutionProvider'
                ],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        mel: np.ndarray
    ) -> np.ndarray:
        result: np.ndarray = \
            self.sess.run(
                output_names=[
                    "output",
                ],
                input_feed={
                    "mel": mel.astype(self.inputs["mel"]),
                }
            )[0]
        return result


class OnnxTextDecoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_decoder'),
                sess_options=sess_options,
                providers=[
                    'CPUExecutionProvider'
                ],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        x: np.ndarray,
        xa: np.ndarray,
        kv_cache: np.ndarray,
        offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = \
            self.sess.run(
                output_names=[
                    "logits",
                    "output_kv_cache",
                    "cross_attention_qks",
                ],
                input_feed={
                    "tokens": x.astype(self.inputs["tokens"]),
                    "audio_features": xa.astype(self.inputs["audio_features"]),
                    "kv_cache": kv_cache.astype(self.inputs["kv_cache"]),
                    "offset": np.array([offset], dtype=self.inputs["offset"]),
                }
            )
        logits: np.ndarray = outputs[0]
        output_kv_cache: np.ndarray = outputs[1]
        cross_attention_qks: np.ndarray = outputs[2]
        return logits.astype(np.float32), output_kv_cache.astype(np.float32)


class Whisper():
    def __init__(
        self,
        dims: ModelDimensions,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.dims = dims
        self.encoder = OnnxAudioEncoder(model=model_name)
        self.decoder = OnnxTextDecoder(model=model_name)

    def embed_audio(
        self,
        mel: np.ndarray,
    ):
        return self.encoder(mel)

    def logits(
        self,
        tokens: np.ndarray,
        audio_features: np.ndarray,
    ):
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, audio_features, kv_cache=kv_cache, offset=0)
        return output

    def __call__(
        self,
        mel: np.ndarray,
        tokens: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, self.encoder(mel), kv_cache=kv_cache, offset=0)
        return output

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def new_kv_cache(
        self,
        n_group: int,
        length: int,
    ):
        if self.model_name == "tiny.en" or self.model_name == "tiny":
            size = [8, n_group, length, 384]
        elif self.model_name == "base.en" or self.model_name == "base":
            size = [12, n_group, length, 512]
        elif self.model_name == "small.en" or self.model_name == "small":
            size = [24, n_group, length, 768]
        elif self.model_name == "medium.en" or self.model_name == "medium":
            size = [48, n_group, length, 1024]
        else:
            raise ValueError(f"Unsupported model type: {self.type}")
        return np.zeros(size, dtype=np.float32)

    detect_language = detect_language_function
    # transcribe = transcribe_function
    decode = decode_function
