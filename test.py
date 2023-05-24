import io
import onnx
import requests
import onnxruntime as ort

def model_download(name: str, onnx_file_save_path: str='.') -> bytes:
    URL = f'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/{name}_11.onnx'
    onnx_serialized_graph = requests.get(URL).content
    with io.BytesIO(onnx_serialized_graph) as f:
        onnx_graph: onnx.ModelProto = onnx.load(f)
        onnx.save(onnx_graph, f'{onnx_file_save_path}/{name}_11.onnx')
    return onnx_serialized_graph

if __name__ == '__main__':
    onnx_serialized_graph = model_download('tiny_encoder')
    ort_sess_encoder = \
        ort.InferenceSession(
            path_or_bytes=onnx_serialized_graph,
            providers=['CUDAExecutionProvider'],
        )
    onnx_serialized_graph = model_download('tiny_decoder')
    ort_sess_decoder = \
        ort.InferenceSession(
            path_or_bytes=onnx_serialized_graph,
            providers=['CUDAExecutionProvider'],
        )
    a=0