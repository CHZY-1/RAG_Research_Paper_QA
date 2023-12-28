from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from constants import load_environment
import os

"""
ctransformers CONFIG

Parameter             | Type    | Description                                              | Default
----------------------|---------|----------------------------------------------------------|--------
top_k                 | int     | The top-k value to use for sampling.                     | 40
top_p                 | float   | The top-p value to use for sampling.                     | 0.95
temperature           | float   | The temperature to use for sampling.                     | 0.8
repetition_penalty    | float   | The repetition penalty to use for sampling.              | 1.1
last_n_tokens         | int     | The number of last tokens to use for repetition penalty. | 64
seed                  | int     | The seed value to use for sampling tokens.               | -1
max_new_tokens        | int     | The maximum number of new tokens to generate.            | 256
stop                  | List[str]| A list of sequences to stop generation when encountered.| None
stream                | bool    | Whether to stream the generated text.                    | False
reset                 | bool    | Whether to reset the model state before generating text. | True
batch_size            | int     | The batch size to use for evaluating tokens in a single prompt. | 8
threads               | int     | The number of threads to use for evaluating tokens.      | -1
context_length        | int     | The maximum context length to use.                       | -1
gpu_layers            | int     | The number of layers to run on GPU.                      | 0

"""

load_environment()
MODELS_DIR = os.environ.get("MODELS_DIR")

MODEL_KWARGS = {
    'llama2_7b': {
        "hf_model": "TheBloke/Llama-2-7b-Chat-GGUF",
        "model_file": "llama-2-7b-chat.Q4_K_M.gguf",
        "model_type": 'llama',
        "local": True
    },
    'llama2_13b': {
        "hf_model": "TheBloke/Llama-2-13b-Chat-GGUF",
        "model_file": "llama-2-13b-chat.Q4_K_M.gguf",
        "model_type": 'llama',
        "local": True
    },
    'llama2_7b_base': {
        "hf_model": "TheBloke/Llama-2-7B-GGUF",
        "model_file": "llama-2-7b.Q4_K_M.gguf",
        "model_type": 'llama',
        "local": True
    },
    'mistral7b_instruct': {
        "hf_model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "model_type": 'mistral',
        "local": True
    },
    'yarnmistral7b_64k' : {
        "hf_model" : "TheBloke/Yarn-Mistral-7B-64k-GGUF",
        "model_file": "yarn-mistral-7b-64k.Q4_K_M.gguf",
        "model_type": 'mistral',
        "local": False
    },
    'yarnmistral7b_128k' : {
        "hf_model": "TheBloke/Yarn-Mistral-7B-128k-GGUF",
        "model_file": "yarn-mistral-7b-128k.Q4_K_M.gguf",
        "model_type": 'mistral',
        "local": True
    },
    'codellama_7b': {
        "hf_model": "TheBloke/CodeLlama-7B-Python-GGML",
        "model_file": "codellama-7b-python.ggmlv3.Q4_K_M.bin",
        "model_type": 'llama',
        "local": False
    },
}

CONFIG = {'gpu_layers' : 25,
          'stream':True,
          'max_new_tokens': 2048,
          'temperature': 0.1,
          'repetition_penalty': 1.18,
          'context_length' : 4096}


def get_supported_models():
    supported_models = list(MODEL_KWARGS.keys())
    supported_models_string = ", ".join(supported_models)
    return supported_models_string


def load_llm(model : str = 'llama2_7b', model_dir : str = MODELS_DIR, config : dict = CONFIG, local : bool = True, verbose : bool = True, streaming : bool = True):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    if model in MODEL_KWARGS:
        model_config = MODEL_KWARGS[model]

        if local and model_config.get("local", False):
            # Load from local with a single parameter
            model_kwargs = {
                "model": model_dir  + model_config.get("model_file", None),
                "model_type": model_config.get("model_type", None),
            }
        else:
            model_kwargs = {
                "model": model_config.get("hf_model", None),
                "model_file": model_config.get("model_file", None),
                "model_type": model_config.get("model_type", None),
            }

        if not model_config.get("local", False) and local:
            raise ValueError(f"Local setup is not existed for model '{model}'. Loading from Hugging Face instead.")

        llm = CTransformers(**model_kwargs,
                            config=config, 
                            callback_manager=callback_manager, 
                            verbose=True,
                            streaming=True)
        return llm
    else:
        supported_models = get_supported_models()
        raise ValueError(f"Model '{model}' not found in MODEL_KWARGS, only these models {supported_models} are available.")
    
if __name__ == "__main__":
    print(get_supported_models())