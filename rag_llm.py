from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_DIR = '../models/'

MODEL_KWARGS = {
    'llama2_7b': {
        "hf_model": "TheBloke/Llama-2-7b-Chat-GGUF",
        "model_file": "llama-2-7b-chat.Q4_K_M.gguf",
        "model_type": 'llama'
    },
    'llama2_13b': {
        "hf_model": "TheBloke/Llama-2-7b-Chat-GGUF",
        "model_file": "llama-2-7b-chat.Q4_K_M.gguf",
        "model_type": 'llama'
    },
    'mistral7b_instruct': {
        "hf_model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "model_type": 'mistral'
    },
}

CONFIG = {'gpu_layers' : 20,
          'stream':True,
          'max_new_tokens': 2048,
          'temperature': 0.1,
          'repetition_penalty': 1.18,
          'context_length' : 4096}


def get_supported_models():
    supported_models = list(MODEL_KWARGS.keys())
    supported_models_string = ", ".join(supported_models)
    return supported_models_string


def load_llm(model : str = 'llama2_7b', model_dir : str = MODEL_DIR, config : dict = CONFIG, local : bool = True, verbose : bool = True, streaming : bool = True):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    if model in MODEL_KWARGS:
        model_config = MODEL_KWARGS[model]

        if local:
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

        llm = CTransformers(**model_kwargs,
                            config=config, 
                            callback_manager=callback_manager, 
                            verbose=True,
                            streaming=True)
        return llm
    else:
        supported_models = get_supported_models()
        raise ValueError(f"Model '{model}' not found in MODEL_KWARGS, only these models {supported_models} are supported")