import panel as pn
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from rag import get_rag_chain, get_rag_chain_async

pn.extension()

# We cache the chains and responses to speed up things
# llm_chains = pn.state.cache["llm_chains"] = pn.state.cache.get("llm_chains", {})
# responses = pn.state.cache["responses"] = pn.state.cache.get("responses", {})

def _get_llm_chain():
    llm_chain = get_rag_chain_async()

    return llm_chain

async def _get_response(contents: str, llm_chain):
    response = llm_chain({"query": contents})

    print(response)
    chunks = []

    for chunk in response["source_documents"][::-1]:
        name = f"Chunk {chunk.metadata['page']}"
        content = chunk.page_content
        chunks.insert(0, (name, content))
    return response, chunks


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    try:
        loading_spinner = pn.indicators.LoadingSpinner(value=True, size=70, name='Loading...')
        instance.append(loading_spinner)

        llm_chain = await _get_llm_chain()

        message = None
        # response = await _get_response(contents, llm_chain)

        response, documents = await _get_response(contents, llm_chain)

        instance.remove(loading_spinner)

        column = pn.Column(sizing_mode="stretch_width")

        pages_layout = pn.Accordion(*documents, sizing_mode="stretch_width", max_width=1000)
        column.append(pages_layout)

        # for chunk in response['result']:
        #     panel = pn.panel(chunk, width_policy="max", sizing_mode="stretch_width")
        #     column.append(panel)

        # instance.send({"user": "Model", "object": column}, respond=False)

        yield {"user": "LangChain(Retriever)", "object": column}


        for chunk in response["result"]:
            message = instance.stream(chunk, user="Assistant", message=message)

    except Exception as e:
        instance.send({"user": "System", "object": f"Error: {e}. Please try again."}, respond=False)



chat_interface = pn.chat.ChatInterface(
    callback=callback,
    placeholder_threshold=0.1,
    sizing_mode="stretch_width")

chat_interface.send(
    "Send a message to get a reply from Research paper QA Chatbot",
    user="System",
    respond=False,
)
chat_interface.servable()

# panel serve ui.py