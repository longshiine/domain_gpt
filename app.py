### default
from llama_index import GPTListIndex, LLMPredictor, GPTSimpleVectorIndex, GPTSimpleKeywordTableIndex, PromptHelper, ServiceContext, QuestionAnswerPrompt
from langchain.chat_models import ChatOpenAI

## Graph
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.composability import ComposableGraph
from llama_index.composability.joint_qa_summary import QASummaryGraphBuilder

import gradio as gr
import sys
import os
from pathlib import Path
from utils.DataLoader import unstructured_loader, korean_pdf_loader, notion_loader, web_loader

## Caching ##
os.environ["OPENAI_API_KEY"] = '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'

max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 20
chunk_size_limit = 512

prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=512)
step_decompose_transform = StepDecomposeQueryTransform(llm_predictor, verbose=True)

# llm_predictor_gpt4 = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
# service_context_gpt4 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt4, chunk_size_limit=1024)

def construct_index(path):
    ### Global Index
    all_docs = []
    doc_sets = {}
    sections = range(4)
    for section in sections:
        documents = unstructured_loader(Path(f'data/book_section/law_{section}.txt'))
        for d in documents:
            d.extra_info = {"section": section}
        doc_sets[section] = documents
        all_docs.extend(documents)

    index_set = {}
    for section in sections:
        cur_index = GPTSimpleVectorIndex.from_documents(
            doc_sets[section],
            service_context=service_context,
        )
        cur_index.index_struct.summary = "Used to answer questions about the korean civil law"
        index_set[section] = cur_index
        cur_index.save_to_disk(f'{path}/index_{section}.json')
    
    summary_set = {}
    for section, title in zip(sections, ['머릿말', '총설', '채권', '물권']):
        summary_set[section] = f'Civil Law Section {section}. {title}'

    index_graph = ComposableGraph.from_indices(
        GPTListIndex,
        [index_set[y] for y in sections],
        [summary_set[y] for y in sections],
        service_context=service_context
    )
    index_graph.save_to_disk('{path}/index_list.json')

    global_index = GPTSimpleVectorIndex.from_documents(all_docs, service_context=service_context)
    global_index.save_to_disk(f'{path}/index_global.json')




def chatbot(query_str):
    ### Simple Index
    # index = GPTSimpleVectorIndex.load_from_disk('pdf_index.json')
    # response = index.query(query_str)

    ### Global Index
    graph_index = ComposableGraph.load_from_disk('experiments/graph/index_list.json')
    global_index = GPTSimpleVectorIndex.load_from_disk(f'experiments/graph/index_global.json')
    query_configs = [
        {
            "index_struct_type": "dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 3,
                # "include_summary": True
            }
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "response_mode": "tree_summarize",
            }
        },
    ]
    if '!' in query_str: 
        response = global_index.query(query_str)
    else:
        response = graph_index.query(query_str, query_configs=query_configs)

    return response.response

if __name__ == "__main__":
    iface = gr.Interface(fn=chatbot,
                        inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                        outputs="text",
                        title="Law GPT")

    index = construct_index(path='experiments/keyword')
    iface.launch(share=True)