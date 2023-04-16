### default
from llama_index import GPTListIndex, LLMPredictor, GPTSimpleVectorIndex, GPTSimpleKeywordTableIndex, PromptHelper, ServiceContext, QuestionAnswerPrompt
from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI

## Graph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.composability import ComposableGraph
from llama_index.composability.joint_qa_summary import QASummaryGraphBuilder

import gradio as gr
import json
import sys
import os
from pathlib import Path
from utils.DataLoader import unstructured_loader, korean_pdf_loader, notion_loader, web_loader, directory_loader

## Caching ##
os.environ["OPENAI_API_KEY"] = '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'

max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 10
chunk_size_limit = 512

prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=512)
step_decompose_transform = StepDecomposeQueryTransform(llm_predictor, verbose=True)

# llm_predictor_gpt4 = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
# service_context_gpt4 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt4, chunk_size_limit=1024)

def construct_index(path, generate=False):
    sections = [1,2]
    part_index_sets = {}
    part_summary_sets = {}
    section_parts = []
    for section in sections:
        parts = range(1, len(os.listdir(Path(f'data/book_section/section{section}')))) # .DS_Store
        for part in parts: 
            if generate:
                documents = SimpleDirectoryReader(f'data/book_section/section{section}/part{part}').load_data(concatenate=False)
                part_index = GPTSimpleVectorIndex.from_documents(documents)
                # part_summary = part_index.query("What is a summary of this part?").response
                # print(f'part_summary: {part_summary}')
                
                section_part = f'{section}-{part}'
                section_parts.append(section_part)
                
                with open("data/book_section/parts.json", "r") as st_json:
                    st_python = json.load(st_json)
                part_summary = st_python[section_part]
                
                part_index_sets[section_part] = part_index
                part_summary_sets[section_part] = part_summary
            
                part_index.save_to_disk(f'{path}/section{section}_part{part}_index.json')
            else:
                with open("data/book_section/parts.json", "r") as st_json:
                    st_python = json.load(st_json)
                section_part = f'section{section}_part{part}'
                section_parts.append(section_part)
                
                part_index = GPTSimpleVectorIndex.load_from_disk(f'{path}/section{section}_part{part}_index.json')
                part_summary = st_python[section_part]
                print(f'part_summary: {part_summary}')
                
                part_index_sets[section_part] = part_index
                part_summary_sets[section_part] = part_summary

    index = ComposableGraph.from_indices(
        GPTListIndex,
        [part_index_sets[section_part] for section_part in section_parts],
        [part_summary_sets[section_part] for section_part in section_parts],
        service_context=service_context,
    )
    index.save_to_disk(f'{path}/index.json')
    
    index = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [part_index_sets[section_part] for section_part in section_parts],
        [part_summary_sets[section_part] for section_part in section_parts],
        service_context=service_context,
    )
    index.save_to_disk(f'{path}/index_keyword.json')


def chatbot(query_str):
    if '!' not in query_str:
        ## List Index
        index = ComposableGraph.load_from_disk('experiments/part/index.json')
        query_configs = [
            {
                "index_struct_type": "simple_dict",
                "query_mode": "default",
                "query_kwargs": {
                    "similarity_top_k": 1,
                    "include_summary": True
                },
            },
            {
                "index_struct_type": "list",
                "query_mode": "default",
                "query_kwargs": {
                    "response_mode": "tree_summarize",
                }
            },
        ]
    else:
        index = ComposableGraph.load_from_disk('experiments/part/index_keyword.json')
        decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)
        query_configs = [
            {
                "index_struct_type": "simple_dict",
                "query_mode": "default",
                "query_kwargs": {
                    "similarity_top_k": 3
                },
                # NOTE: set query transform for subindices
                "query_transform": decompose_transform
            },
            {
                "index_struct_type": "keyword_table",
                "query_mode": "simple",
                "query_kwargs": {
                    "response_mode": "tree_summarize",
                    "verbose": True
                },
            },
        ]
    response = index.query(query_str, query_configs=query_configs)
    print(response.source_nodes)

    return response.response

if __name__ == "__main__":
    iface = gr.Interface(fn=chatbot,
                        inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                        outputs="text",
                        title="Law GPT")

    # index = construct_index(path='experiments/part', generate=False)
    iface.launch(share=True)