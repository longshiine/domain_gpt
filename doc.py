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
import sys
import os
from pathlib import Path
from utils.DataLoader import unstructured_loader, korean_pdf_loader, notion_loader, web_loader, directory_loader

## Caching ##
from dotenv import load_dotenv
load_dotenv(verbose=True)
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 20
chunk_size_limit = 512

prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=512)
step_decompose_transform = StepDecomposeQueryTransform(llm_predictor, verbose=True)

# llm_predictor_gpt4 = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
# service_context_gpt4 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt4, chunk_size_limit=1024)

def construct_index(path, generate=False):
    sections = [1,2]
    doc_index_sets = {}
    doc_summary_sets = {}
    section_part_docs = []
    for section in sections:
        parts = range(1, len(os.listdir(Path(f'data/book_section/section{section}')))) # .DS_Store
        for part in parts:
            docs = range(1, len(os.listdir(Path(f'data/book_section/section{section}/part{part}'))))
            for doc in docs:
                if generate:
                    documents = SimpleDirectoryReader(input_files=[f'data/book_section/section{section}/part{part}/doc{doc}.txt']).load_data(concatenate=False)
                    doc_index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
                    doc_summary = doc_index.query("What is a summary of this paragraph?").response
                    print(f'doc_summary: {doc_summary}')
                    
                    section_part_doc = f'{section}-{part}-{doc}'
                    section_part_docs.append(section_part_doc)
                    doc_index_sets[section_part_doc] = doc_index
                    doc_summary_sets[section_part_doc] = doc_summary
                
                    doc_index.save_to_disk(f'{path}/section{section}_part{part}_doc{doc}_index.json')
                else:
                    doc_index = GPTSimpleVectorIndex.load_from_disk(f'{path}/section{section}_part{part}_doc{doc}_index.json')
                    doc_summary = doc_index.query("What is a summary of this paragraph?").response
                    print(f'doc_summary: {doc_summary}')
                    
                    section_part_doc = f'{section}-{part}-{doc}'
                    section_part_docs.append(section_part_doc)
                    doc_index_sets[section_part_doc] = doc_index
                    doc_summary_sets[section_part_doc] = doc_summary
                    

    index = ComposableGraph.from_indices(
        GPTListIndex,
        [doc_index_sets[section_part_doc] for section_part_doc in section_part_docs],
        [doc_summary_sets[section_part_doc] for section_part_doc in section_part_docs],
        service_context=service_context,
    )
    index.save_to_disk(f'{path}/index.json')

    index_keyword = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [doc_index_sets[section_part_doc] for section_part_doc in section_part_docs],
        [doc_summary_sets[section_part_doc] for section_part_doc in section_part_docs],
        service_context=service_context,
    )
    index_keyword.save_to_disk(f'{path}/index_keyword.json')

def chatbot(query_str):
    if '!' not in query_str:
        ### List Index
        index = ComposableGraph.load_from_disk('experiments/doc/index.json')
        query_configs = [
            {
                "index_struct_type": "dict",
                "query_mode": "default",
                "query_kwargs": {
                    "similarity_top_k": 1,
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
    else:
        index = ComposableGraph.load_from_disk('experiments/doc/index_keyword.json')
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
    print(response.source_nodes[0].source_text)
    return response.response

if __name__ == "__main__":
    iface = gr.Interface(fn=chatbot,
                        inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                        outputs="text",
                        title="Law GPT")

    # index = construct_index(path='experiments/doc', generate=False)
    iface.launch(share=True)