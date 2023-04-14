### default
from llama_index import GPTListIndex, LLMPredictor, GPTSimpleVectorIndex, GPTSimpleKeywordTableIndex, PromptHelper, ServiceContext, QuestionAnswerPrompt
from langchain.chat_models import ChatOpenAI

## Graph
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.composability import ComposableGraph
from llama_index.composability.joint_qa_summary import QASummaryGraphBuilder

import gradio as gr
import sys
import os
from pathlib import Path
from utils.DataLoader import directory_loader, korean_pdf_loader, notion_loader, web_loader

## Caching ##
os.environ["OPENAI_API_KEY"] = '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'

max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 20
chunk_size_limit = 600

prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=600)

# llm_predictor_gpt4 = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
# service_context_gpt4 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt4, chunk_size_limit=1024)

def construct_index():
    ### Knowledge Index Generate
    # documents = korean_pdf_loader(Path('data/book_text.pdf'))
    # index = GPTKnowledgeGraphIndex.from_documents(
    #     book_documents, 
    #     max_triplets_per_chunk=2,
    #     service_context=service_context,
    #     include_embeddings=True
    # )
    # index.save_to_disk('book_index.json')

    
    ### Global Index Generate
    # all_docs = []
    # book_document = korean_pdf_loader(Path('data/book_text.pdf'))
    # law_document = korean_pdf_loader(Path('data/law2.pdf'))
    # notion_document = notion_loader(['1f5dd8ba19864270a8fbdb0b34b55abe'])
    # all_docs.extend(book_document)
    # all_docs.extend(law_document)
    # all_docs.extend(notion_document)
    
    # global_index = GPTSimpleVectorIndex.from_documents(all_docs, service_context=service_context)
    # global_index.save_to_disk(f'index_global.json')


    ### Global Graph Index Generate
    loaders = [korean_pdf_loader, korean_pdf_loader] # , notion_loader
    datas = [Path('data/book.pdf'), Path('data/law.pdf')] # , ['1f5dd8ba19864270a8fbdb0b34b55abe']
    index_type = ['law', 'book'] #, 'notion'
    index_set = {}
    summary_set = {
        'law': '이 문서는 민법전으로서 민법전은 민법의 명칭을 가진 법전을 말한다. 이 문서에는 명문화된 민법의 원칙과 정의가 담겨있다.',
        'book': '이 문서는 고려대학교 교수의 『민법학원론』으로 단순히 민법의 조각을 열거하며 서술하기보다는 개념 또는 제도의 원리와 맥락을 드러내고자 힘쓴 문서이다. 일반법으로서 민법이 가지는 추상성을 완화하기 위해서 가급적 구체적 사례를 들어 설명되었다.',
        # 'notion': '이 문서에는 book에 담긴 내용을 요약 정리한 축약된 정보가 들어있음.',
    }

    for loader, data, type in zip(loaders, datas, index_type):
        print(type)
        
        ## document loader
        doc = loader(data)

        ## index
        index = GPTSimpleVectorIndex.from_documents(doc, service_context=service_context)
        index.save_to_disk(f'{type}_index.json')
        index_set[type] = index

        # summary = index.query("What is a summary of this document?", mode="summarize")
        # summary_set[type] = summary

    
    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [index_set[type] for type in index_type],
        [summary_set[type] for type in index_type],
        # max_keywords_per_chunk=50,
        service_context=service_context
    )

    global_graph_index = graph.save_to_disk('global_graph_index.json')

    return global_graph_index



def chatbot(query_str):
    ### define custom QuestionAnswerPrompt
    # query_str
    # QA_PROMPT_TMPL = (
    #     "활용해야 할 정보는 다음과 같다.\n"
    #     "---------------------\n"
    #     "{context_str}"
    #     "\n---------------------\n"
    #     "주어진 정보에 활용해 다음 문제에 대하여,"
    #     "답변해주세요: {query_str}\n"
    # )
    # QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
    
    ### Build GPTSimpleVectorIndex
    # index = GPTSimpleVectorIndex.load_from_disk('index.json')
    # response = index.query(query_str, text_qa_template=QA_PROMPT)

    ### Build GPTKnowledgeGraphIndex
    # new_index = GPTKnowledgeGraphIndex.load_from_disk('index.json', service_context=service_context)
    # response = new_index.query(
    #     query_str, 
    #     text_qa_template=QA_PROMPT,
    #     include_text=True, 
    #     response_mode="tree_summarize",
    #     embedding_mode='hybrid',
    #     similarity_top_k=5
    # )

    ### Global Index
    # global_index = GPTSimpleVectorIndex.load_from_disk(f'index_global.json', service_context=service_context)
    # response = global_index.query(query_str, text_qa_template=QA_PROMPT,  similarity_top_k=5)

    ### Global Graph Index
    graph = ComposableGraph.load_from_disk('global_graph_index.json', service_context=service_context)
    # query_configs = [
    #     {
    #         "index_struct_type": "dict",
    #         "query_mode": "default",
    #         "query_kwargs": {
    #             "similarity_top_k": 1
    #         }
    #     },
    #     {
    #         "index_struct_type": "keyword_table",
    #         "query_mode": "simple",
    #         "query_kwargs": {}
    #     },
    # ]
    query_configs = query_configs = [
        {
            # NOTE: index_struct_id is optional
            "index_struct_type": "tree",
            "query_mode": "default",
            "query_kwargs": {
                "child_branch_factor": 2
            }
        },
        {
            "index_struct_type": "keyword_table",
            "query_mode": "simple",
            "query_kwargs": {}
        },
    ]
    response = graph.query(query_str)#, query_configs=query_configs)
    print(response.response)
    # print(response.get_formatted_sources())

    return response.response

if __name__ == "__main__":
    iface = gr.Interface(fn=chatbot,
                        inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                        outputs="text",
                        title="Law GPT")

    # index = construct_index()
    iface.launch(share=True)