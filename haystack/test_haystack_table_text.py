# %%
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# %%
import pandas as pd
import os, json

from haystack.document_stores import FAISSDocumentStore
# from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import fetch_archive_from_http, print_answers
from haystack import Document

from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes import TableReader, FARMReader, RouteDocuments, JoinAnswers

# %%
# Get the host where Elasticsearch is running, default to localhost
# host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_index = "document"
# document_store = ElasticsearchDocumentStore( host=host, index=document_index )
document_store = FAISSDocumentStore(similarity="cosine", embedding_dim=768, index=document_index)

# %%
# doc_dir = "data/tutorial15"
# s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/table_text_dataset.zip"
# fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# %%
def read_json_to_tables(filename):
    processed_tables = []
    with open(filename) as tables:
        tables = json.load(tables)
        for key, table in tables.items():
            current_columns = table["header"]
            current_rows = table["data"]
            current_df = pd.DataFrame(columns=current_columns, data=current_rows)
            document = Document(content=current_df, content_type="table", id=key)
            processed_tables.append(document)

    return processed_tables

def read_csv_to_tables(filename):
    processed_tables = []
    with open(filename) as file:
        current_df = pd.read_csv(file, nrows=10000, index_col=False)
        print(filename, current_df)
        document = Document(content=current_df, content_type="table", id=os.path.splitext(os.path.basename(filename))[0])
        processed_tables.append(document)

    return processed_tables

# tables = read_json_to_tables(f"{doc_dir}/tables.json")
# https://insights.stackoverflow.com/survey
tables = read_csv_to_tables("data/survey/survey_results_schema.csv")
tables = read_csv_to_tables("data/survey/survey_results_public.csv")
document_store.write_documents(tables, index=document_index)

# Showing content field and meta field of one of the Documents of content_type 'table'
print(tables[0].content)
print("="*50)
print(tables[0].meta)

# %%
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/all-mpnet-base-v2-table")

# retriever = EmbeddingRetriever(document_store=document_store, embedding_model="davinci",model_format='openai',api_key=os.getenv("OPENAI_API_KEY"))
document_store.update_embeddings(retriever=retriever)
retrieved_tables = retriever.retrieve("Explain this dataset", top_k=5)
print(retrieved_tables[0].content)

# %%
reader = TableReader(model_name_or_path="google/tapas-base-finetuned-wtq", max_seq_len=512)
table_doc = document_store.get_document_by_id("datasurveysurvey_results_schema.csv")
# table_doc = document_store.get_document_by_id("36964e90-3735-4ba1-8e6a-bec236e88bb2")
print(table_doc.content)

# %%
prediction = reader.predict(query="Explain what is this dataset", documents=[table_doc])
print_answers(prediction, details="all")
print("="*50)
print(f"Predicted answer: {prediction['answers'][0].answer}")
print(f"Meta field: {prediction['answers'][0].meta}")

# %%
# Initialize pipeline
from haystack import Pipeline

table_qa_pipeline = Pipeline()
table_qa_pipeline.add_node(component=retriever, name="EmbeddingRetriever", inputs=["Query"])
table_qa_pipeline.add_node(component=reader, name="TableReader", inputs=["EmbeddingRetriever"])
# prediction = table_qa_pipeline.run("When was Guilty Gear Xrd : Sign released?", params={"top_k": 3})
prediction = table_qa_pipeline.run("Explain what is this dataset", params={"top_k": 3})
print_answers(prediction, details="minimum")
print("="*50)
print(f"Predicted answer: {prediction['answers'][0].answer}")
print(f"Meta field: {prediction['answers'][0].meta}")

# # %%
# text_reader = FARMReader("deepset/roberta-base-squad2")
# # In order to get meaningful scores from the TableReader, use "deepset/tapas-large-nq-hn-reader" or
# # "deepset/tapas-large-nq-reader" as TableReader models. The disadvantage of these models is, however,
# # that they are not capable of doing aggregations over multiple table cells.
# table_reader = TableReader("deepset/tapas-large-nq-hn-reader")
# route_documents = RouteDocuments()
# join_answers = JoinAnswers()

# text_table_qa_pipeline = Pipeline()
# text_table_qa_pipeline.add_node(component=retriever, name="EmbeddingRetriever", inputs=["Query"])
# text_table_qa_pipeline.add_node(component=route_documents, name="RouteDocuments", inputs=["EmbeddingRetriever"])
# text_table_qa_pipeline.add_node(component=text_reader, name="TextReader", inputs=["RouteDocuments.output_1"])
# text_table_qa_pipeline.add_node(component=table_reader, name="TableReader", inputs=["RouteDocuments.output_2"])
# text_table_qa_pipeline.add_node(component=join_answers, name="JoinAnswers", inputs=["TextReader", "TableReader"])

# # Example query whose answer resides in a text passage
# # predictions = text_table_qa_pipeline.run(query="Who was Thomas Alva Edison?")
# predictions = text_table_qa_pipeline.run(query="Who played Gregory House in the series House?")
# # We can see both text passages and tables as contexts of the predicted answers.
# print_answers(predictions, details="minimum")
# # %%

# %%
