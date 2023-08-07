import os

from haystack import Document
from haystack.nodes import PromptNode, PromptTemplate
from haystack.pipelines import Pipeline

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

documents = [Document("Berlin is the capital of Germany.")]
pt = PromptTemplate("Given the context please answer the question, don't elaborate. \n\n"
                    "Context: {join(documents)}; \n\n Question: {query} \n\nAnswer:")
lfqa_node = PromptNode(model_name_or_path="gpt-3.5-turbo",
                       api_key=openai_key,
                       max_length=512,
                       default_prompt_template=pt,
                       output_variable="my_answer")

elaboration_prompt = PromptTemplate("Provide additional details about this topic: {my_answer}")
elaboration_node = PromptNode(model_name_or_path="gpt-3.5-turbo",
                              api_key=openai_key,
                              max_length=512,
                              default_prompt_template=elaboration_prompt)

pipe = Pipeline()
pipe.add_node(component=lfqa_node, name="lfqa_node", inputs=["Query"])
pipe.add_node(component=elaboration_node, name="elaboration_node", inputs=["lfqa_node"])

result = pipe.run(query="What is the capital of Germany?", documents=documents)
print(result)