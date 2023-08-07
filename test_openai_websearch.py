import openai,os
from dotenv import load_dotenv

# Local imports
from includes import helper_openai
from includes.tool_search import search_google

load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_search_query(question):
    prompt = f"Generate a search engine query for the following question: {question}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    print(helper_openai.print_job_detail(response))
    return response.choices[0].text.strip()

def generate_answer(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    print(helper_openai.print_job_detail(response))
    return response.choices[0].text.strip()

# Main function
def main():
    original_question = "What is the high temperature for Tokyo yesterday in Celsius"
    print("Original Question:", original_question)
    print("="*50)

    # Generate a search query using OpenAI
    search_query = generate_search_query(original_question)
    print("Generated Search Query:", search_query)
    print("="*50)
    
    # Use the search query to fetch results
    search_result = search_google(search_query.replace('"',''), os.getenv("SERPAPI_API_KEY"))
    print("Search Result:", search_result)
    print("="*50)
    
    # Generate an answer using the search result as a prompt
    answer_prompt = f"Given the search result: {search_result}, answer the original question: {original_question}"
    answer = generate_answer(answer_prompt)
    
    print("Answer:", answer)

if __name__ == "__main__":
    main()
