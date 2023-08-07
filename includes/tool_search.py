import requests, os
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

def search_bing(question):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    
    query = "+".join(question.split())
    url = f"https://www.bing.com/search?q={query}"
    print(url)
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print(soup)

    answer_box = soup.find('div', class_='b_ans')
    if answer_box:
        return answer_box.get_text()
    
    snippets = soup.find_all('div', class_='b_caption')
    if snippets:
        return snippets[0].get_text()
    
    return None

def search_google(query, api_key):
    params = {
        "q": query,
        "api_key": api_key
    }

    url = "https://serpapi.com/search"

    response = requests.get(url, params=params)
    data = response.json()

    if "organic_results" in data:
        first_result = data["organic_results"][0]
        if "snippet" in first_result:
            return first_result["snippet"]

    return None

if __name__ == '__main__':

    tools = {
        'search': {
            'description': (
                'a search engine. useful for when you need to answer questions about '
                'current events. input should be a search query.'
            ),
            'execute': search_google,
        },
    }

    # Example usage
    question = "Tokyo high temperature yesterday"
    result = tools['search']['execute'](question, os.getenv("SERPAPI_API_KEY"))
    print(result)