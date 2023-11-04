import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import re
from fastapi.middleware.cors import CORSMiddleware

# Define the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

@app.post("/rating", response_model=float)  # Change response_model to float
async def calculate_similarity(url_input: URLInput):
    url = url_input.url

    # Create a fake user agent
    user_agent = UserAgent()

    # Send an HTTP request to the URL with the fake user agent
    headers = {'User-Agent': user_agent.random}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find and extract text within both <h> (header) and <p> (paragraph) tags
        header_text = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
        paragraph_text = [p.get_text() for p in soup.find_all('p')]

        # Combine all the extracted text into a single string
        all_text = ' '.join(header_text + paragraph_text)

        # Use a regular expression to filter for Arabic text
        arabic_text = re.findall(r'[\u0600-\u06FF\s]+', all_text)
        arabic_text = ' '.join(arabic_text)

    else:
        return 0.0  # Return 0.0 for failure to retrieve the page

    # Load keywords from a text file
    file_path = 'D:\AI - Aliasghar\Ali Asghar\Arabic Text\keywords.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        keywords = file.read()

    # Load a pre-trained Arabic BERT model and tokenizer
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the texts and obtain embeddings
    text1_tokens = tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True)
    text2_tokens = tokenizer(keywords, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        text1_embeddings = model(**text1_tokens).last_hidden_state.mean(dim=1)
        text2_embeddings = model(**text2_tokens).last_hidden_state.mean(dim=1)

    # Calculate the cosine similarity between the two text embeddings
    similarity = cosine_similarity(text1_embeddings, text2_embeddings)
    similarity_score = similarity[0][0]

    return similarity_score  # Return the similarity score as a float

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7000)