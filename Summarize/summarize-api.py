# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import re
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from nltk.tokenize import sent_tokenize
from fastapi.middleware.cors import CORSMiddleware

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

class SummarizedText(BaseModel):
    text: str

async def process_summarization(final_resultant_text):
    # Process the summarization result, e.g., save to a database or send a notification
    print(final_resultant_text)

async def fetch_url(url, session):
    async with session.get(url) as response:
        return await response.text()

async def summarize(url_input: URLInput, background_tasks: BackgroundTasks):
    url = url_input.url

    # Create a fake user agent
    user_agent = UserAgent()

    async with aiohttp.ClientSession(headers={'User-Agent': user_agent.random}) as session:
        try:
            html_content = await fetch_url(url, session)

            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find and extract text within both <h> (header) and <p> (paragraph) tags
            header_text = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            paragraph_text = [p.get_text() for p in soup.find_all('p')]

            # Combine all the extracted text into a single string
            all_text = ' '.join(header_text + paragraph_text)

            # Use the regular expression to filter for Arabic text
            arabic_text = re.findall(r'[\u0600-\u06FF\s]+', all_text)

            # Join the extracted Arabic text into a single string
            extracted_arabic_text = ' '.join(arabic_text)

            # Initialize the summarization model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            m_name = "marefa-nlp/summarization-arabic-english-news"

            tokenizer = AutoTokenizer.from_pretrained(m_name)
            model = AutoModelWithLMHead.from_pretrained(m_name).to(device)

            def get_summary(text, tokenizer, model, device="cpu", num_beams=2):
                if len(text.strip()) < 50:
                    return ["Please provide a longer text"]

                text = "summarize: <paragraph> " + " <paragraph> ".join([s.strip() for s in sent_tokenize(text) if s.strip() != ""]) + " </s>"
                text = text.strip().replace("\n", "")

                tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)

                summary_ids = model.generate(
                    tokenized_text,
                    max_length=512,
                    num_beams=num_beams,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    early_stopping=True
                )

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return [s.strip() for s in output.split("<hl>") if s.strip() != ""]

            # Initialize the final resultant text
            final_resultant_text = ""

            # Break the Arabic text into chunks of 1000 characters
            chunk_size = 1000
            chunks = [extracted_arabic_text[i:i + chunk_size] for i in range(0, len(extracted_arabic_text), chunk_size)]

            # Generate and append each chunk's summary to the final resultant text
            for chunk in chunks:
                summaries = get_summary(chunk, tokenizer, model, device)
                for summary in summaries:
                    final_resultant_text += summary + " "

            # Add an asynchronous background task to process the summarization result
            background_tasks.add_task(process_summarization, final_resultant_text)

            return {"text": final_resultant_text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=SummarizedText)
async def async_summarize(url_input: URLInput, background_tasks: BackgroundTasks):
    try:
        # Set a timeout for the summarization task
        timeout = 2000  # Adjust as needed
        summarization_result = await asyncio.wait_for(summarize(url_input, background_tasks), timeout=timeout)

        return summarization_result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
