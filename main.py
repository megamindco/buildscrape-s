from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configuration
app = Flask(__name__)
BEARER_TOKEN = os.getenv('BEARER_TOKEN')  # Load the bearer token from the .env file
REQUEST_LIMIT = 100
request_count = 0
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

# Function to check bearer token
def check_bearer_token(token):
    return token == BEARER_TOKEN

# Asynchronous function to fetch webpage
async def fetch_webpage(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            return await response.text()

# Function to preprocess content
def preprocess_content(content):
    """Clean and format the content for better summarization."""
    return " ".join(content.split())[:5000]  # Limit size for performance

# Function to summarize content in parallel
def summarize_content_parallel(content_chunks):
    with ThreadPoolExecutor() as executor:
        summaries = list(
            executor.map(
                lambda chunk: summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'],
                content_chunks
            )
        )
    return " ".join(summaries)

# Function to scrape and summarize website content
async def scrape_and_summarize(url, summarize=False):
    # Fetch webpage asynchronously
    html_content = await fetch_webpage(url)
    if not html_content:
        return {'error': 'Failed to fetch the website'}

    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    full_content = " ".join(
        [tag.get_text(strip=True) for tag in soup.find_all(['h1', 'h2', 'p'])]
    )

    # Preprocess and optionally summarize content
    if summarize:
        full_content = preprocess_content(full_content)
        chunk_size = 1024
        content_chunks = [full_content[i:i + chunk_size] for i in range(0, len(full_content), chunk_size)]
        full_content = summarize_content_parallel(content_chunks)

    return {'content': full_content}

@app.route('/scrape', methods=['GET'])
async def scrape():
    global request_count
    if request_count >= REQUEST_LIMIT:
        return jsonify({'error': 'Request limit reached'}), 429

    token = request.headers.get('Authorization')
    if not token or not check_bearer_token(token.split(' ')[1]):
        return jsonify({'error': 'Unauthorized'}), 401

    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'URL parameter is required'}), 400

    summarize = request.args.get('summarize', 'false').lower() == 'true'

    data = await scrape_and_summarize(url, summarize)
    request_count += 1
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
