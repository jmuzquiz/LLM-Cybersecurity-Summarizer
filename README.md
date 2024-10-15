# In Progress
# LLM Cybersecurity Article Summarizer

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Limitations](#limitations)
7. [Conclusions](#conclusions)
8. [Future Work](#future-work)

## Project Overview

The **LLM Cybersecurity Summarizer** is a tool designed to extract and summarize articles related to cybersecurity. This project was developed to gain practical experience with Natural Language Processing (NLP) and large language models (LLMs). During the development process, I explored various models, including the BART model from the `transformers` library for summarization. 

## Introduction
This project focuses on extracting and summarizing articles from the web using two approaches: **Beautiful Soup** and **newspaper3k**. The primary objective is to demonstrate the effectiveness of these libraries for web scraping and natural language processing tasks.

## Background
The growth of digital content has made it increasingly important to have tools that can efficiently extract relevant information from online articles. This project implements two methods for web scraping to compare performance and reliability.

## Technologies Used
- **Python**: The programming language used for the implementation.
- **Beautiful Soup**: A library for parsing HTML and XML documents.
- **newspaper3k**: A library specifically designed for article extraction and summarization.
- **Requests**: For making HTTP requests to access web pages.

## Technologies Used

- **Languages and Libraries**:
  - Python
  - `transformers` (for LLMs and summarization)
  - `newspaper3k` (for article extraction)
  - `BeautifulSoup` (for web scraping)
  - `nltk` (for text processing)
  
- **Environment**:
  - Google Colab (for development and testing)

## Challenges Encountered

During the development of this project, I faced several challenges, including:

- **Punctuation and Formatting Errors**: Initial attempts at text extraction sometimes resulted in inconsistent punctuation and formatting issues, complicating the summarization process.
- **Ambiguity in Pronouns**: Summarization occasionally referenced vague pronouns (e.g., "he" or "they") without sufficient context, making it unclear who was being discussed.
- **Tense Discrepancies**: Some extracted text had inconsistent verb tenses, leading to confusion in the summarization output.


## Installation
To set up the project, you will need to install the required libraries. You can do this using pip:

```python
pip install requests beautifulsoup4 transformers

!pip install newspaper3k
!pip install nltk
!pip install transformers
```
## Usage

### Beautiful Soup Approach

```python
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def extract_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return ' '.join(article_text.split()).strip()
    except Exception as e:
        return f"Failed to extract article text: {str(e)}"

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    max_chunk_size = 800
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = [summarizer(chunk, max_length=80, min_length=30, do_sample=False)[0]['summary_text'] for chunk in text_chunks]
    final_summary = ' '.join(summaries)
    return '. '.join(final_summary.split('.')[:5]) + '.'

if __name__ == "__main__":
    url = input("Enter Article URL: ")
    article_text = extract_article_text(url)
    if not article_text.startswith("Failed"):
        summary = summarize_text(article_text)
        print("Summary:")
        print(summary)
    else:
        print(article_text)
```
### Newspaper3k Approach

```python
import nltk
from newspaper import Article

nltk.download('punkt')

def extract_article_info(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        print(f'Title: {article.title}')
        print(f'Authors: {article.authors}')
        print(f'Publication Date: {article.publish_date}')
        print(f'Summary: {article.summary}')
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    url = input("Enter Article URL: ")
    extract_article_info(url)
```

## Approaches Explored

I initially experimented with different approaches to extract and summarize article content. The two primary approaches that proved most effective are:

1. **Beautiful Soup for Web Scraping**:
   - **Pros**:
     - Flexibility in extracting specific HTML elements.
     - Useful for custom web scraping tasks.
   - **Cons**:
     - Slower execution time (averaged around 4 minutes for longer articles).
     - Requires additional parsing logic for different website structures.

2. **Newspaper3k for Article Extraction**:
   - **Pros**:
     - Quick and efficient (typically under 20 seconds).
     - Automatically handles various article formats and provides built-in NLP features.
   - **Cons**:
     - May encounter download errors (e.g., 403 Forbidden) for some URLs.
     - Limited control over the extraction process compared to Beautiful Soup.

## Example Articles and Summaries

*Note: The articles used for summarization were found by searching for "cybersecurity articles" on Google, ensuring a diverse range of topics and writing styles.*

### Article 1: [Title of Article 1](URL)
https://www.imf.org/external/pubs/ft/fandd/2021/03/global-cyber-threat-to-financial-systems-maurer.htm
*Note: The following summaries are attempts to summarize the entire article, not just the excerpts below.*

**Original Text Excerpt**:
> [Insert a short excerpt of the original text, if necessary]

**Beautiful Soup Summary**:
> [Insert the summary output from the Beautiful Soup approach]

**Newspaper3k Summary**:
> [Insert the summary output from the Newspaper3k approach]

---

### Article 2: [Title of Article 2](URL)
https://news.vt.edu/articles/2024/08/it-cybersecurity-protections-enhanced-2-factor.html
*Note: The following summaries are attempts to summarize the entire article, not just the excerpts below.*

**Original Text Excerpt**:
> [Insert a short excerpt of the original text, if necessary]

**Beautiful Soup Summary**:
> [Insert the summary output from the Beautiful Soup approach]

**Newspaper3k Summary**:
> [Insert the summary output from the Newspaper3k approach]

---
3  https://www.ifac.org/knowledge-gateway/discussion/cybersecurity-critical-all-organizations-large-and-small
4  https://news.vt.edu/articles/2024/10/cci-cyberarts-2024-exhibit.html
5  https://www.propublica.org/article/cybersecurity-expert-finds-another-flaw-in-georgia-voter-portal
### Additional Articles
- [Title of Article 3](URL) - Beautiful Soup Summary: [Insert summary], Newspaper3k Summary: [Insert summary]
- [Title of Article 4](URL) - Beautiful Soup Summary: [Insert summary], Newspaper3k Summary: [Insert summary]


## Limitations

- **Data Extraction Failures**: The `newspaper` library may encounter issues with certain URLs, resulting in download failures (e.g., 403 Forbidden errors). This limits the range of articles that can be processed.
- **Summary Length**: The summarization process may not capture the entirety of longer articles, which could lead to the omission of important details. Adjusting the chunk sizes may help, but it can also introduce incoherence in the final summary.
- **Dependency on External Libraries**: The effectiveness of the summarization relies heavily on the `transformers` library and its underlying models. Changes or deprecations in these libraries could affect the functionality of the summarization process.
- **Limited Context**: The summarizer may struggle with maintaining context across large chunks of text, potentially resulting in summaries that lack coherence or fail to convey the main ideas accurately.

## Conclusions

This project successfully demonstrates the ability to extract and summarize articles related to cybersecurity using large language models. By leveraging the `newspaper` library for article extraction and the `transformers` library for summarization, the application efficiently processes online content, providing concise and relevant summaries. The results indicate that while the tool is effective for many articles, there are areas for improvement, particularly in handling longer texts and ensuring comprehensive summaries.

## Future Work

- **Enhancing Error Handling**: Implement additional mechanisms to handle download failures and other exceptions, providing more informative feedback to users.
- **Improving Summary Quality**: Experiment with different summarization models or fine-tune existing models to better capture the main ideas of lengthy articles.
- **User Interface Development**: Consider developing a simple graphical user interface (GUI) to make the tool more accessible to users who may not be familiar with command-line interfaces.
- **Expansion of Sources**: Extend the functionality to support more diverse news sources and formats, improving the range of articles that can be summarized.
- **Integration with Other NLP Tasks**: Explore the potential for integrating other natural language processing tasks, such as sentiment analysis or keyword extraction, to enhance the tool’s capabilities.

