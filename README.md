# In Progress
# LLM Cybersecurity Article Summarizer

## Table of Contents
1. [Project Overview](#project-overview)
2. [Introduction](#introduction)
3. [Background](#background)
4. [Technologies Used](#technologies-used)
5. [Google Colab Notebook](#google-colab-notebook)
6. [Installation](#installation)
7. [Usage](#usage)
   - [Beautiful Soup Approach](#beautiful-soup-approach)
   - [Newspaper3k Approach](#newspaper3k-approach)
8. [Approaches Explored](#approaches-explored)
9. [Challenges Encountered](#challenges-encountered)
10. [Example Articles and Summaries](#example-articles-and-summaries)
11. [Limitations](#limitations)
12. [Conclusions](#conclusions)
13. [Future Work](#future-work)

## Project Overview

The **LLM Cybersecurity Article Summarizer** is a tool designed to extract and summarize articles related to cybersecurity. This project was developed to gain practical experience with Natural Language Processing (NLP) and large language models (LLMs), using models provided by Hugging Face's `transformers` library. During development, I experimented with several summarization models, including `facebook/bart-large-cnn` and `sshleifer/distilbart-cnn-12-6`.

## Introduction
This project focuses on extracting and summarizing articles from the web using two approaches: **Beautiful Soup** and **newspaper3k**. The primary objective is to demonstrate the effectiveness of these libraries for web scraping and natural language processing tasks.

## Background
The growth of digital content has made it increasingly important to have tools that can efficiently extract relevant information from online articles. This project implements two methods for web scraping to compare performance and reliability.

## Technologies Used

- **Python**: The primary programming language used for development.
- **Beautiful Soup**: Used for web scraping to extract article text.
- **Newspaper3k**: Utilized for article extraction, including built-in NLP operations and tokenization.
- **Requests**: Employed to fetch the content of web pages for scraping.
- **NLTK**: Used for tokenization tasks, particularly for processing text data.
- **Hugging Face Transformers**: Implemented for various summarization models.
- **Google Colab** used for development and testing, jupyter notebook idk

## Google Colab Notebook *update toc

You can view and run the project in the Google Colab notebook [here](<your_colab_link>).

## Installation *is this okay
To set up the project, you will need to install the required libraries.

```python
!pip install requests beautifulsoup4 transformers newspaper3k nltk
```
## Usage *mention imf code difference somewhere

### Beautiful Soup Approach

```python
# Import necessary libraries
# requests: To download web content from the specified URL
# BeautifulSoup: For parsing and extracting information from HTML content
# transformers: To use a pre-trained model (BART) for text summarization
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Extract and clean article text from a given URL
def extract_article_text(url):
    try:
        # Send a GET request to the URL and raise an error for any bad response codes (e.g., 404)
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the HTML content of the article using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text content from all paragraph tags in the HTML document
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])

        # Clean up the extracted text by removing any extra spaces
        article_text = ' '.join(article_text.split())  # Normalize spaces
        return article_text.strip()  # Return the cleaned article text
    except Exception as e:
        # Handle errors that occur during the text extraction process
        return f"Failed to extract article text: {str(e)}"

# Summarize the text using a pre-trained transformer model
def summarize_text(text):
    # Initialize the pre-trained summarization model (BART Large CNN model)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Split the input text into chunks of up to 800 characters, as the model has input size limitations
    max_chunk_size = 800
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    # Summarize each chunk and combine the resulting summaries into one
    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=80, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Join the summaries and ensure the final result is clean and coherent
    final_summary = ' '.join(summaries)
    sentences = final_summary.split('.')
    sentences = [s.strip() for s in sentences if s]

    # Return a concise summary, limited to the first 5 sentences
    final_summary = '.\n'.join(sentences[:5])
    # Safeguard to ensure each summary ends with a period
    final_summary = final_summary + '.' if final_summary and not final_summary.endswith('.') else final_summary
    return final_summary

# Main execution flow
if __name__ == "__main__":
    # Prompt the user to enter the article URL
    url = input("Enter Article URL: ")

    # Extract the article text from the specified URL
    article_text = extract_article_text(url)

    # If the text extraction was successful, proceed to summarization
    if not article_text.startswith("Failed"):
        summary = summarize_text(article_text)  # Summarize the extracted text
        print("Summary:")
        print(summary)  # Display the final summary
    else:
        # Print the error message if extraction failed
        print(article_text)
```
### Newspaper3k Approach

```python
# Import necessary libraries
# nltk: For natural language processing tasks like sentence tokenization
# Article: From the newspaper library, to easily handle web articles
import nltk
from newspaper import Article

# Download the 'punkt' resource from nltk, used for sentence tokenization in NLP tasks
nltk.download('punkt')

# Function to extract article information from a given URL
def extract_article_info(url):
    try:
        # Create an Article object with the provided URL
        article = Article(url)

        # Download the article's HTML content
        article.download()

        # Parse the downloaded content to extract the article's text, title, authors, etc.
        article.parse()

        # Perform NLP tasks such as keyword extraction and summarization
        article.nlp()

        # Display key information about the article
        print(f'Title: {article.title}')  # Print the title of the article
        print(f'Authors: {article.authors}')  # Print the list of authors
        print(f'Publication Date: {article.publish_date}')  # Print the publication date
        print(f'Summary: {article.summary}')  # Print the summarized text of the article

    except Exception as e:
        # Handle any errors that occur during article extraction and display the error message
        print(f"An error occurred: {str(e)}")

# Main block of code to execute the program
if __name__ == "__main__":
    # Prompt the user to input the URL of the article they wish to extract
    url = input("Enter Article URL: ")

    # Call the function to extract and display the article's information
    extract_article_info(url)
```
## Approaches Explored *fix both

I experimented with multiple summarization models using Hugging Face's `transformers` library to achieve the best summarization quality:

### 1. `facebook/bart-large-cnn`
   - **Description**: A large pre-trained abstractive summarization model trained on the CNN/Daily Mail dataset, capable of generating human-readable summaries.
   - **Pros**:
     - Effective at summarizing short articles and concise sections.
     - Captured key ideas well in shorter sections.
   - **Cons**:
     - Struggled to maintain coherence when summarizing longer articles or large chunks of text.
     - Execution time increased with the length of the article.
  
### 2. `sshleifer/distilbart-cnn-12-6`
   - **Description**: A distilled version of the BART model, designed for faster inference while maintaining reasonable summarization accuracy.
   - **Pros**:
     - Faster execution, with summaries produced in significantly less time than the full BART model.
     - Suitable for medium-length articles with moderate complexity.
   - **Cons**:
     - Less comprehensive summaries compared to the full BART model.
     - Occasionally missed key details in longer or more complex articles.

## Approaches Explored

I initially experimented with different approaches to extract and summarize article content. The two primary approaches that proved most effective are:

1. **Beautiful Soup for Web Scraping**:
   - **Pros**:
     - Flexibility in extracting specific HTML elements.
     - Useful for custom web scraping tasks.
   - **Cons**:
     - Slower execution time (averaged around 4 minutes for longer articles).
     - Captured only the introduction or first few paragraphs of longer articles.
     - Requires additional parsing logic for different website structures.

2. **Newspaper3k for Article Extraction**:
   - **Pros**:
     - Quick and efficient (typically under 20 seconds).
     - Automatically handles various article formats and provides built-in NLP features.
     - Consistently summarizes about 75% of longer articles.
   - **Cons**:
     - May encounter download errors (e.g., 403 Forbidden) for some URLs.
     - Limited control over the extraction process compared to Beautiful Soup.
    
## Challenges Encountered

During the development of this project, I faced several challenges, including:

- **Punctuation and Formatting Errors**: Initial attempts at text extraction sometimes resulted in inconsistent punctuation and formatting issues, complicating the summarization process.
- **Ambiguity in Pronouns**: Summarization occasionally referenced vague pronouns (e.g., "he" or "they") without sufficient context, making it unclear who was being discussed.
- **Tense Discrepancies**: Some extracted text had inconsistent verb tenses, leading to confusion in the summarization output.
- **Chunking Sensitivity**: While trying different models, I found that chunking was sensitive; one early attempt summarized a whole 10-minute read article into one sentence, while another just paraphrased the whole thing. I had to try many different methods to get anywhere near a 4-5 sentence summary.

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
- **Summary Length**: Neither approach was able to summarize the entirety of longer articles effectively. The Beautiful Soup method captured only the introduction section or the first few paragraphs, while the newspaper3k method consistently summarized about 75% of the article. However, both methods excelled at summarizing smaller sections of the articles.
- **Dependency on External Libraries**: The effectiveness of the summarization relies heavily on the `transformers` library and its underlying models. Changes or deprecations in these libraries could affect the functionality of the summarization process.
- **Limited Context**: The summarizer may struggle with maintaining context across large chunks of text, potentially resulting in summaries that lack coherence or fail to convey the main ideas accurately.

## Conclusions

This project successfully demonstrates the ability to extract and summarize articles related to cybersecurity using large language models. By leveraging the `newspaper` library for article extraction and the `transformers` library for summarization, the application efficiently processes online content, providing concise and relevant summaries. The results indicate that while the tool is effective for many articles, there are areas for improvement, particularly in handling longer texts and ensuring comprehensive summaries.

## Future Work

- **Enhancing Error Handling**: Implement additional mechanisms to handle download failures and other exceptions, providing more informative feedback to users, such as retry mechanisms or detailed error logs.
- **Improving Summary Quality**: Experiment with different LLM-based models or fine-tune existing models to better summarize lengthy articles, addressing the limitation that current approaches don’t always capture the full content of long articles.
- **User Interface Development**: Consider developing a simple graphical user interface (GUI) to make the tool more accessible, expanding its usability beyond command-line users.
- **Expansion of Sources**: Extend support for more diverse news sources, including various article formats, to broaden the tool's capabilities and ensure more comprehensive coverage of cybersecurity content.
- **Integration with Other NLP Tasks**: Explore integrating additional natural language processing tasks, such as sentiment analysis or keyword extraction, to enhance insights alongside summarization, offering more analytical value.
- **Deployment and Automation**: Explore deploying the tool as a web application and automating article retrieval for continuous summarization of cybersecurity news, allowing for real-time updates and broader accessibility.
- **Importance of Human Oversight**: Acknowledge the value of human review in ensuring that AI-generated summaries preserve context and nuances, ensuring the overall quality and reliability of the output.

