# In Progress
# LLM Cybersecurity Article Summarizer

## Table of Contents
1. [Project Overview](#project-overview)
2. [Introduction](#introduction)
3. [Background](#background)
4. [Technologies Used](#technologies-used)
5. [Installation](#installation)
6. [Usage](#usage)
   - [Beautiful Soup Approach](#beautiful-soup-approach)
   - [Newspaper3k Approach](#newspaper3k-approach)
7. [Google Colab Notebook](#google-colab-notebook)
8. [Approaches Explored](#approaches-explored)
   - [Article Extraction Approaches](#article-extraction-approaches)
   - [Summarization Model Approaches](#summarization-model-approaches)
   - [Final Decision](#final-decision)
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
- **Google Colab** Cloud-based platform used for development and testing, with full support for Jupyter notebooks.

## Google Colab Notebook

You can view and run the project in the Google Colab notebook [here](<your_colab_link>).*

## Installation
To run this project in Google Colab, install the necessary libraries by running the following command:

```python
!pip install requests beautifulsoup4 transformers newspaper3k nltk
```
## Usage

To use the summarization tool, follow these steps:

### User Input

When you run the code for either summarization approach, you will be prompted to input the URL of the article you want to summarize. The program will then continue running to extract the article content and generate a summary.

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
## Approaches Explored

Throughout the project, I experimented with both article extraction methods and summarization models to optimize the quality and efficiency of the summarization process.

### Article Extraction Approaches

1. **Beautiful Soup for Web Scraping**:
   - **Description**: Custom web scraping approach using Beautiful Soup to extract specific HTML elements.
   - **Pros**:
     - Flexibility in extracting specific content from a variety of website structures.
     - Useful for handling complex web pages with custom parsing.
     - Typically produces fewer grammatical issues in the extracted content.
   - **Cons**:
     - Slower execution time (averaging around 4 minutes for longer articles).
     - Captured only the introduction or first few paragraphs in most cases.
     - Requires additional logic for handling different website structures.

2. **Newspaper3k for Article Extraction**:
   - **Description**: A library specifically designed for fast and efficient article extraction, with built-in NLP capabilities.
   - **Pros**:
     - Quick and efficient extraction (typically under 20 seconds).
     - Automatically handles various article formats and includes built-in NLP features.
     - Produces summaries that sound more natural and closer to human speech.
     - Extracts and summarizes around 75% of the content from longer articles.
   - **Cons**:
     - May encounter download errors, such as 403 Forbidden for certain URLs.
     - Less control over the extraction process compared to custom web scraping.
     - Tends to generate more grammatical errors in the extracted content.

### Summarization Model Approaches

1. **`facebook/bart-large-cnn`**:
   - **Description**: A pre-trained abstractive summarization model trained on the CNN/Daily Mail dataset.
   - **Pros**:
     - Generates human-readable summaries and effectively captures key ideas in shorter sections.
     - Capable of summarizing complex content with good accuracy.
   - **Cons**:
     - Struggles with maintaining coherence for longer articles or larger chunks of text.
     - Slower execution, especially as the article length increases.

2. **`sshleifer/distilbart-cnn-12-6`**:
   - **Description**: A distilled version of the BART model, optimized for faster inference.
   - **Pros**:
     - Faster execution times, making it ideal for quick summarization.
     - Suitable for medium-length articles with moderate complexity.
   - **Cons**:
     - Tended to paraphrase the entire article rather than producing concise summaries, often resulting in summaries nearly as long as the original text.
     - Occasionally misses key details in longer or more complex articles.

### Final Decision

Although I experimented with the `sshleifer/distilbart-cnn-12-6` model, I ultimately selected the `facebook/bart-large-cnn` model for the final implementation due to its superior summarization quality and its ability to generate concise summaries. I encountered issues with the sshleifer model, as it tended to paraphrase the entire article, resulting in summaries that were nearly as long as the original text. For article extraction, I utilized both Beautiful Soup and Newspaper3k to showcase the differences in quality and speed.
    
## Challenges Encountered

During the development of this project, I faced several challenges, including:

- **Punctuation and Formatting Errors**: Initial attempts at text extraction sometimes resulted in inconsistent punctuation and formatting issues, complicating the summarization process.
- **Ambiguity in Pronouns**: Summarization occasionally referenced vague pronouns (e.g., "he" or "they") without sufficient context, making it unclear who was being discussed.
- **Tense Discrepancies**: Some extracted text had inconsistent verb tenses, leading to confusion in the summarization output.
- **Chunking Sensitivity**: While trying different models, I found that chunking was sensitive; one early attempt summarized a whole 10-minute read article into one sentence, while another just paraphrased the whole thing. I had to try many different methods to get anywhere near a 4-5 sentence summary.
- **Website-Related Content**: At times, the summarization models attempted to summarize content that was more related to the website itself rather than the article's core message. For example, one summary began with a sentence about "The IMF Press Center is a password-protected site for working journalists," which was not relevant to the actual article content.

## Example Articles and Summaries

*Note: The articles used for summarization were found by searching for "cybersecurity articles" on Google, ensuring a diverse range of topics and writing styles. The following summaries are attempts to summarize the entire article, not just the excerpts below.*

### Article 1: [The Global Cyber Threat](https://www.imf.org/external/pubs/ft/fandd/2021/03/global-cyber-threat-to-financial-systems-maurer.htm)

**Original Text Excerpt:**
> Cyber threats to the financial system are growing, and the global community must cooperate to protect it.  
> In February 2016, hackers targeted the central bank of Bangladesh and exploited vulnerabilities in SWIFT, the global financial system’s main electronic payment messaging system, trying to steal $1 billion. While most transactions were blocked, $101 million still disappeared. The heist was a wake-up call for the finance world that systemic cyber risks in the financial system had been severely underestimated.

**Summaries:**
- **Beautiful Soup Summary:**
  > The IMF Press Center is a password-protected site for working journalists.
Cyber threats to the financial system are growing, and the global community must cooperate to protect it.
In February 2016, hackers targeted the central bank of Bangladesh and exploited vulnerabilities in SWIFT.
The world’s governments and companies continue to struggle to contain the threat.
It remains unclear who is responsible for protecting the system.
  > *Note: The first sentence exemplifies how some extraneous content from the website may be included in the article summaries.

- **Newspaper3k Summary:**
  > First, the global financial system is going through an unprecedented digital transformation, which is being accelerated by the COVID-19 pandemic.
Second, malicious actors are taking advantage of this digital transformation and pose a growing threat to the global financial system, financial stability, and confidence in the integrity of the system.
Although they do advance financial inclusion, digital financial services also offer a target-rich environment for hackers.
Better protecting the global financial system is primarily an organizational challenge.
This responsibility gap and continued uncertainty about roles and mandates to protect the global financial system fuel risks.

---

### Article 2: [Cybersecurity Protection Thwarts Sophisticated Attack; Additional Safeguards Under Consideration](https://news.vt.edu/articles/2024/08/it-cybersecurity-protections-enhanced-2-factor.html)

**Original Text Excerpt:**
> In July, a wave of phishing emails targeting Virginia Tech employees attempted to — and nearly succeeded in — diverting direct deposits, including pay, away from their legitimate destination.  
> Here’s what happened: Employees received phishing emails urging them to log into a well-disguised fake login screen. If a user responded, the hackers had that employee's username and password. The hackers then attempted to log into the user’s Virginia Tech email, prompting a Duo authentication request. If the user initially denied this request, the hackers kept trying until the user approved one, a tactic known as "2-factor fatigue."

**Summaries:**
- **Beautiful Soup Summary:**
  > A wave of phishing emails targeting Virginia Tech employees attempted to diverting direct deposits, including pay, away from their legitimate destination.
Fortunately, newly deployed cybersecurity protections within the Division of Information Technology detected the unusual login activity and put a stop to the hack.
Hackers are getting better at what they do, and they are studying our business processes to find vulnerabilities.
Each member of the university community has a role to play in staying safe online.
"We must continue to find ways to shore up cyber defenses, to include a more informed and security-aware community," he says.

- **Newspaper3k Summary:**
  > *An error occurred while attempting to summarize this article: 403 Client Error: Forbidden for the article URL.* 
  > *This illustrates a limitation of the `newspaper` library, which can sometimes fail to retrieve articles despite its faster processing time compared to Beautiful Soup.*

---

### Additional Articles

1. **[Cybersecurity Is Critical for All Organizations – Large and Small](https://www.ifac.org/knowledge-gateway/discussion/cybersecurity-critical-all-organizations-large-and-small)**
   - **Beautiful Soup Summary:**
     > Cybercrime is becoming big business and cyber risk a focus of organizations and governments globally.
Monetary and reputational risks are high if organizations don’t have an appropriate cybersecurity plan.
Cyber-attacks have been steadily climbing for four consecutive years.
The manufacturing sector experienced the greatest proportion of cyber-attacks in 2022.
Recent cases have involved thefts of sensitive information.
   - **Newspaper3k Summary:**
     > Cybersecurity is making sure your organization's data is safe from attacks from both internal and external bad actors.
Once infected, the organization’s data continues to be inaccessible as the encrypts the data using the attackers encryption key.
Cybersecurity GovernanceA cybersecurity governance and risk management program should be established which is appropriate for the size of the organization.
Cybersecurity risk needs to be considered as a significant business risk by the owners and directors.
Reporting of any possible breach of security, unauthorized access, or disclosure of the organizations data.
     > *This summary demonstrates some grammatical issues that can arise with the Newspaper3k approach, highlighting a known limitation.*

2. **[Explore Cybersecurity Through Art at New Exhibition](https://news.vt.edu/articles/2024/10/cci-cyberarts-2024-exhibit.html)** 
   - **Beautiful Soup Summary:**
     > CyberArts 2024 opens at the Torpedo Factory Art Center in Alexandria.
The opening reception will be held on Oct.
18 from 6-8 p.
m.
Registration is required.
   - **Newspaper3k Summary:**
     > *An error occurred while attempting to summarize this article: 403 Client Error: Forbidden for the article URL.* 
     > *This again illustrates a limitation of the `newspaper` library, which can sometimes fail to retrieve articles despite its faster processing time compared to Beautiful Soup.*

3. **[“A Terrible Vulnerability”: Cybersecurity Researcher Discovers Yet Another Flaw in Georgia’s Voter Cancellation Portal](https://www.propublica.org/article/cybersecurity-expert-finds-another-flaw-in-georgia-voter-portal)** fix
   - **Beautiful Soup Summary:**
     > Until Monday, a new online portal run by the Georgia Secretary of State’s Office contained what experts describe as a serious security vulnerability.
The flaw was brought to the attention of ProPublica and Atlanta News First over the weekend.
The issue was “as bad as any voter cancellation bug could be,” a cybersecurity researcher says.
The Georgia Secretary of State’s Office said it had no records of Parker's attempts to reach out.
The Secretary of State’s Office told the news organizations that it quickly fixed the portal.
   - **Newspaper3k Summary:**
     > Parker, who uses they/them pronouns, said that after discovering it, they attempted to contact the Georgia Secretary of State’s Office.
The Secretary of State’s Office told the news organizations that it quickly fixed the portal.
This one would allow any user of the portal to bypass the screen that requires a driver’s license number and submit the cancellation request without it.
A window popped up stating that “Your cancellation request has been successfully submitted” and that county election workers would process the request within a week.
(Parker’s cancellation request would have lacked a driver’s license number.)
     > *Note: This summary demonstrates a pronoun reference issue, which is a common problem when using automated summarization tools like Newspaper3k. The pronoun "it" in the first sentence lacks a clear antecedent, making it unclear what Parker discovered. A more explicit reference would improve clarity (e.g., "Parker discovered a flaw in the voter cancellation portal").*

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

