import os
import requests
from bs4 import BeautifulSoup
from readability import Document
from urllib.parse import urljoin, urlparse

def is_valid_url(url, base_domain):
    """
    Checks if the URL is valid and belongs to the same domain as the start URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme) and base_domain in parsed.netloc

def get_all_links(url):
    """
    Returns all the links found on a web page.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = [a.get('href') for a in soup.find_all('a', href=True)]
        return links
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return []

def download_pdf(url, output_directory):
    """
    Downloads a PDF file from the given URL.
    """
    try:
        response = requests.get(url)
        file_name = os.path.join(output_directory, url.split('/')[-1])
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded PDF: {file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

def download_readable_html(url, output_directory):
    """
    Downloads the readable content of an HTML page from the given URL.
    """
    try:
        response = requests.get(url)
        doc = Document(response.text)
        readable_html = doc.summary()

        # Create a file name based on the URL path
        parsed_url = urlparse(url)
        path = parsed_url.path if parsed_url.path else "index.html"
        file_name = os.path.join(output_directory, path.strip("/").replace("/", "_") + ".html")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # Save the readable content
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(readable_html)
        print(f"Downloaded readable HTML: {file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

def scrape_links_and_download(start_url, output_directory, visited):
    """
    Scrapes links from the start URL and downloads all PDFs and readable HTML pages.
    """
    to_visit = [start_url]
    base_domain = urlparse(start_url).netloc

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"Visiting: {current_url}")
        
        # Download the current page readable HTML
        download_readable_html(current_url, output_directory)
        
        links = get_all_links(current_url)

        for link in links:
            absolute_link = urljoin(start_url, link)
            if is_valid_url(absolute_link, base_domain):
                if absolute_link.endswith(".pdf"):
                    download_pdf(absolute_link, output_directory)
                elif absolute_link not in visited:
                    to_visit.append(absolute_link)

if __name__ == "__main__":
    start_url = "https://adm.blumenau.ufsc.br/planosENSINO/"
    output_directory = "downloaded_files"  # Directory to save downloaded files

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    visited = set()
    # Start scraping and downloading files
    scrape_links_and_download(start_url, output_directory, visited)
    print("Download complete.")
