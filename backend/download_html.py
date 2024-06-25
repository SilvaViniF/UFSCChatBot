import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def download_html(url, output_dir):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Calculate the ratio of text to overall content
            text = soup.get_text(strip=True)
            if len(text) > 500:  # Example threshold for considering a page to have substantial text
                # Create a valid filename from URL
                url_path = urlparse(url).path.strip("/")
                filename = os.path.join(output_dir, url_path.replace("/", "_") + ".html")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(response.text)
                print(f"Downloaded HTML from {url}")
            else:
                print(f"Skipped {url} - not enough textual content")
        else:
            print(f"Failed to download HTML from {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error while downloading HTML from {url}: {e}")

def scrape_links_and_download(start_url, output_dir, visited=None):
    if visited is None:
        visited = set()

    try:
        response = requests.get(start_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            download_html(start_url, output_dir)
            visited.add(start_url)

            for link in soup.find_all('a', href=True):
                url = urljoin(start_url, link['href'])
                if is_valid_url(url) and url not in visited and urlparse(url).netloc == urlparse(start_url).netloc:
                    scrape_links_and_download(url, output_dir, visited)
    except Exception as e:
        print(f"Error while scraping {start_url}: {e}")

if __name__ == "__main__":
    start_url = "https://blumenau.ufsc.br/"
    output_directory = "files"  # Directory to save downloaded HTML files

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Start scraping and downloading
    scrape_links_and_download(start_url, output_directory)
    print("HTML download complete.")
