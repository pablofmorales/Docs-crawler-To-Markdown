#!/usr/bin/env python3

import argparse
import requests
import html2text
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import deque
import logging
import re

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def is_valid_url(url):
    """Checks if a URL has a valid scheme and network location."""
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def get_domain(url):
    """Extracts the domain name (netloc) from a URL."""
    try:
        return urlparse(url).netloc
    except Exception as e:
        logging.error(f"Could not parse domain for URL '{url}': {e}")
        return None

def sanitize_filename(url_path):
    """Creates a safe filename from a URL path."""
    # Remove scheme and domain if present (should mostly be paths)
    if url_path.startswith(('http://', 'https://')):
        url_path = urlparse(url_path).path

    # Remove leading/trailing slashes and replace others with underscores
    sanitized = url_path.strip('/').replace('/', '_')

    # Remove invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)

    # Handle empty paths (root page)
    if not sanitized:
        return "index"

    # Limit length (optional)
    max_len = 100
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]

    return sanitized

def fetch_page(url, headers):
    """Fetches the content of a URL."""
    try:
        response = requests.get(url, headers=headers, timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        # Check content type - only process HTML
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            return response.text
        else:
            logging.info(f"Skipping non-HTML content at: {url} (Content-Type: {response.headers.get('Content-Type')})")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching {url}: {e}")
        return None

def extract_links(html_content, base_url, target_domain):
    """Extracts all valid, same-domain links from HTML content."""
    links = set()
    soup = BeautifulSoup(html_content, 'html.parser')
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        # Join relative URLs with the base URL
        full_url = urljoin(base_url, href)
        # Clean URL (remove fragment identifiers like #section)
        full_url = urlparse(full_url)._replace(fragment="").geturl()

        # Validate the URL and check if it's within the target domain
        if is_valid_url(full_url) and get_domain(full_url) == target_domain:
            links.add(full_url)
    return links

def convert_to_markdown(html_content):
    """Converts HTML content to Markdown."""
    if not html_content:
        return ""
    try:
        h = html2text.HTML2Text()
        # Configure html2text (optional, defaults are often fine)
        h.ignore_links = False
        h.ignore_images = True # Ignore images for LLM context
        h.body_width = 0 # Don't wrap lines
        return h.handle(html_content)
    except Exception as e:
        logging.error(f"Error converting HTML to Markdown: {e}")
        return "" # Return empty string on conversion error

def save_markdown(markdown_content, url, output_dir):
    """Saves Markdown content to a file named after the URL path."""
    if not markdown_content:
        logging.warning(f"No content to save for URL: {url}")
        return

    # Create filename based on the URL path
    url_path = urlparse(url).path
    filename_base = sanitize_filename(url_path)
    filename = f"{filename_base}.md"
    filepath = os.path.join(output_dir, filename)

    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Source URL: {url}\n\n") # Add source URL as header
            f.write(markdown_content)
        logging.info(f"Saved Markdown for {url} to {filepath}")
    except IOError as e:
        logging.error(f"Error saving file {filepath}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred saving {filepath}: {e}")


# --- Main Crawling Logic ---

def crawl_site(start_url, output_dir, max_pages=100):
    """Crawls a website starting from start_url and saves pages as Markdown."""
    if not is_valid_url(start_url):
        logging.error(f"Invalid start URL provided: {start_url}")
        return

    target_domain = get_domain(start_url)
    if not target_domain:
        logging.error(f"Could not determine domain for start URL: {start_url}")
        return

    # Use a queue for URLs to visit (FIFO - Breadth-First Search)
    queue = deque([start_url])
    # Use a set to keep track of visited URLs to avoid loops and re-processing
    visited = {start_url}
    pages_processed = 0

    # Basic User-Agent
    headers = {
        'User-Agent': 'SimpleMarkdownCrawler/1.0 (https://github.com/your-repo; mailto:your-email@example.com)'
        # Replace with your actual contact info if making public
    }

    logging.info(f"Starting crawl at: {start_url}")
    logging.info(f"Target domain: {target_domain}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Max pages to crawl: {max_pages}")

    while queue and pages_processed < max_pages:
        current_url = queue.popleft()
        logging.info(f"Processing ({pages_processed + 1}/{max_pages}): {current_url}")

        html_content = fetch_page(current_url, headers)

        if html_content:
            # Convert to Markdown and save
            markdown_content = convert_to_markdown(html_content)
            save_markdown(markdown_content, current_url, output_dir)
            pages_processed += 1

            # Find new links on the current page
            new_links = extract_links(html_content, current_url, target_domain)

            # Add new, unvisited links to the queue
            for link in new_links:
                if link not in visited:
                    visited.add(link)
                    queue.append(link)
                    logging.debug(f"Added to queue: {link}")
        else:
            logging.warning(f"Skipping processing due to fetch error or non-HTML content: {current_url}")


    logging.info(f"Crawling finished. Processed {pages_processed} pages.")
    if queue and pages_processed >= max_pages:
        logging.warning(f"Stopped crawling because max_pages ({max_pages}) limit was reached.")


# --- Command-Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl a website starting from a URL, convert pages to Markdown, and save them."
    )
    parser.add_argument(
        "start_url",
        metavar="URL",
        help="The starting URL to crawl (e.g., https://docs.example.com)."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        default="markdown_output",
        help="Directory to save the Markdown files (default: markdown_output)."
    )
    parser.add_argument(
        "-m", "--max-pages",
        dest="max_pages",
        type=int,
        default=100,
        help="Maximum number of pages to crawl (default: 100)."
    )

    args = parser.parse_args()

    crawl_site(args.start_url, args.output_dir, args.max_pages)
