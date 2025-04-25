#!/usr/bin/env python3

import argparse
import requests
import html2text
import os
from bs4 import BeautifulSoup, FeatureNotFound # Import FeatureNotFound
from urllib.parse import urlparse, urljoin
from collections import deque
import logging
import re
import sys
import time # Import time for potential delays
import urllib3 # Import urllib3

# Import Rich components
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskID
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# --- Configuration ---
# Initialize Rich Console for printing colored/styled text
# Force terminal=True if output is being piped but we still want colors/progress
console = Console(force_terminal=True) if not sys.stdout.isatty() else Console()


# Configure logging using RichHandler
# This handler automatically formats logs nicely and works with Rich's output
logging.basicConfig(
    level=logging.WARNING, # Default level
    format="%(message)s", # Let RichHandler handle the formatting
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True, tracebacks_suppress=[requests, urllib3])] # Suppress long tracebacks from requests/urllib3
)
logger = logging.getLogger() # Get the root logger

# --- Helper Functions ---

# (is_valid_url, get_domain, sanitize_filename remain the same)

def is_valid_url(url):
    """Checks if a URL has a valid scheme and network location."""
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def get_domain(url):
    """Extracts the domain name (netloc) from a URL."""
    try:
        return urlparse(url).netloc
    except Exception as e:
        logger.error(f"Could not parse domain for URL '{url}': {e}")
        return None

def sanitize_filename(url_path):
    """Creates a safe filename from a URL path."""
    if url_path.startswith(('http://', 'https')):
        url_path = urlparse(url_path).path
    sanitized = url_path.strip('/').replace('/', '_')
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)
    if not sanitized:
        return "index"
    max_len = 100
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized

# (fetch_page, extract_links, convert_to_markdown, save_markdown remain largely the same,
#  just using logger directly)

def fetch_page(url, headers, get_content=True):
    """Fetches a URL. Optionally gets only headers or full content."""
    method = 'GET' if get_content else 'HEAD' # Use HEAD if not getting content
    logger.debug(f"Fetching URL ({method}): [link={url}]{url}[/link]") # Use Rich link markup
    try:
        # Try HEAD first if we don't need content, fallback to GET if HEAD fails/is disallowed
        if not get_content:
            try:
                # Use a slightly shorter timeout for HEAD requests
                response = requests.request(method, url, headers=headers, timeout=8, allow_redirects=True) # Allow redirects for HEAD
                response.raise_for_status()
                # Check content type even for HEAD request if possible
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type:
                     # If HEAD worked and it's HTML, we might still need GET for links
                     # Let's fetch again with GET but minimal content read
                     response_get = requests.get(url, headers=headers, timeout=10, stream=True) # Shorter timeout for partial read
                     response_get.raise_for_status()
                     # Read a small chunk to parse links, avoid full download
                     html_chunk = b"" # Initialize as bytes
                     try:
                        # Read up to 512KB, should be enough for head + links in most cases
                        # Use iter_content for better control over streamed reading
                        for chunk in response_get.iter_content(chunk_size=8192, decode_unicode=False):
                             html_chunk += chunk
                             if len(html_chunk) > 512 * 1024:
                                 logger.debug(f"Reached partial read limit (512KB) for link discovery: [link={url}]{url}[/link]")
                                 break
                        logger.debug(f"Successfully fetched partial HTML ({len(html_chunk)} bytes) for links from: [link={url}]{url}[/link]")
                        # Decode after reading needed chunks
                        return html_chunk.decode('utf-8', errors='ignore')
                     finally:
                        response_get.close() # Ensure connection is closed
                else:
                    logger.debug(f"Skipping non-HTML content ({content_type}) identified via HEAD: [link={url}]{url}[/link]")
                    return None # HEAD successful but not HTML
            except requests.exceptions.RequestException as head_err:
                 logger.debug(f"HEAD request failed for [link={url}]{url}[/link] ({head_err}), falling back to GET for link discovery.")
                 # Fall through to GET request below

        # Perform GET request if get_content is True or HEAD failed/skipped
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            logger.debug(f"Successfully fetched full HTML from: [link={url}]{url}[/link]")
            return response.text # Return full content
        else:
            logger.debug(f"Skipping non-HTML content ({content_type}) at: [link={url}]{url}[/link]")
            return None

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching [link={url}]{url}[/link]")
        return None
    except requests.exceptions.HTTPError as e:
        # Log HTTP errors (like 404 Not Found, 403 Forbidden)
        # Don't log 404 as warning during discovery if verbose is not set
        log_level = logging.WARNING if e.response.status_code != 404 or logger.isEnabledFor(logging.INFO) else logging.DEBUG
        logger.log(log_level, f"HTTP error fetching [link={url}]{url}[/link]: {e.response.status_code} {e.response.reason}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Fetch error for [link={url}]{url}[/link]: {e}")
        return None
    except Exception as e:
        # Log full error if verbose, otherwise just a summary
        if logger.isEnabledFor(logging.DEBUG):
             logger.exception(f"An unexpected error occurred fetching [link={url}]{url}[/link]:") # Use logger.exception for traceback
        else:
             logger.error(f"An unexpected error occurred fetching [link={url}]{url}[/link]: {e}")
        return None


def extract_links(html_content, base_url, target_domain, parser_choice='html.parser'):
    """Extracts all valid, same-domain links from HTML content."""
    links = set()
    if not html_content:
        return links
    logger.debug(f"Extracting links from: [link={base_url}]{base_url}[/link] using [i]{parser_choice}[/i]")
    try:
        # Use the chosen parser ('lxml' or 'html.parser')
        try:
             soup = BeautifulSoup(html_content, parser_choice)
        except FeatureNotFound:
             logger.warning(f"Parser '[i]{parser_choice}[/i]' not found. Falling back to '[i]html.parser[/i]'. Install '[b]lxml[/b]' for potential speed improvements.")
             soup = BeautifulSoup(html_content, 'html.parser') # Fallback parser

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if not href or href.startswith('#') or href.lower().startswith(('javascript:', 'mailto:', 'tel:')):
                 continue # Skip empty, fragment, or non-http links

            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)

            # Ensure it's http or https before proceeding
            if parsed_url.scheme not in ['http', 'https']:
                continue

            full_url = parsed_url._replace(fragment="").geturl() # Remove fragment

            if is_valid_url(full_url) and get_domain(full_url) == target_domain:
                links.add(full_url)

        logger.debug(f"Found {len(links)} potential same-domain links on [link={base_url}]{base_url}[/link]")
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception(f"Error parsing links from [link={base_url}]{base_url}[/link]:")
        else:
            logger.error(f"Error parsing links from [link={base_url}]{base_url}[/link]: {e}")
    return links

def convert_to_markdown(html_content, url):
    """Converts HTML content to Markdown."""
    if not html_content:
        return ""
    logger.debug(f"Converting HTML to Markdown for: [link={url}]{url}[/link]")
    try:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        h.ignore_emphasis = True
        h.skip_internal_links = True
        h.ignore_tables = True
        h.ignore_tags = ('nav', 'footer', 'script', 'style', 'aside', 'header', 'button', 'form', 'input', 'textarea', 'select', 'figure', 'figcaption') # Added figure tags
        markdown = h.handle(html_content)
        # Advanced cleaning: remove lines that are likely headers/footers based on link density or common patterns
        lines = markdown.split('\n')
        cleaned_lines = [line for line in lines if not (line.count('[') > 2 and len(line) < 150)] # Simple heuristic
        markdown = '\n'.join(cleaned_lines)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown).strip() # Collapse excessive newlines
        logger.debug(f"Conversion successful for: [link={url}]{url}[/link]")
        return markdown
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception(f"Error converting HTML to Markdown for [link={url}]{url}[/link]:")
        else:
            logger.error(f"Error converting HTML to Markdown for [link={url}]{url}[/link]: {e}")
        return ""

def save_markdown(markdown_content, url, output_dir):
    """Saves Markdown content to a file named after the URL path."""
    if not markdown_content or not markdown_content.strip():
        logger.debug(f"No substantial content to save for URL: [link={url}]{url}[/link]")
        return False

    url_path = urlparse(url).path
    filename_base = sanitize_filename(url_path)
    filename = f"{filename_base}.md"
    filepath = os.path.join(output_dir, filename)

    logger.debug(f"Attempting to save Markdown to: [repr.filename]{filepath}[/]")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Source URL: {url}\n\n")
            f.write(markdown_content.strip())
        logger.debug(f"Successfully saved: [repr.filename]{filepath}[/]")
        return True
    except IOError as e:
        logger.error(f"Error saving file [repr.filename]{filepath}[/]: {e}")
        return False
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
             logger.exception(f"An unexpected error occurred saving [repr.filename]{filepath}[/]:")
        else:
             logger.error(f"An unexpected error occurred saving [repr.filename]{filepath}[/]: {e}")
        return False

# --- Phase 1: Discovery ---
def discover_all_links(start_url, target_domain, headers, parser_choice, progress):
    """Discovers all reachable, unique, same-domain URLs via BFS using Rich Progress."""
    # Log start only if verbose
    if logger.isEnabledFor(logging.INFO):
        logger.info("[bold cyan]Phase 1:[/bold cyan] Discovering all reachable URLs...")

    queue = deque([start_url])
    visited = {start_url}

    # Add a task to the Rich Progress object for discovery
    task_id = progress.add_task("[cyan]Discovering...", total=None, start_url=start_url)
    progress.update(task_id, total=1, completed=0) # Set initial total to 1

    while queue:
        current_url = queue.popleft()
        logger.debug(f"Discovery: Checking [link={current_url}]{current_url}[/link]")
        # Update progress description to show the URL being checked in discovery phase
        display_url = current_url
        if len(display_url) > 70: # Simple truncation for display
            display_url = display_url[:35] + "..." + display_url[-32:]
        progress.update(task_id, description=f"[cyan]Discovering...[/] [dim link={current_url}]{display_url}[/]")


        # Fetch minimal content just for links
        html_content = fetch_page(current_url, headers, get_content=False)

        if html_content:
            # Pass the parser choice to extract_links
            new_links = extract_links(html_content, current_url, target_domain, parser_choice)
            added_count = 0
            for link in new_links:
                if link not in visited:
                    visited.add(link)
                    queue.append(link)
                    added_count += 1
                    # Update total for the progress bar dynamically
                    progress.update(task_id, total=len(visited))

            # Advance progress only after processing a URL and finding its links
            progress.advance(task_id)

            if added_count > 0:
                 logger.debug(f"Discovery: Added [bold]{added_count}[/] new links from [link={current_url}]{current_url}[/link]. Total unique: [bold]{len(visited)}[/]")
        else:
             # Advance progress even if fetch failed, as we processed the queue item
             progress.advance(task_id)

        # Optional delay to be polite
        # time.sleep(0.05) # Short delay

    # Log completion only if verbose
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"[bold cyan]Phase 1 Discovery complete.[/bold cyan] Found [bold]{len(visited)}[/] unique URLs.")
    return sorted(list(visited))


# --- Phase 2: Processing ---
def process_and_save_pages(urls_to_process, output_dir, max_pages, headers, progress):
    """Fetches, converts, and saves Markdown for the given URLs using Rich Progress."""
    pages_to_save_count = min(len(urls_to_process), max_pages)
    if pages_to_save_count == 0:
        logger.warning("No URLs to process or max_pages is 0.")
        return 0 # Return 0 pages saved

    # Log start only if verbose
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"[bold magenta]Phase 2:[/bold magenta] Processing and saving up to [bold]{pages_to_save_count}[/] pages...")

    pages_saved = 0
    # Add a task for the processing phase - set initial description
    task_id = progress.add_task("[magenta]Processing...", total=pages_to_save_count)

    for i, current_url in enumerate(urls_to_process):
        if pages_saved >= max_pages:
            logger.warning(f"Reached max_pages limit ([bold]{max_pages}[/]). Stopping processing.")
            break # Stop processing more URLs

        # Update the progress bar description to show the current URL
        # This makes the URL part of the progress bar line itself
        display_url = current_url
        if len(display_url) > 70: # Simple truncation for display
            display_url = display_url[:35] + "..." + display_url[-32:]
        progress.update(task_id, description=f"[magenta]Processing {i+1}/{pages_to_save_count}[/] [dim link={current_url}]{display_url}[/]")

        logger.debug(f"Processing URL ({i+1}/{len(urls_to_process)}): [link={current_url}]{current_url}[/link]") # Keep debug log

        # Fetch full content this time
        html_content = fetch_page(current_url, headers, get_content=True)

        if html_content:
            markdown_content = convert_to_markdown(html_content, current_url)
            saved_successfully = save_markdown(markdown_content, current_url, output_dir)

            if saved_successfully:
                pages_saved += 1
                # Advance progress only when a page is successfully saved
                progress.advance(task_id)
        else:
            logger.debug(f"Skipping save for [link={current_url}]{current_url}[/link] (no HTML content or fetch error)")
            # Do not advance progress bar if page wasn't saved

        # Optional delay
        # time.sleep(0.05)

    # Log completion only if verbose
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"[bold magenta]Phase 2 Processing complete.[/bold magenta] Successfully saved [bold]{pages_saved}[/] pages.")
    return pages_saved


# --- Command-Line Interface ---

if __name__ == "__main__":
    # --- Define the Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Crawl a website starting from a URL, convert pages to Markdown within the same domain, and save them. Performs a discovery phase first.",
        formatter_class=argparse.RawTextHelpFormatter, # Use RawTextHelpFormatter to preserve formatting in epilog
        epilog="""\
[bold]Examples:[/bold]\n
  [dim]# Process ONLY the single starting URL[/]\n
  [cyan]%(prog)s https://docs.example.com/intro[/]\n

  [dim]# Discover all pages, then crawl & save up to 50 pages to 'docs_md'[/]\n
  [cyan]%(prog)s --all https://docs.example.com/ -m 50 -o docs_md[/]\n

  [dim]# Discover and crawl all pages (up to default 1000 limit) with verbose logging[/]\n
  [cyan]%(prog)s --all https://anothersite.org/start -v[/]\n

  [dim]# Discover and crawl quietly (no progress bars or logs except errors)[/]\n
  [cyan]%(prog)s --all https://yetanothersite.com/ -q[/]\n

  [dim]# Use the standard html.parser instead of lxml[/]\n
  [cyan]%(prog)s --all https://someothersite.com/ --parser html.parser[/]\n
"""
    )

    # --- Define Arguments ---
    parser.add_argument(
        "start_url",
        metavar="URL",
        nargs='?', # Make start_url optional
        default=None, # Default to None if not provided
        help="The URL to process. By default, only this single page is processed. Use --all to crawl the entire site starting from this URL."
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
        default=1000, # Increased default max pages
        help="Maximum number of pages to *save* when using --all (default: 1000)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Crawl the entire website starting from the URL, instead of just processing the single URL."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress bars and warning/info/debug messages (show only errors)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed informational and debug logging messages (implies progress bars)."
    )
    # Add lxml dependency info to help
    try:
        import lxml
        default_parser = 'lxml'
        parser_help = "HTML parser ('lxml' or 'html.parser'). (default: lxml)"
    except ImportError:
        default_parser = 'html.parser'
        parser_help = "HTML parser ('html.parser' only; install 'lxml' for potential speed improvements). (default: html.parser)"
    parser.add_argument('--parser', default=default_parser, choices=['lxml', 'html.parser'], help=parser_help)


    # --- Initial Argument Check for Default Help ---
    args = parser.parse_args()

    if args.start_url is None:
         # Use Rich console to print help for better formatting if possible
         console.print(Panel(parser.format_help(), title="[bold]Help[/]", border_style="yellow"))
         sys.exit(0)


    # --- Adjust Logging Level Based on Flags ---
    log_level = logging.WARNING # Default
    is_quiet = args.quiet
    if args.verbose:
        log_level = logging.DEBUG # Show DEBUG, INFO, WARNING, ERROR
        is_quiet = False # Verbose overrides quiet for logging/progress
    elif args.quiet:
        log_level = logging.ERROR # Show only ERROR
        # Disable logging below ERROR level if quiet
        logging.disable(logging.WARNING)

    logger.setLevel(log_level)
    # RichHandler automatically respects the logger's level


    # --- Main Execution ---
    # Removed screen clear

    # Log start only if verbose
    logger.info("[bold green]Starting Markdown Crawler...[/]")
    logger.info(f"Target URL: [link={args.start_url}]{args.start_url}[/link]") # Log target URL
    if args.all:
        logger.info("Mode: [bold cyan]Full Site Crawl (--all)[/bold cyan]")
    else:
        logger.info("Mode: [bold cyan]Single Page Processing[/bold cyan]")

    if not is_valid_url(args.start_url):
        logger.error(f"Invalid start URL provided: {args.start_url}")
        sys.exit(1) # Exit if start URL is invalid

    target_domain = get_domain(args.start_url)
    if not target_domain:
        logger.error(f"Could not determine domain for start URL: {args.start_url}")
        sys.exit(1) # Exit if domain cannot be determined

    # Construct the domain-specific output directory
    domain_output_dir = os.path.join(args.output_dir, target_domain)
    logger.info(f"Output directory: [repr.filename]{os.path.abspath(domain_output_dir)}[/]")

    # Define headers used for both phases
    headers = {
        'User-Agent': f'SimpleMarkdownCrawler/2.6 (+https://github.com/your-repo/your-crawler)' # Version bump
        # Consider adding Accept-Language, etc.
        # 'Accept-Language': 'en-US,en;q=0.9',
        # 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }

    discovered_urls = []
    pages_actually_saved = 0

    # Define Rich Progress columns (used for both phases if --all is used)
    # Keep the description column simple here
    progress_columns = (
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
    )

    try:
        if args.all:
            # --- Phase 1 Execution (Full Crawl) ---
            # Use Progress context manager for discovery
            # Set transient=True so this bar disappears after completion
            with Progress(*progress_columns, console=console, disable=is_quiet, transient=True) as progress:
                discovered_urls = discover_all_links(args.start_url, target_domain, headers, args.parser, progress)

            # --- Phase 2 Execution (Full Crawl) ---
            if discovered_urls:
                # Add a print statement for separation if not quiet
                if not is_quiet:
                    console.print() # Print a blank line for separation

                # Use a *new* Progress context manager for processing
                # Set transient=False to keep this bar on screen after completion
                with Progress(*progress_columns, console=console, disable=is_quiet, transient=False) as progress:
                    # Pass the domain_output_dir here
                    pages_actually_saved = process_and_save_pages(discovered_urls, domain_output_dir, args.max_pages, headers, progress)
            else:
                # Log warning only if not quiet
                if not is_quiet:
                    logger.warning("No URLs were discovered via crawling. Nothing to process.")
        else:
            # --- Single Page Execution ---
            logger.info(f"Fetching single page: [link={args.start_url}]{args.start_url}[/link]")
            html_content = fetch_page(args.start_url, headers, get_content=True)
            if html_content:
                logger.info("Converting to Markdown...")
                markdown_content = convert_to_markdown(html_content, args.start_url)
                logger.info("Saving Markdown...")
                # Pass the domain_output_dir here
                saved_successfully = save_markdown(markdown_content, args.start_url, domain_output_dir)
                if saved_successfully:
                    pages_actually_saved = 1
                    logger.info("Markdown saved successfully.")
                else:
                     logger.error("Failed to save Markdown.")
            else:
                 logger.warning(f"Could not fetch or process HTML content from the URL: [link={args.start_url}]{args.start_url}[/link]")

        # Log finish only if verbose (and not quiet)
        if not is_quiet:
            logger.info("[bold green]Crawler finished normally.[/]")

    except KeyboardInterrupt:
        # Catch Ctrl+C gracefully
        console.print("\n[bold yellow]Process interrupted by user. Exiting.[/]")
        sys.exit(0) # Clean exit
    except Exception as e:
         # Log any other unexpected errors
         logger.exception("[bold red]An unexpected error occurred during the crawl:[/]")
         sys.exit(1) # Exit with error status


    # --- Final Confirmation Message ---
    # Show final message unless quiet mode is enabled
    if not is_quiet:
        # Use Rich Panel for a nicer final message
        summary_text = Text.assemble(
            ("Crawler run complete.\n", "white"),
        )
        if args.all:
            summary_text.append_text(Text.assemble(
                ("Discovered: ", "white"), (f"{len(discovered_urls)}", "bold cyan"), (" URLs\n", "white"),
            ))
        summary_text.append_text(Text.assemble(
            ("Saved: ", "white"), (f"{pages_actually_saved}", "bold magenta"), (" pages\n", "white"),
            # Update the output path in the summary
            ("Output: ", "white"), (f"'{os.path.abspath(domain_output_dir)}'", "bold green")
        ))

        console.print(Panel(summary_text, title="[bold]Summary[/]", border_style="blue", expand=False))


