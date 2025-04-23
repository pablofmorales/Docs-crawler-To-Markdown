# Doc Crawler

| Simple a small functionality that goes over a documentation website and save the information in markdown format ready to add to your LLM chat, to give newer context in case that the model is outdated

## How to install

- Clone this repository

- Create the environment

```bash

python3 -m venv .venv
```

```bash
source .venv/bin/activate

```

- Install the dependencies

```bash
pip install -r requirements.txt

```

- Lastly add execution permissions for the script

```bash
chmod +x crawl_to_markdown.py
```

## Usage

Simple run the command with the URL as a parameter for example

```bash
./crawl_to_markdown.py https://docs.python.org/3/library/index.html
```

This will generate a folder with all the documentation in \*.md for you to use in your LLMs
