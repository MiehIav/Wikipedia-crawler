# Ultimate Wikipedia Crawler

This is an advanced Wikipedia crawler that uses reinforcement learning to explore Wikipedia pages and provide comprehensive analytics.

## Features

- Crawls Wikipedia pages starting from a given URL or a random page
- Uses Q-learning to guide the crawling process
- Extracts various information from each page, including content, metadata, and link structure
- Performs sentiment analysis on the content of each page
- Creates a network graph of the crawled pages
- Generates various visualizations and analyses

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ultimate-wikipedia-crawler.git
   cd ultimate-wikipedia-crawler
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```
streamlit run ultimate_wikipedia_crawler.py
```

This will open a web interface where you can:
- Enter a starting Wikipedia URL (or use a random one)
- Set the maximum number of pages to crawl
- Adjust the time range for each page visit
- Start the crawling process
- View real-time progress and results

## License

This project is licensed under the MIT License.
