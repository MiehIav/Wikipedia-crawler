# Ultimate Wikipedia Crawler

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

An advanced Wikipedia crawler that uses reinforcement learning to explore Wikipedia pages and provide comprehensive analytics.

## 🌟 Features

- 🕷️ Crawls Wikipedia pages starting from a given URL or a random page
- 🧠 Uses Q-learning to guide the crawling process
- 📊 Extracts various information from each page, including content, metadata, and link structure
- 😃 Performs sentiment analysis on the content of each page
- 🕸️ Creates a network graph of the crawled pages
- 📈 Generates various visualizations and analyses

## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- pip (Python package manager)
- Git

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ultimate-wikipedia-crawler.git
   cd ultimate-wikipedia-crawler
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install streamlit plotly transformers scikit-learn networkx pyvis nltk requests beautifulsoup4 pandas matplotlib seaborn wordcloud
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run ultimate_wikipedia_crawler.py
   ```

5. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## 🖥️ Usage

1. In the web interface, you can:
   - Enter a starting Wikipedia URL or leave it blank for a random start
   - Set the maximum number of pages to crawl (10-100)
   - Adjust the minimum and maximum time spent on each page

2. Click "Start Crawling" to begin the exploration process.

3. Monitor the crawling process:
   - Watch the real-time progress bar and status updates
   - View the list of processed pages as they are crawled

4. After crawling is complete, you'll see various analyses and visualizations:
   - Interactive exploration graph
   - Rewards per page visit chart
   - Topic model of explored content
   - Q-Table heatmap
   - Exploration summary
   - Word cloud of explored content
   - Category distribution
   - Sentiment distribution
   - Link distribution
   - Content length distribution
   - Reward components chart
   - Detailed page information for each visited page

## 🛠️ Key Components

- `UltimateWikipediaCrawler`: The main class that handles the crawling process and analysis
- Reinforcement Learning: Uses Q-learning to guide the crawling process
- Natural Language Processing: Utilizes NLTK and Hugging Face Transformers for text analysis and sentiment analysis
- Data Visualization: Employs Plotly, Matplotlib, and Streamlit for creating interactive charts and graphs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

If you have any questions or feedback, please reach out:

- Your Name - [your.email@example.com](mailto:your.email@example.com)
- Project Link: [https://github.com/yourusername/ultimate-wikipedia-crawler](https://github.com/yourusername/ultimate-wikipedia-crawler)

---

Happy crawling! 🕷️📚
