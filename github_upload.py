import os
import subprocess
import requests

def create_readme():
    readme_content = """# Ultimate Wikipedia Crawler

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
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
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
"""
    with open("README.md", "w") as f:
        f.write(readme_content)

def create_requirements():
    requirements = [
        "streamlit",
        "plotly",
        "transformers",
        "scikit-learn",
        "networkx",
        "pyvis",
        "nltk",
        "requests",
        "beautifulsoup4",
        "pandas",
        "matplotlib",
        "seaborn",
        "wordcloud"
    ]
    with open("requirements.txt", "w") as f:
        for req in requirements:
            f.write(f"{req}\n")

def create_gitignore():
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/

# Streamlit
.streamlit/

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# Operating System Files
.DS_Store
Thumbs.db
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)

def init_git_repo():
    subprocess.run(["git", "init"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Initial commit"])

def create_github_repo(token, repo_name):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": repo_name,
        "description": "An advanced Wikipedia crawler using reinforcement learning",
        "private": False
    }
    response = requests.post("https://api.github.com/user/repos", headers=headers, json=data)
    if response.status_code == 201:
        return response.json()["clone_url"]
    else:
        raise Exception(f"Failed to create repository: {response.content}")

def push_to_github(repo_url):
    subprocess.run(["git", "remote", "add", "origin", repo_url])
    subprocess.run(["git", "push", "-u", "origin", "master"])

def main():
    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    create_readme()
    create_requirements()
    create_gitignore()
    init_git_repo()

    github_token = input("Enter your GitHub personal access token: ")
    repo_name = input("Enter the name for your new GitHub repository: ")

    repo_url = create_github_repo(github_token, repo_name)
    push_to_github(repo_url)

    print(f"Successfully uploaded to GitHub: {repo_url}")
    print("Remember to update the clone URL in the README.md file.")

if __name__ == "__main__":
    main()