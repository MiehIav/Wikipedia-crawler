import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from pyvis.network import Network
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
import random
import time
import json
from datetime import datetime
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class UltimateWikipediaCrawler:
    def __init__(self, start_url, max_pages=50, page_time_range=(1, 5)):
        self.start_url = start_url
        self.max_pages = max_pages
        self.visited_pages = []
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1
        self.page_time_range = page_time_range
        self.graph = nx.DiGraph()
        self.session = requests.Session()
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

    def get_wikipedia_links(self, url):
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.find(id="mw-content-text")
            if not content_div:
                return [], "", "", {}

            title = soup.find(id="firstHeading").text.strip()
            paragraphs = content_div.find_all('p')
            full_content = ' '.join([p.text for p in paragraphs])
            snippet = paragraphs[0].text.strip() if paragraphs else "No content found"
            snippet = (snippet[:200] + '...') if len(snippet) > 200 else snippet

            links = content_div.find_all('a', href=True)
            wiki_links = [
                'https://en.wikipedia.org' + link['href']
                for link in links
                if link['href'].startswith('/wiki/')
                and ':' not in link['href']
                and 'Main_Page' not in link['href']
            ]

            infobox = soup.find('table', {'class': 'infobox'})
            infobox_data = {}
            if infobox:
                rows = infobox.find_all('tr')
                for row in rows:
                    header = row.find('th')
                    data = row.find('td')
                    if header and data:
                        infobox_data[header.text.strip()] = data.text.strip()

            page_stats = {
                'length': len(full_content),
                'internal_links': len(wiki_links),
                'external_links': len([link for link in links if link['href'].startswith('http') and 'wikipedia.org' not in link['href']])
            }

            metadata = {
                'last_modified': soup.find(id="footer-info-lastmod").text if soup.find(id="footer-info-lastmod") else "",
                'categories': [cat.text for cat in soup.find_all("div", {"class": "mw-normal-catlinks"})] if soup.find("div", {"class": "mw-normal-catlinks"}) else [],
                'references': len(soup.find_all("ol", {"class": "references"})) if soup.find("ol", {"class": "references"}) else 0,
                'images': len(soup.find_all("img")) if soup.find_all("img") else 0,
                'tables': len(soup.find_all("table")) if soup.find_all("table") else 0,
                'infobox': infobox_data,
                'page_stats': page_stats
            }

            return wiki_links, title, full_content, metadata
        except requests.RequestException as e:
            st.error(f"Error fetching {url}: {str(e)}")
            return [], "", "", {}

    def calculate_reward(self, content, metadata):
        length_reward = min(len(content) / 5000, 1)
        unique_words = set(word_tokenize(content.lower())) - self.stop_words
        diversity_reward = min(len(unique_words) / 500, 1)
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        sentiment_reward = abs(sentiment_result['score'] - 0.5) * 2
        ref_reward = min(metadata['references'] / 50, 1)
        media_reward = min((metadata['images'] + metadata['tables']) / 10, 1)
        
        if metadata['last_modified']:
            try:
                last_modified = datetime.strptime(metadata['last_modified'], "This page was last edited on %d %B %Y, at %H:%M.")
                days_since_modified = (datetime.now() - last_modified).days
                recency_reward = max(1 - (days_since_modified / 365), 0)
            except ValueError:
                recency_reward = 0
        else:
            recency_reward = 0

        total_reward = (length_reward + diversity_reward + sentiment_reward + ref_reward + media_reward + recency_reward) / 6
        return total_reward

    def choose_action(self, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            if state not in self.q_table:
                return random.choice(possible_actions)
            best_actions = []
            best_q_value = float('-inf')
            for action in possible_actions:
                q_value = self.q_table[state].get(action, 0)
                if q_value > best_q_value:
                    best_actions = [action]
                    best_q_value = q_value
                elif q_value == best_q_value:
                    best_actions.append(action)
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

    def explore_wikipedia(self):
        current_url = self.start_url
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(self.max_pages):
            links, title, content, metadata = self.get_wikipedia_links(current_url)
            if not links:
                break

            reward = self.calculate_reward(content, metadata)
            state = current_url.split('/')[-1]
            
            self.graph.add_node(state, title=title, url=current_url, reward=reward)
            if self.visited_pages:
                previous_state = self.visited_pages[-1]['url'].split('/')[-1]
                self.graph.add_edge(previous_state, state)
            
            self.visited_pages.append({
                'url': current_url,
                'title': title,
                'reward': reward,
                'content': content,
                'metadata': metadata
            })
            
            st.text(f"Processed page: {title} (Reward: {reward:.2f})")
            
            next_url = self.choose_action(state, links)
            self.update_q_value(state, next_url, reward, next_url.split('/')[-1])
            
            progress = (i + 1) / self.max_pages
            progress_bar.progress(progress)
            status_text.text(f"Processed {i + 1} out of {self.max_pages} pages")
            
            current_url = next_url
            time.sleep(random.uniform(*self.page_time_range))

        status_text.text(f"Exploration complete. Visited {len(self.visited_pages)} pages.")
        self.run_analysis()

    def create_interactive_graph(self):
        net = Network(notebook=True, height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        for node, data in self.graph.nodes(data=True):
            size = 10 + data['reward'] * 20
            color = f"rgb({min(255, int(data['reward']*255))}, {min(255, int(255 - data['reward']*255))}, 0)"
            net.add_node(node, label=data['title'], title=f"URL: {data['url']}\nReward: {data['reward']:.2f}", 
                         size=size, color=color)
        
        for edge in self.graph.edges():
            net.add_edge(edge[0], edge[1])
        
        net.show_buttons(filter_=['physics'])
        html = net.generate_html()
        st.subheader("Interactive Exploration Graph")
        st.components.v1.html(html, height=600)

    def create_reward_chart(self):
        rewards = [page['reward'] for page in self.visited_pages]
        fig = go.Figure(data=go.Scatter(x=list(range(len(rewards))), y=rewards, mode='lines+markers'))
        fig.update_layout(title="Rewards per Page Visit", xaxis_title="Visit Order", yaxis_title="Reward")
        st.plotly_chart(fig)

    def create_topic_model(self):
        documents = [page['content'] for page in self.visited_pages]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        nmf = NMF(n_components=5, random_state=42)
        nmf.fit(doc_term_matrix)
        
        topics = []
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(nmf.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words))
        
        st.subheader("Topic Model")
        for topic in topics:
            st.write(topic)

    def create_q_table_heatmap(self):
        states = list(self.graph.nodes())
        q_values = [[self.q_table.get(state, {}).get(action, 0) for action in states] for state in states]
        fig = go.Figure(data=go.Heatmap(z=q_values, x=states, y=states, colorscale="YlGnBu"))
        fig.update_layout(title="Q-Table Heatmap", xaxis_title="Actions (Next States)", yaxis_title="Current States")
        st.plotly_chart(fig)

    def get_most_common_categories(self):
        all_categories = [cat for page in self.visited_pages for cat in page['metadata']['categories']]
        return Counter(all_categories).most_common(10)

    def get_sentiment_distribution(self):
        sentiments = [self.sentiment_analyzer(page['content'][:512])[0]['label'] for page in self.visited_pages]
        return dict(Counter(sentiments))

    def get_content_length_stats(self):
        lengths = [len(page['content']) for page in self.visited_pages]
        return {
            "min": min(lengths),
            "max": max(lengths),
            "average": sum(lengths) / len(lengths)
        }

    def create_exploration_summary(self):
        summary = {
            "total_pages_visited": len(self.visited_pages),
            "average_reward": sum(page['reward'] for page in self.visited_pages) / len(self.visited_pages),
            "most_rewarding_page": max(self.visited_pages, key=lambda x: x['reward'])['title'],
            "least_rewarding_page": min(self.visited_pages, key=lambda x: x['reward'])['title'],
            "average_references": sum(page['metadata']['references'] for page in self.visited_pages) / len(self.visited_pages),
            "average_images": sum(page['metadata']['images'] for page in self.visited_pages) / len(self.visited_pages),
            "average_tables": sum(page['metadata']['tables'] for page in self.visited_pages) / len(self.visited_pages),
            "most_common_categories": self.get_most_common_categories(),
            "sentiment_distribution": self.get_sentiment_distribution(),
            "content_length_stats": self.get_content_length_stats(),
            "average_internal_links": sum(page['metadata']['page_stats']['internal_links'] for page in self.visited_pages) / len(self.visited_pages),
            "average_external_links": sum(page['metadata']['page_stats']['external_links'] for page in self.visited_pages) / len(self.visited_pages),
        }
        
        st.subheader("Exploration Summary")
        st.json(summary)

    def create_word_cloud(self):
        all_text = ' '.join([page['content'] for page in self.visited_pages])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Explored Content')
        st.pyplot(plt)

    def create_category_distribution(self):
        all_categories = [cat for page in self.visited_pages for cat in page['metadata']['categories']]
        category_counts = Counter(all_categories)
        
        df = pd.DataFrame.from_dict(category_counts, orient='index', columns=['count']).reset_index()
        df = df.sort_values('count', ascending=False).head(10)
        
        fig = px.bar(df, x='index', y='count', title='Top 10 Categories')
        st.plotly_chart(fig)

    def create_sentiment_distribution(self):
        sentiments = [self.sentiment_analyzer(page['content'][:512])[0]['label'] for page in self.visited_pages]
        sentiment_counts = Counter(sentiments)
        
        fig = px.pie(values=list(sentiment_counts.values()), names=list(sentiment_counts.keys()), title='Sentiment Distribution')
        st.plotly_chart(fig)

    def create_link_distribution(self):
        internal_links = [page['metadata']['page_stats']['internal_links'] for page in self.visited_pages]
        external_links = [page['metadata']['page_stats']['external_links'] for page in self.visited_pages]
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=internal_links, name='Internal Links'))
        fig.add_trace(go.Box(y=external_links, name='External Links'))
        fig.update_layout(title='Distribution of Internal and External Links', yaxis_title='Number of Links')
        st.plotly_chart(fig)

    def create_content_length_distribution(self):
        lengths = [page['metadata']['page_stats']['length'] for page in self.visited_pages]
        fig = go.Figure(data=[go.Histogram(x=lengths)])
        fig.update_layout(title='Distribution of Page Lengths', xaxis_title='Length (characters)', yaxis_title='Count')
        st.plotly_chart(fig)

    def create_reward_components_chart(self):
        data = []
        for page in self.visited_pages:
            content = page['content']
            metadata = page['metadata']
            length_reward = min(len(content) / 5000, 1)
            unique_words = set(word_tokenize(content.lower())) - self.stop_words
            diversity_reward = min(len(unique_words) / 500, 1)
            sentiment_result = self.sentiment_analyzer(content[:512])[0]
            sentiment_reward = abs(sentiment_result['score'] - 0.5) * 2
            ref_reward = min(metadata['references'] / 50, 1)
            media_reward = min((metadata['images'] + metadata['tables']) / 10, 1)
            
            data.append({
                'Page': page['title'],
                'Length': length_reward,
                'Diversity': diversity_reward,
                'Sentiment': sentiment_reward,
                'References': ref_reward,
                'Media': media_reward
            })
        
        df = pd.DataFrame(data)
        fig = px.bar(df, x='Page', y=['Length', 'Diversity', 'Sentiment', 'References', 'Media'],
                     title='Reward Components for Each Page')
        st.plotly_chart(fig)

    def create_network_metrics(self):
        st.subheader("Network Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Nodes", len(self.graph.nodes()))
        with col2:
            st.metric("Number of Edges", len(self.graph.edges()))
        with col3:
            st.metric("Average Degree", sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()))

        centrality = nx.degree_centrality(self.graph)
        most_central = max(centrality, key=centrality.get)
        st.write(f"Most central page: {self.graph.nodes[most_central]['title']}")

    def display_page_details(self):
        st.subheader("Detailed Page Information")
        for page in self.visited_pages:
            with st.expander(f"{page['title']} (Reward: {page['reward']:.2f})"):
                st.write(f"URL: {page['url']}")
                st.write(f"Length: {page['metadata']['page_stats']['length']} characters")
                st.write(f"Internal Links: {page['metadata']['page_stats']['internal_links']}")
                st.write(f"External Links: {page['metadata']['page_stats']['external_links']}")
                st.write(f"References: {page['metadata']['references']}")
                st.write(f"Images: {page['metadata']['images']}")
                st.write(f"Tables: {page['metadata']['tables']}")
                
                if page['metadata']['infobox']:
                    st.write("Infobox Data:")
                    for key, value in page['metadata']['infobox'].items():
                        st.write(f"  {key}: {value}")
                
                st.write("Categories:")
                for category in page['metadata']['categories']:
                    st.write(f"  - {category}")
                
                sentiment = self.sentiment_analyzer(page['content'][:512])[0]
                st.write(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")

    def run_analysis(self):
        st.title("Wikipedia Crawler Analysis")

        self.create_network_metrics()
        self.create_interactive_graph()
        self.create_reward_chart()
        self.create_topic_model()
        self.create_q_table_heatmap()
        self.create_exploration_summary()
        self.create_word_cloud()
        self.create_category_distribution()
        self.create_sentiment_distribution()
        self.create_link_distribution()
        self.create_content_length_distribution()
        self.create_reward_components_chart()
        self.display_page_details()

def main():
    st.set_page_config(page_title="Ultimate Wikipedia Crawler", layout="wide")
    st.title("Ultimate Wikipedia Crawler")
    
    start_url = st.text_input("Enter starting Wikipedia URL (leave blank for random)", "https://en.wikipedia.org/wiki/Special:Random")
    max_pages = st.slider("Maximum number of pages to crawl", 10, 100, 50)
    min_time = st.slider("Minimum time on page (seconds)", 1, 10, 1)
    max_time = st.slider("Maximum time on page (seconds)", min_time, 20, 5)
    
    if st.button("Start Crawling"):
        crawler = UltimateWikipediaCrawler(start_url, max_pages=max_pages, page_time_range=(min_time, max_time))
        crawler.explore_wikipedia()

if __name__ == "__main__":
    main()