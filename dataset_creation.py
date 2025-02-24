import os
import pandas as pd
from bs4 import BeautifulSoup
from keybert import KeyBERT  
from tqdm import tqdm 

kw_model = KeyBERT()
xml_dir = "./blogs"
data = []

# Function to extract text from XML safely
def extract_text_from_xml(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()
    
    soup = BeautifulSoup(content, "lxml-xml") 
    posts = soup.find_all("post")

    return " ".join(post.text.strip() for post in posts if post.text).strip()  # Ensure no empty content

# Function to generate title from content
def generate_title(text):
    return text.split(".")[0] if "." in text else text[:50]  # First sentence or first 50 chars

# Function to extract keywords
def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=3)
    return ", ".join([kw[0] for kw in keywords])

# Get a list of XML files
xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]

# Initialize counter
processed_files = 0

# Process each XML with a progress bar
for file_name in tqdm(xml_files, desc="Processing Blogs", unit="file"):
    if len(data) >= 2100:
        break  

    file_path = os.path.join(xml_dir, file_name)
    blog_content = extract_text_from_xml(file_path)

    if not blog_content:  
        continue

    domain = extract_domain(file_name)
    title = generate_title(blog_content)
    keywords = extract_keywords(blog_content)

    data.append([title, keywords, blog_content, domain])
    processed_files += 1  

df = pd.DataFrame(data, columns=["Title", "Keyword", "Content"])

csv_filename = "processed_2000_blogs.csv"
df.to_csv(csv_filename, index=False)

print(f"\n Created {csv_filename} successfully with {len(df)} rows!")
