import os
from dotenv import load_dotenv
import csv
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#load environment variables 
load_dotenv()
#split the input text into chunks of specified length


def split_into_chunks(text, chunk_size=1024):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i: i+chunk_size])

    return chunks


chunks = []



with open("./mini-llama-articles.csv", mode="r", encoding="utf-8") as file:
    csv_reader =csv.reader(file)


    for idx, row in enumerate(csv_reader):
        if idx == 0: continue
        chunks.extend(split_into_chunks(row[1]))

 

df = pd.DataFrame(chunks, columns=['chunk'])

client = OpenAI()

#define a function that convert text to embedding vector using openAI 'Ada model
def get_embedding(text):
    try:
        #remove new line
       text = text.replace('\n', ' ')
       res = client.embeddings.create(input= [text], model="text-embedding-3-small")
       return res.data[0].embedding
        
    except:
        return None
    
#Generate embeddings
print("Generating embeddings")
embeddings = []

for index, row in tqdm(df.iterrows(),total=len(df)):
  embeddings.append(get_embedding(row['chunk']))


#add the "embedding" column to the dataframe
embedding_values = pd.Series(embeddings)
df.insert(loc=1, column='embedding', value=embedding_values)

# Define the user question, and convert it to embedding.
QUESTION = "How many parameters does the LLaMA2 model have?"

QUESTION_EMB = get_embedding(QUESTION)
BAD_SOURCE_EMB = get_embedding("The Sky is Blue")
GOOD_SOURCE_EMB = get_embedding('LLaMA2 model has a total of 2B parameters.')

print("> Bad Response Score: ",cosine_similarity([QUESTION_EMB], [BAD_SOURCE_EMB]))
print("> Good response Score:", cosine_similarity([QUESTION_EMB], [GOOD_SOURCE_EMB]))

