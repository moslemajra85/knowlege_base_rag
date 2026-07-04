import os
from dotenv import load_dotenv
import csv
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# load environment variables
load_dotenv()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

# split the input text into chunks of specified length


def split_into_chunks(text, chunk_size=1024):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i: i+chunk_size])

    return chunks


chunks = []


with open("./mini-llama-articles.csv", mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)


    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue
        chunks.extend(split_into_chunks(row[1]))

 

df = pd.DataFrame(chunks, columns=['chunk'])

client = OpenAI()

# define a function that converts text to an embedding vector
def get_embedding(text):
    try:
        # remove new lines
        text = text.replace('\n', ' ')
        res = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return res.data[0].embedding
        
    except Exception as e:
        raise RuntimeError(f"Failed to create embedding: {e}") from e
    
#Generate embeddings
print("Generating embeddings")
embeddings = []

for index, row in tqdm(df.iterrows(), total=len(df)):
    embeddings.append(get_embedding(row['chunk']))


#add the "embedding" column to the dataframe
embedding_values = pd.Series(embeddings)
df.insert(loc=1, column='embedding', value=embedding_values)

# Define the user question, and convert it to embedding.
QUESTION = "How many parameters does the LLaMA2 model have?"

QUESTION_EMB = get_embedding(QUESTION)
# BAD_SOURCE_EMB = get_embedding("The Sky is Blue")
# GOOD_SOURCE_EMB = get_embedding('LLaMA2 model has a total of 2B parameters.')

# print("> Bad Response Score: ",cosine_similarity([QUESTION_EMB], [BAD_SOURCE_EMB]))
# print("> Good response Score:", cosine_similarity([QUESTION_EMB], [GOOD_SOURCE_EMB]))

# The similarity between the questions and each part of the essay.
similarities  = cosine_similarity([QUESTION_EMB], df['embedding'].tolist())

#print(similarities)
# number of chunks to retrieve
number_of_chunks_to_retrieve = 3

# Sort and find the index of N highest scored chunks
indices = np.argsort(similarities[0])[::-1][:number_of_chunks_to_retrieve]

#print(indices)


# formulate the system prompt and condition the model to answer only AI-related questions
system_prompt = (
    "You are an assistant and an expert in answering questions from chunks of content. "
    "Only answer AI-related questions. Otherwise, say that you cannot answer the question."
)

#create a user prompt with user's question
prompt = (
        "Read the following informations that might contain the context you require to answer the question. You can use the informations starting from the <START_OF_CONTEXT> tag and end with the <END_OF_CONTEXT> tag. Here is the content:\n\n<START_OF_CONTEXT>\n{}\n<END_OF_CONTEXT>\n\n"
        "Please provide an informative and accurate answer to the following question based on the avaiable context. Be concise and take your time. \nQuestion: {}\nAnswer:"
    )

# Add the retrieved pieces of text to the prompt.
retrieved_context = " ".join(df.iloc[indices]["chunk"])
prompt = prompt.format(retrieved_context, QUESTION)

response = client.chat.completions.create(
    model=OPENAI_CHAT_MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
)

print(response.choices[0].message.content)
