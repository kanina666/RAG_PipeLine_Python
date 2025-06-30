Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using the StackSample dataset (a 10% sample of Stack Overflow data). The goal is to build an end-to-end system that retrieves relevant information from a knowledge base and uses a language model to generate accurate, context-aware answers to user questions.

The pipeline is developed incrementally through lab assignments 3 to 6, resulting in a working prototype that:

1. Performs semantic search in a vector database.

2. Constructs a prompt based on retrieved context and user query.

3. Uses a large language model (LLM) to generate a final response.

Development Stages
Lab 3: Data Preprocessing and Exploration(data_analysis.ipynb)
- Worked with the StackSample dataset
- 
- Cleaned raw text data by removing HTML, Markdown, code snippets, and noise

- Normalized, tokenized, and lemmatized question and answer texts

- Merged questions with their answers

- Aggregated statistics such as most common tags and question length distribution

- Performed basic data visualization

Lab 4: Classical Information Retrieval(classis_search.ipynb)
- Built an inverted index from cleaned questions

- Implemented keyword-based search using TF-IDF 

- Compared user queries with dataset questions using cosine similarity and ranking

- Retrieved top matching questions and their accepted or highest-rated answers

 - Evaluated retrieval quality using simple metrics like Precision@k

Lab 5: Semantic Search with Embeddings(semantic_search.ipynb)
- Used pretrained models SentenceTransformers to compute vector embeddings

- Stored embeddings in a Chroma vector database

- Implemented nearest neighbor search using cosine similarity

- Retrieved relevant documents based on semantic closeness rather than keyword overlap

- Compared performance against classical search

Lab 6: Full Retrieval-Augmented Generation Pipeline(RAG_pipeline.ipynb)
- Combined semantic search with a generative LLM

 - Used Google AI Studio API for language generation

On user input:

 - Encoded the query into an embedding

 - Retrieved top-3 relevant Q&A entries from Chroma

 - Constructed a prompt with retrieved context and user question

 - Sent prompt to LLM and received generated response

- Built a complete loop: input → retrieval → generation → output

- Evaluated effectiveness of the final system and documented key findings

Technologies Used

- Python 3.9+

- Pandas, NLTK, scikit-learn, Matplotlib, seaborn, torch
 
- SentenceTransofrmer
 
- Chroma (vector DB)

- Google AI Studio API
