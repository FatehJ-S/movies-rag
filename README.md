# movies-rag
This notebook uses the following 5 datasets from imdb, cleans them, and combines them into one dataset sorted by genre. 
- name.basics 
- title.basics
- title.ratings
- title.crew.tsv
- title.principals 
  
The datasets are downloaded as tab separated value files. The code removes nulls and duplicates from the files and drops uneeded columns. It also removes any rows that do not have any key information, such as a movie's title or release year. The 5 datasets are then combined into one sorted by genre

It uses sentence-transformers to embed the data locally and stores the embeddings in a FAISS index that is then downloaded to a file for later use.

When a query is sent, the notebook uses two functions, one to embed the query and retrieve closest data points from the FAISS index, and the other to generate a response using OpenAI.
