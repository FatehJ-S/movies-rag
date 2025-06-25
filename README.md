# movies-rag
This notebook uses the following 5 datasets from imdb, cleans them, and combines them into one dataset sorted by genre. 
  name.basics 
  title.basics
  title.ratings
  title.crew.tsv
  title.principals 
  
The datasets are downloaded as tab separated value files. The code removes nulls and duplicates from the files and drops uneeded columns. It also removes any rows that do not have any key information, such as a movie's title or release year. The 5 datasets are then combined into one sorted by genre
