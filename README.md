# rfaiss


**rfaiss** is an experimental R wrapper around [FAISS](https://github.com/facebookresearch/faiss), Facebook AI’s library for efficient similarity search and clustering of dense vectors.  
It is designed for users who are already familiar with **Python embeddings workflows** (e.g., [sentence-transformers](https://www.sbert.net/)) and want to experiment with vector search inside R via **reticulate**.  

⚠️ **Important:**  
- This package is experimental and not intended for general use.  
- It is not on CRAN and likely never will be.  
- Expect rough edges, higher memory usage, and a dependency on a functioning Python + FAISS installation.  

---

## Why use rfaiss?  

- **Direct access to a true vector store**: Unlike some R-native approaches, FAISS is a production-grade library widely used in large-scale AI applications.  
- **Reticulate bridge**: Lets you tap into the FAISS Python ecosystem directly from R, without having to leave your R workflow.  
- **Good for experimentation**: Useful for prototyping retrieval-augmented generation (RAG) workflows in R when you already rely on embeddings generated with Python models.  

---

## Why *not* use rfaiss?  

- **Memory usage**: FAISS indexes are fast but RAM-hungry. Even small experiments can use a decent amount of memory.  
- **Experimental state**: API stability is not guaranteed.  
- **Alternative in R**: The tidyverse team is building [`ragnar`](https://ragnar.tidyverse.org/), which uses **DuckDB** for embedding workflows. This will likely be the long-term ecosystem standard for R users.  
- **CRAN status**: rfaiss will probably never be on CRAN, given its reliance on Python and FAISS.  

---

## Installation  

Because rfaiss depends on Python, FAISS, and reticulate, you must ensure these are available in your environment.  

```r
# Install from GitHub
remotes::install_github("lazasaurus-ai//rfaiss")

# Load the package
library(rfaiss)
```

On the Python side you need:

```python
pip install faiss-cpu sentence-transformers
```

## Example 

```r
library(rfaiss)

# Create some random embeddings
set.seed(123)
mat <- matrix(rnorm(100 * 128), nrow = 100, ncol = 128)

# Build an index
index <- faiss_index(mat)

# Query the index
query <- matrix(rnorm(1 * 128), nrow = 1)
faiss_search(index, query, k = 5)

```


## Notes

* Ensure your Python environment is correctly configured with reticulate::use_virtualenv() or reticulate::use_condaenv().

* Best used by those who already work with `sentence-transformers` in Python and simply want to experiment with FAISS from R.

* For R-native workflows, keep an eye on `ragnar`.

