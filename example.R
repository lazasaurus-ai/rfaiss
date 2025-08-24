# 0. (optional) ensure .Renviron has RETICULATE_PYTHON and AWS creds set
# Sys.setenv(RETICULATE_PYTHON = "/usr/local/bin/python")  # only if you need to override for this session

# 1. Rebuild docs & load package so new functions are available
devtools::document()
devtools::load_all()   # now build_kb(), build_flatl2_from_dir(), rag_query_aws(), etc. should be available

# 2. Build KB next to demo/ (will create demo/kb_TIMESTAMP.faiss and _manifest.rds)
res <- build_kb("demo", use_tmp = TRUE, readme = TRUE)
res$index_path
res$manifest_path
res$readme_path
# You can view the generated README:
cat(readLines(res$readme_path), sep = "\n")

# 3. Quick retrieval smoke test (search_dir_index)
sr <- search_dir_index(index_path = res$index_path,
                       manifest_path = res$manifest_path,
                       queries = "eligibility criteria",
                       k = 5)
head(sr)

# 4. Run a RAG query through AWS Bedrock (via ellmer)
#    If you want to pass an explicit Bedrock model, include bedrock_args = list(model = "...").
#    rag_query_aws will require AWS env vars in .Renviron (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).
ans <- rag_query_aws("What is tidymodels?",
                     index_path = res$index_path,
                     manifest_path = res$manifest_path,
                     k = 5,
                     embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
                     python = Sys.getenv("RETICULATE_PYTHON", unset = NA_character_),
                     bedrock_args = list(model = "anthropic.claude-3-7-sonnet-20250219-v1:0"),
                     max_context_chars = 3000)
cat("=== ANSWER ===\n")
cat(ans$answer, "\n\n")
print(ans$used)
