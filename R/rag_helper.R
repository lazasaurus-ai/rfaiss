# R/rag_helper.R
# Retrieval-augmented-generation helper using Ellmer + AWS Bedrock
#
# This file provides:
#  - assemble_context(): merge retrieved snippets into a single promptable context
#  - build_rag_prompt(): create a clear instruction + context + question prompt
#  - rag_query_aws(): run retrieval (search_dir_index) and call AWS Bedrock via ellmer
#
# The implementation assumes you already have:
#  - a FAISS index and manifest (produced by build_flatl2_from_dir() or build_kb())
#  - the 'ellmer' package installed & configured to talk to Bedrock (AWS creds in env)
#  - a working embedding backend (.embed_texts) and search_dir_index() present

#' Assemble document context from retrieved snippets
#'
#' Collect retrieved snippets in order and join them into a single context string
#' until a character budget is reached.
#'
#' @param hits data.frame with columns file, chunk_id, snippet, rank, distance (rows ordered by preference)
#' @param max_chars integer approximate character budget for assembled context
#' @param sep string separator inserted between snippets in the assembled context
#' @return list(text = character(1) assembled context, used_hits = data.frame of used rows)
#' @export
assemble_context <- function(hits, max_chars = 3000L, sep = "\n\n---\n\n") {
  if (is.null(hits) || nrow(hits) == 0L) return(list(text = "", used_hits = hits))
  # ensure predictable columns and order
  if (!all(c("file", "chunk_id", "snippet") %in% names(hits))) {
    stop("`hits` must include columns: file, chunk_id, snippet")
  }
  
  used <- hits[0, , drop = FALSE]
  ctx_parts <- character(0)
  total <- 0L
  
  for (i in seq_len(nrow(hits))) {
    s <- as.character(hits$snippet[i])
    if (is.na(s) || nchar(s, type = "chars") == 0L) next
    len <- nchar(s, type = "chars")
    # if adding this snippet would exceed budget
    if ((total + len) > max_chars) {
      if (total == 0L) {
        # if nothing added yet, truncate current snippet to budget
        s_trunc <- substr(s, 1, max_chars)
        ctx_parts <- c(ctx_parts, s_trunc)
        used <- rbind(used, hits[i, , drop = FALSE])
      }
      break
    }
    ctx_parts <- c(ctx_parts, s)
    used <- rbind(used, hits[i, , drop = FALSE])
    total <- total + len
  }
  
  list(text = paste(ctx_parts, collapse = sep), used_hits = used)
}

#' Build a RAG prompt instructing the model to use the provided context
#'
#' @param query character scalar (user question)
#' @param context_text character scalar (assembled context)
#' @param instruct optional extra instructions appended to the system instructions
#' @return single string prompt
#' @export
build_rag_prompt <- function(query, context_text, instruct = NULL) {
  if (length(query) != 1L) stop("query must be a single string")
  if (is.null(context_text)) context_text <- ""
  
  pre <- paste(
    "You are given contextual excerpts from a set of documents below.",
    "Use only the information provided when answering the question.",
    "If the answer is not contained in the context, say you don't know rather than inventing facts.",
    "When you quote or refer to items, mention the source file and chunk id.",
    sep = " "
  )
  
  if (!is.null(instruct) && nzchar(instruct)) {
    pre <- paste(pre, instruct, sep = "\n\n")
  }
  
  paste0(
    pre,
    "\n\nContext:\n",
    context_text,
    "\n\nQuestion:\n",
    query,
    "\n\nAnswer (be concise, and cite sources like: file=<filename> chunk_id=<n>):"
  )
}

#' Run a RAG-style query using a saved index + manifest and AWS Bedrock via ellmer
#'
#' Defaults to `bedrock_args$model = "anthropic.claude-3-7-sonnet-20250219-v1:0"` if not provided.
#'
#' @param query character scalar
#' @param index_path path to .faiss index
#' @param manifest_path path to corresponding _manifest.rds (contains file, chunk_id, snippet)
#' @param k integer how many retrievals to request from the index (per query)
#' @param embedding_model embedding model used to build index (passed to search_dir_index)
#' @param python optional python binary path passed to embedder functions (NA or NULL => let reticulate choose)
#' @param bedrock_args list of args passed to ellmer::chat_aws_bedrock(...) (e.g., model, api_key, etc.)
#' @param max_context_chars approximate char budget for the concatenated context
#' @param system_prompt optional system prompt to include before the generated prompt (prepended to prompt body)
#' @return list(answer = character, used = data.frame of used hits, raw_search = full search results)
#' @export
rag_query_aws <- function(query,
                          index_path,
                          manifest_path,
                          k = 5L,
                          embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
                          python = Sys.getenv("RETICULATE_PYTHON", unset = NA_character_),
                          bedrock_args = list(),
                          max_context_chars = 3000L,
                          system_prompt = NULL) {
  stopifnot(length(query) == 1L)
  stopifnot(file.exists(index_path), file.exists(manifest_path))
  
  # check ellmer availability
  if (!requireNamespace("ellmer", quietly = TRUE)) {
    stop("ellmer package required for rag_query_aws; install it first (install.packages('ellmer') or from source).")
  }
  
  # required AWS env vars (Bedrock client will typically read these)
  needed_env <- c("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
  missing_env <- needed_env[nzchar(Sys.getenv(needed_env)) == FALSE]
  if (length(missing_env) > 0) {
    stop("Missing AWS env vars required for Bedrock: ", paste(missing_env, collapse = ", "),
         ". Please set them in your .Renviron (or environment) before calling rag_query_aws().")
  }
  
  # default bedrock model if not provided
  if (is.null(bedrock_args$model)) {
    bedrock_args$model <- "anthropic.claude-3-7-sonnet-20250219-v1:0"
  }
  
  # run retrieval using the search helper (search_dir_index must be present in the namespace)
  search_res <- tryCatch(
    search_dir_index(index_path = index_path, manifest_path = manifest_path,
                     queries = query, k = as.integer(k),
                     model = embedding_model, python = python),
    error = function(e) stop("Retrieval failed: ", conditionMessage(e))
  )
  
  if (nrow(search_res) == 0L) {
    return(list(answer = NA_character_, used = search_res, raw_search = search_res))
  }
  
  # dedupe/unique hits while preserving order and stable ranking (rank then distance)
  uniq_cols <- c("file", "chunk_id", "snippet", "distance", "rank", "query_id")
  uniq_hits <- unique(search_res[, uniq_cols, drop = FALSE])
  uniq_hits <- uniq_hits[order(uniq_hits$rank, uniq_hits$distance), , drop = FALSE]
  
  # assemble context up to budget
  ctx <- assemble_context(uniq_hits, max_chars = max_context_chars)
  ctx_text <- ctx$text
  used_hits <- ctx$used_hits
  
  if (nchar(ctx_text) == 0L) {
    return(list(answer = "No relevant context found.", used = used_hits, raw_search = search_res))
  }
  
  # build the prompt
  prompt_body <- build_rag_prompt(query = query, context_text = ctx_text, instruct = NULL)
  if (!is.null(system_prompt) && nzchar(system_prompt)) {
    full_prompt <- paste0("SYSTEM: ", system_prompt, "\n\n", prompt_body)
  } else {
    full_prompt <- prompt_body
  }
  
  # create chat object using ellmer::chat_aws_bedrock and provided bedrock_args
  chat_obj <- tryCatch({
    do.call(ellmer::chat_aws_bedrock, bedrock_args)
  }, error = function(e) {
    stop("Failed to create AWS Bedrock chat object via ellmer::chat_aws_bedrock(): ", conditionMessage(e))
  })
  
  # call LLM (defensive coercion of the response into a string)
  answer <- tryCatch({
    if (!is.null(chat_obj$chat) && is.function(chat_obj$chat)) {
      res <- chat_obj$chat(full_prompt)
      # coerce common return shapes into a character string
      if (is.list(res) && length(res) == 1 && !is.null(res[[1]])) {
        res_text <- as.character(res[[1]])
      } else if (is.character(res) && length(res) >= 1) {
        res_text <- paste(res, collapse = "\n")
      } else if (is.list(res) && !is.null(res$content)) {
        res_text <- as.character(res$content)
      } else {
        res_text <- as.character(res)
      }
    } else {
      stop("chat object does not expose a usable $chat() method.")
    }
    res_text
  }, error = function(e) {
    stop("LLM call failed: ", conditionMessage(e))
  })
  
  list(answer = answer, used = used_hits, raw_search = search_res)
}
