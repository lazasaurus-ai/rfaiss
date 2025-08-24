#' Search a saved directory index with text queries
#'
#' Compatible with the previous API but adds optional reranking strategies:
#'  - rerank = "faiss" (default): use FAISS results as-is
#'  - rerank = "emb": rerank FAISS top-k by exact saved-embeddings cosine
#'  - rerank = "mmr": MMR rerank using saved embeddings (reduces redundancy)
#'  - rerank = "llm": use an ellmer chat object as a reranker (requires llm_chat)
#'
#' @param index_path path to .faiss file
#' @param manifest_path path to corresponding _manifest.rds
#' @param queries character vector
#' @param k neighbors per query
#' @param model embedding model (should match one used to build)
#' @param python optional python binary path (character) or NULL. If omitted/NA, we pass NULL to reticulate.
#' @param use_cosine Logical; if TRUE will call index_search_cosine() (default FALSE). Kept for backwards compatibility.
#' @param rerank One of "faiss" (default), "emb", "mmr", "llm"
#' @param embeddings_path optional path to saved embeddings.rds (if NULL we'll try to infer from index_path)
#' @param mmr_lambda lambda for MMR (0..1)
#' @param llm_chat optional ellmer chat object (required when rerank = "llm")
#' @return data.frame with one row per query*k result, columns: query_id, rank, distance, file, chunk_id, snippet
#' @export
search_dir_index <- function(index_path,
                             manifest_path,
                             queries,
                             k = 5L,
                             model = "sentence-transformers/all-MiniLM-L6-v2",
                             python = Sys.getenv("RETICULATE_PYTHON", unset = NA_character_),
                             use_cosine = FALSE,
                             rerank = c("faiss", "emb", "mmr", "llm"),
                             embeddings_path = NULL,
                             mmr_lambda = 0.5,
                             llm_chat = NULL) {
  rerank <- match.arg(rerank)
  
  stopifnot(is.character(queries), length(queries) > 0)
  stopifnot(file.exists(index_path))
  stopifnot(file.exists(manifest_path))
  
  # normalize python arg: reticulate expects NULL not NA
  if (is.na(python) || !nzchar(python)) python <- NULL
  
  man <- readRDS(manifest_path)
  # manifest must have at least file, chunk_id, snippet (if not present we create empty columns)
  if (!"file" %in% names(man)) man$file <- NA_character_
  if (!"chunk_id" %in% names(man)) man$chunk_id <- NA_integer_
  if (!"snippet" %in% names(man)) man$snippet <- NA_character_
  
  h <- load_index(index_path)
  
  # get embeddings for queries
  qemb <- .embed_texts(queries, model = model, python = python)
  # ensure the embedder returned rows == length(queries)
  if (nrow(qemb) != length(queries)) {
    stop(sprintf("Embedding error: expected %d embeddings (rows) but got %d", length(queries), nrow(qemb)))
  }
  
  # run search (choose cosine vs L2) — this gets top-k from FAISS
  if (use_cosine) {
    res <- index_search_cosine(h, qemb, k = as.integer(k))
  } else {
    res <- index_search(h, qemb, k = as.integer(k))
  }
  
  I <- as.matrix(res$indices)
  D <- as.matrix(res$distances)
  
  n_queries <- nrow(I)
  k_ret <- ncol(I)
  
  # If no advanced rerank requested, produce result like before
  if (rerank == "faiss") {
    rows_list <- vector("list", n_queries)
    for (i in seq_len(n_queries)) {
      inds <- I[i, ]
      dists <- D[i, ]
      # coerce indices and mark invalid
      inds_int <- suppressWarnings(as.integer(inds))
      inds_int[is.na(inds_int) | inds_int <= 0 | inds_int > nrow(man)] <- NA_integer_
      
      rows <- data.frame(
        query_id = rep(i, k_ret),
        rank = seq_len(k_ret),
        distance = as.numeric(dists),
        file = rep(NA_character_, k_ret),
        chunk_id = rep(NA_integer_, k_ret),
        snippet = rep(NA_character_, k_ret),
        stringsAsFactors = FALSE
      )
      
      valid_mask <- !is.na(inds_int)
      if (any(valid_mask)) {
        rows$file[valid_mask] <- man$file[inds_int[valid_mask]]
        rows$chunk_id[valid_mask] <- man$chunk_id[inds_int[valid_mask]]
        rows$snippet[valid_mask] <- man$snippet[inds_int[valid_mask]]
      }
      rows_list[[i]] <- rows
    }
    out <- do.call(rbind, rows_list)
    out <- out[order(out$query_id, out$rank), , drop = FALSE]
    rownames(out) <- NULL
    return(out)
  }
  
  # For reranking strategies 'emb' and 'mmr' we need saved embeddings
  if (is.null(embeddings_path) || !nzchar(embeddings_path) || !file.exists(embeddings_path)) {
    # try to infer: replace .faiss with _embeddings.rds
    inferred <- sub("\\.faiss$", "_embeddings.rds", index_path)
    if (file.exists(inferred)) {
      embeddings_path <- inferred
    } else {
      stop("embeddings_path is required (or save embeddings with build_from_dir). Could not infer from index_path.")
    }
  }
  emb_mat <- readRDS(embeddings_path)    # expected n x d numeric matrix
  
  # helpers
  .cosine_sim <- function(q, X) {
    qn <- q / sqrt(rowSums(q * q))
    Xn <- X / sqrt(rowSums(X * X))
    as.matrix(qn %*% t(Xn))
  }
  .mmr_select <- function(query_emb, candidate_embs, lambda = 0.5, top_n = 5) {
    n <- nrow(candidate_embs)
    if (top_n <= 0) return(integer(0))
    if (top_n >= n) return(seq_len(n))
    qn <- as.numeric(query_emb / sqrt(sum(query_emb * query_emb)))
    cn <- candidate_embs / sqrt(rowSums(candidate_embs * candidate_embs))
    sim_q_c <- as.numeric(qn %*% t(cn))
    sim_c_c <- cn %*% t(cn)
    selected <- integer(0)
    remaining <- seq_len(n)
    first <- which.max(sim_q_c)
    selected <- c(selected, first)
    remaining <- setdiff(remaining, first)
    while (length(selected) < top_n && length(remaining) > 0) {
      scores <- sapply(remaining, function(j) {
        rel <- sim_q_c[j]
        red <- max(sim_c_c[j, selected])
        lambda * rel - (1 - lambda) * red
      })
      best <- remaining[which.max(scores)]
      selected <- c(selected, best)
      remaining <- setdiff(remaining, best)
    }
    selected
  }
  
  out_rows <- vector("list", n_queries)
  for (i in seq_len(n_queries)) {
    cand_ids <- I[i, ]
    dists <- D[i, ]
    # coerce and sanitize
    cand_ids_int <- suppressWarnings(as.integer(cand_ids))
    cand_ids_int[is.na(cand_ids_int) | cand_ids_int <= 0 | cand_ids_int > nrow(man)] <- NA_integer_
    valid_mask <- !is.na(cand_ids_int)
    if (!any(valid_mask)) {
      # produce empty NA row set
      out_rows[[i]] <- data.frame(
        query_id = integer(0), rank = integer(0), distance = numeric(0),
        file = character(0), chunk_id = integer(0), snippet = character(0),
        stringsAsFactors = FALSE
      )
      next
    }
    cand_ids_valid <- cand_ids_int[valid_mask]
    # Embedding rerank: compute cosine on saved embeddings for candidates
    if (rerank == "emb") {
      qvec <- qemb[i, , drop = FALSE]
      cand_embs <- emb_mat[cand_ids_valid, , drop = FALSE]
      sims <- as.numeric(.cosine_sim(qvec, cand_embs))
      order_idx <- order(sims, decreasing = TRUE)
      new_ids <- cand_ids_valid[order_idx]
      new_scores <- sims[order_idx]
      rows <- data.frame(
        query_id = i,
        rank = seq_along(new_ids),
        distance = new_scores,
        file = man$file[new_ids],
        chunk_id = man$chunk_id[new_ids],
        snippet = man$snippet[new_ids],
        stringsAsFactors = FALSE
      )
      out_rows[[i]] <- rows
    } else if (rerank == "mmr") {
      qvec <- as.numeric(qemb[i, , drop = TRUE])
      cand_embs <- emb_mat[cand_ids_valid, , drop = FALSE]
      pick_idx <- .mmr_select(qvec, cand_embs, lambda = mmr_lambda, top_n = length(cand_ids_valid))
      new_ids <- cand_ids_valid[pick_idx]
      new_scores <- as.numeric(.cosine_sim(matrix(qvec, nrow = 1), emb_mat[new_ids, , drop = FALSE]))
      rows <- data.frame(
        query_id = i,
        rank = seq_along(new_ids),
        distance = new_scores,
        file = man$file[new_ids],
        chunk_id = man$chunk_id[new_ids],
        snippet = man$snippet[new_ids],
        stringsAsFactors = FALSE
      )
      out_rows[[i]] <- rows
    } else if (rerank == "llm") {
      if (is.null(llm_chat)) stop("llm_chat must be provided for rerank = 'llm'")
      snippets <- man$snippet[cand_ids_valid]
      # small scoring prompt (user can change)
      prompt <- paste0(
        "Rate the relevance of each snippet to the question below on a scale 0-1.\n\n",
        "Question: ", queries[i], "\n\n",
        "Snippets:\n",
        paste0(seq_along(snippets), ": ", substr(snippets, 1, 600), collapse = "\n\n"),
        "\n\nRespond with a JSON array of numbers only; e.g. [0.9, 0.2, 0.1]"
      )
      # Use ellmer Chat object to ask; caller is responsible for providing one
      llm_res <- llm_chat$chat(prompt, echo = "none")
      scores <- tryCatch({
        txt <- paste(llm_res, collapse = "\n")
        jsonlite::fromJSON(txt)
      }, error = function(e) {
        rep(NA_real_, length(snippets))
      })
      if (length(scores) != length(snippets)) scores <- rep(NA_real_, length(snippets))
      order_idx <- order(scores, decreasing = TRUE, na.last = TRUE)
      new_ids <- cand_ids_valid[order_idx]
      new_scores <- scores[order_idx]
      rows <- data.frame(
        query_id = i,
        rank = seq_along(new_ids),
        distance = new_scores,
        file = man$file[new_ids],
        chunk_id = man$chunk_id[new_ids],
        snippet = man$snippet[new_ids],
        stringsAsFactors = FALSE
      )
      out_rows[[i]] <- rows
    }
  }
  
  out <- do.call(rbind, out_rows)
  # keep only top-k per query and order
  out <- do.call(rbind, lapply(split(out, out$query_id), function(df_q) df_q[seq_len(min(k, nrow(df_q))), , drop = FALSE]))
  out <- out[order(out$query_id, out$rank), , drop = FALSE]
  rownames(out) <- NULL
  out
}