#' Cosine similarity (row-wise); accepts matrices q (m x d) and X (n x d)
cosine_sim <- function(q, X) {
  # normalize
  qn <- q / sqrt(rowSums(q * q))
  Xn <- X / sqrt(rowSums(X * X))
  # result: m x n
  as.matrix(qn %*% t(Xn))
}

#' MMR reranking
#' @param query_emb numeric vector (d)
#' @param candidate_embs matrix (n x d)
#' @param lambda tradeoff 0..1
#' @param top_n number of output indices to pick (<= n)
#' @return integer vector of selected indices (1-based, into candidate_embs)
mmr_select <- function(query_emb, candidate_embs, lambda = 0.5, top_n = 5) {
  n <- nrow(candidate_embs)
  if (top_n <= 0) return(integer(0))
  if (top_n >= n) return(seq_len(n))
  # cosine sims
  qn <- as.numeric(query_emb / sqrt(sum(query_emb * query_emb)))
  cn <- candidate_embs / sqrt(rowSums(candidate_embs * candidate_embs))
  sim_q_c <- as.numeric(qn %*% t(cn))   # length n
  sim_c_c <- cn %*% t(cn)              # n x n
  
  selected <- integer(0)
  remaining <- seq_len(n)
  
  # pick highest sim to query first
  first <- which.max(sim_q_c)
  selected <- c(selected, first)
  remaining <- setdiff(remaining, first)
  
  while (length(selected) < top_n && length(remaining) > 0) {
    scores <- sapply(remaining, function(j) {
      rel <- sim_q_c[j]
      # redundancy: max similarity to already selected
      red <- max(sim_c_c[j, selected])
      lambda * rel - (1 - lambda) * red
    })
    best <- remaining[which.max(scores)]
    selected <- c(selected, best)
    remaining <- setdiff(remaining, best)
  }
  selected
}

#' Rerank FAISS top-k using saved embeddings (exact similarity)
#' @param qemb matrix 1 x d or m x d
#' @param emb_mat full embeddings matrix n x d
#' @param candidate_ids integer vector (1-based indices into emb_mat) returned by FAISS
#' @return order indices of candidate_ids by exact similarity (descending)
rerank_by_embeddings <- function(qemb, emb_mat, candidate_ids) {
  # qemb maybe m x d, but for a single query we expect 1 x d
  if (is.null(dim(qemb))) qemb <- matrix(qemb, nrow = 1)
  # take candidate embeddings
  cand <- emb_mat[candidate_ids, , drop = FALSE]
  sims <- cosine_sim(qemb, cand)   # m x k
  # return order for each query (as list if multiple queries)
  apply(sims, 1, function(row) order(row, decreasing = TRUE))
}
