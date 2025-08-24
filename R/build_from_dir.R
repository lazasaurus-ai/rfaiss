#' Build a FAISS index from a directory and save embeddings/manifest
#'
#' @param dir directory to scan
#' @param out_prefix path prefix to save index and manifest (no extension)
#' @param pattern regex file filter
#' @param chunk_chars chunk char length
#' @param overlap chunk overlap in chars
#' @param model embedding model
#' @param python optional python binary path
#' @param metric "l2" or "cosine" (cosine uses IndexFlatIP with row-normalized vectors)
#' @param save_embeddings logical, whether to save embeddings to OUT_PREFIX_embeddings.rds
#' @export
build_flatl2_from_dir <- function(
    dir,
    out_prefix,
    pattern = "\\.(txt|md|r|py|qmd|rmd|pdf|csv|tsv|json|ya?ml)$",
    chunk_chars = 1200L,
    overlap = 200L,
    model = "sentence-transformers/all-MiniLM-L6-v2",
    python = Sys.getenv("RETICULATE_PYTHON", unset = NA_character_),
    metric = c("l2", "cosine"),
    save_embeddings = TRUE
) {
  metric <- match.arg(metric)
  stopifnot(dir.exists(dir))
  files <- list_files(dir, pattern = pattern, recursive = TRUE)
  if (length(files) == 0) stop("No files matched in: ", dir)
  
  rows <- list()
  for (f in files) {
    txt <- tryCatch(read_text_safe(f), error = function(e) "")
    ch <- chunk_text(txt, chunk_chars = chunk_chars, overlap = overlap)
    if (length(ch) == 0) next
    rows[[f]] <- data.frame(file = f, chunk_id = seq_along(ch), text = ch,
                            stringsAsFactors = FALSE)
  }
  df <- do.call(rbind, rows)
  if (is.null(df) || nrow(df) == 0) stop("No text chunks produced.")
  
  message("Embedding ", nrow(df), " chunks with model ", model, " ...")
  em <- .embed_texts(df$text, model = model, python = python)
  
  if (nrow(em) != nrow(df)) {
    stop("Embedding shape mismatch: got ", nrow(em), " embeddings but ", nrow(df), " chunks.")
  }
  
  d <- ncol(em)
  message("Embedding dim: ", d)
  
  # choose index type
  if (metric == "cosine") {
    h <- create_flat_cosine(d)
    # index_add_cosine will normalize rows
    index_add_cosine(h, em)
  } else {
    h <- create_flat_l2(d)
    index_add(h, em)
  }
  
  # persist index + manifest + embeddings
  idx_path <- paste0(out_prefix, ".faiss")
  man_path <- paste0(out_prefix, "_manifest.rds")
  emb_path <- paste0(out_prefix, "_embeddings.rds")
  
  save_index(h, idx_path)
  
  # Attach snippets for convenience (first 200 chars)
  df$snippet <- substr(df$text, 1, 200)
  manifest <- df[, c("file", "chunk_id", "snippet")]
  saveRDS(manifest, man_path)
  
  if (isTRUE(save_embeddings)) {
    # save numeric matrix n x d in RDS so we can rerank later
    saveRDS(em, emb_path)
  }
  
  invisible(list(handle = h, index_path = idx_path, manifest_path = man_path,
                 embeddings_path = if (isTRUE(save_embeddings)) emb_path else NULL,
                 manifest = manifest, dim = d, n_chunks = nrow(df)))
}
