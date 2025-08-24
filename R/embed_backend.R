# Embedding backend using sentence-transformers via reticulate
#
# Robustly handles the various shapes that python/reticulate may return:
#  - numeric vector (length d) -> 1 x d
#  - matrix d x 1  -> transpose -> 1 x d
#  - matrix d x n  -> transpose -> n x d
#  - matrix n x d  -> keep
# Returns an n x d numeric matrix (rows = texts)
#
# Notes:
# - We set a few environment variables before importing Python to reduce
#   warnings and avoid OpenBLAS / tokenizers parallelism issues.
# - Use `force_reload = TRUE` if you need to recreate the Python model object
#   (e.g., change model name or python binary at runtime).

.emb_init <- local({
  model_state <- list(model = NULL, python = NULL, obj = NULL)
  
  function(model = "sentence-transformers/all-MiniLM-L6-v2",
           python = NULL,
           env_overrides = list(),
           force_reload = FALSE) {
    
    if (!requireNamespace("reticulate", quietly = TRUE))
      stop("The 'reticulate' package is required. Install with install.packages('reticulate')", call. = FALSE)
    
    # ---- Recommended env settings to avoid tokenizers/OpenBLAS warnings ----
    # Set these BEFORE importing Python modules.
    set_if_missing <- function(name, val) {
      cur <- Sys.getenv(name, unset = NA)
      if (is.na(cur) || cur == "") Sys.setenv(name = val)
    }
    set_if_missing("TOKENIZERS_PARALLELISM", "false")
    set_if_missing("OPENBLAS_NUM_THREADS", "1")
    set_if_missing("OMP_NUM_THREADS", "1")
    set_if_missing("MKL_NUM_THREADS", "1")
    
    # allow caller to override/add envs if desired (explicit overrides)
    if (length(env_overrides)) {
      for (nm in names(env_overrides)) Sys.setenv(nm, as.character(env_overrides[[nm]]))
    }
    
    # Optionally pick python BEFORE import (do this before any import)
    if (!is.null(python)) {
      # use_python will error if the binary is not valid
      reticulate::use_python(python, required = TRUE)
    }
    
    # If we already have model_state and it matches, return it (unless forced)
    if (!force_reload &&
        !is.null(model_state$obj) &&
        identical(model_state$model, model) &&
        identical(model_state$python, python)) {
      return(model_state$obj)
    }
    
    # Try to import sentence_transformers with a friendly error
    st <- tryCatch(
      reticulate::import("sentence_transformers", delay_load = TRUE),
      error = function(e) {
        stop(
          "Failed to import Python module 'sentence_transformers'.\n",
          "Make sure the package is installed in the Python used by reticulate.\n",
          "You can install it with `pip install sentence-transformers` in that environment.\n",
          "Original Python error: ", conditionMessage(e),
          call. = FALSE
        )
      }
    )
    
    model_obj <- tryCatch(
      st$SentenceTransformer(model),
      error = function(e) {
        stop(
          "Failed to construct SentenceTransformer('", model, "').\n",
          "Confirm the model id is correct and internet access is available (if model needs to be downloaded).\n",
          "Original Python error: ", conditionMessage(e),
          call. = FALSE
        )
      }
    )
    
    # save state
    model_state$model <- model
    model_state$python <- python
    model_state$obj <- model_obj
    
    model_obj
  }
})

#' Embed texts with sentence-transformers (via reticulate)
#'
#' @param texts character vector of texts to embed.
#' @param model character model id passed to SentenceTransformer (default "sentence-transformers/all-MiniLM-L6-v2").
#' @param python optional path to Python binary to force reticulate to use.
#' @param show_progress logical; show sentence-transformers progress bar.
#' @return numeric matrix n x d (rows = texts).
#' @keywords internal
.embed_texts <- function(texts,
                         model = "sentence-transformers/all-MiniLM-L6-v2",
                         python = NULL,
                         show_progress = FALSE) {
  if (!length(texts)) {
    # return 0 x 0 matrix for empty input
    return(matrix(numeric(0), nrow = 0, ncol = 0))
  }
  
  st <- .emb_init(model = model, python = python)
  
  embs <- tryCatch(
    st$encode(texts, convert_to_numpy = TRUE, show_progress_bar = show_progress),
    error = function(e) {
      stop("sentence-transformers encode() failed: ", conditionMessage(e), call. = FALSE)
    }
  )
  
  em_r <- reticulate::py_to_r(embs)
  
  # If it's a 1-D numeric vector -> make 1 x d
  if (is.null(dim(em_r))) {
    em_r <- matrix(em_r, nrow = 1)
  } else {
    em_r <- as.matrix(em_r)
  }
  
  # Now we expect rows == length(texts). If not, try common transposes:
  if (nrow(em_r) != length(texts)) {
    # case: embeddings came back as columns (d x n) -> transpose
    if (ncol(em_r) == length(texts)) {
      em_r <- t(em_r)
    } else if (length(texts) == 1 && ncol(em_r) == 1 && nrow(em_r) > 1) {
      # case: d x 1 single embedding -> transpose to 1 x d
      em_r <- t(em_r)
    } else {
      stop(sprintf(
        "Unexpected embedding shape: after conversion got %d x %d matrix but expected %d rows (texts).",
        nrow(em_r), ncol(em_r), length(texts)
      ), call. = FALSE)
    }
  }
  
  storage.mode(em_r) <- "double"
  em_r
}
