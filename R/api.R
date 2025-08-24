# -------- Core Python-backed FAISS API --------

#' Create a FlatL2 index
#' @param d Integer, vector dimension.
#' @return An rfaiss index handle (python-backed).
#' @export
create_flat_l2 <- function(d) {
  faiss <- .faiss_mod()
  idx <- faiss$IndexFlatL2(as.integer(d))
  structure(list(.x = idx, d = as.integer(d), engine = "python"),
            class = "rfaiss_py_index")
}

# helper: ensure object is an n x d numeric matrix; allow a length-d vector -> 1 x d
.ensure_matrix_d <- function(x, d, name = "x") {
  if (is.null(dim(x))) {
    # vector
    if (length(x) == 0) {
      return(matrix(numeric(0), nrow = 0, ncol = d))
    }
    if (length(x) == d) {
      x <- matrix(x, nrow = 1)
    } else {
      stop(sprintf("Dimension mismatch: length(%s) = %d but index$d = %d", name, length(x), d))
    }
  } else {
    x <- as.matrix(x)
    if (ncol(x) != d) {
      stop(sprintf("Dimension mismatch: ncol(%s) = %d but index$d = %d", name, ncol(x), d))
    }
  }
  storage.mode(x) <- "double"
  x
}

#' Add vectors to an index (L2)
#'
#' @param handle Object returned by [create_flat_l2()].
#' @param x Numeric matrix (rows = vectors, ncol must equal index dimension).
#' @export
index_add <- function(handle, x) {
  stopifnot(inherits(handle, "rfaiss_py_index"))
  x <- .ensure_matrix_d(x, handle$d, "x")
  # convert to float32 / numpy (helper in faiss_py.R)
  x32 <- .to_float32(x)
  handle$.x$add(x32)
  invisible(NULL)
}

#' Search k nearest neighbors (L2)
#'
#' @param handle rfaiss index handle.
#' @param q Query matrix (rows = queries, ncol must match index dimension).
#' @param k Number of neighbors to return.
#' @return A list with `distances` and `indices` matrices.
#' @export
index_search <- function(handle, q, k = 5L) {
  stopifnot(inherits(handle, "rfaiss_py_index"))
  # coerce and ensure shape n x d
  q <- .ensure_matrix_d(q, handle$d, "q")
  q32 <- .to_float32(q, expect_d = handle$d)
  di <- handle$.x$search(q32, as.integer(k))
  .faiss_py_search_result(di)
}


# -------- Cosine / Inner-Product helpers (normalize + IndexFlatIP) --------

#' Create an IndexFlatIP (inner-product) index for cosine similarity (use with normalized vectors)
#' @param d Integer, vector dimension.
#' @return An rfaiss index handle (python-backed).
#' @export
create_flat_cosine <- function(d) {
  faiss <- .faiss_mod()
  idx <- faiss$IndexFlatIP(as.integer(d))
  structure(list(.x = idx, d = as.integer(d), engine = "python"),
            class = "rfaiss_py_index")
}

#' Add vectors to an IndexFlatIP after normalizing rows (so inner product == cosine)
#'
#' @param handle rfaiss index handle created by [create_flat_cosine()].
#' @param x Numeric matrix (rows = vectors).
#' @export
index_add_cosine <- function(handle, x) {
  stopifnot(inherits(handle, "rfaiss_py_index"))
  x <- .ensure_matrix_d(x, handle$d, "x")
  # normalize rows (safe)
  norms <- sqrt(rowSums(x * x))
  norms[norms == 0] <- 1
  x_norm <- x / norms
  x32 <- .to_float32(x_norm)
  handle$.x$add(x32)
  invisible(NULL)
}

#' Search using IndexFlatIP (for cosine use; returned "distances" are inner-products / cosine scores)
#'
#' NOTE: IndexFlatIP returns inner-products (higher = better). To maintain compatibility with the
#' L2 API we still return a `distances` matrix — in the cosine case these are similarity scores.
#'
#' @param handle rfaiss index handle (created by [create_flat_cosine()]).
#' @param q Query matrix (rows = queries). Should be normalized the same way you normalized index vectors.
#' @param k Number of neighbors to return.
#' @return A list with `distances` (similarities) and `indices`.
#' @export
index_search_cosine <- function(handle, q, k = 5L) {
  stopifnot(inherits(handle, "rfaiss_py_index"))
  q <- .ensure_matrix_d(q, handle$d, "q")
  # normalize queries
  norms <- sqrt(rowSums(q * q))
  norms[norms == 0] <- 1
  q_norm <- q / norms
  q32 <- .to_float32(q_norm, expect_d = handle$d)
  di <- handle$.x$search(q32, as.integer(k))
  .faiss_py_search_result(di)
}


# -------- Optional: save / load --------

#' Save a FAISS index to disk
#' @param handle rfaiss index handle.
#' @param path File path (e.g., "myindex.faiss").
#' @export
save_index <- function(handle, path) {
  stopifnot(inherits(handle, "rfaiss_py_index"))
  faiss <- .faiss_mod()
  faiss$write_index(handle$.x, normalizePath(path, mustWork = FALSE))
  invisible(path)
}

#' Load a FAISS index from disk
#' @param path Path created by [save_index()].
#' @return rfaiss index handle.
#' @export
load_index <- function(path) {
  faiss <- .faiss_mod()
  idx <- faiss$read_index(normalizePath(path, mustWork = TRUE))
  # discover d from the index (swig object exposes $d)
  d <- as.integer(idx$d)
  structure(list(.x = idx, d = d, engine = "python"),
            class = "rfaiss_py_index")
}
