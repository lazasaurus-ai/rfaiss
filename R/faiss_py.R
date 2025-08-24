# Internal helpers for Python/NumPy/FAISS interop
#
# - robust imports with helpful errors
# - shape helpers to coerce/transpose common numpy -> R shapes
# - to_float32: returns a C-contiguous numpy.float32 array suitable for FAISS
# - faiss_search_result: converts (D, I) from FAISS into R, turning -1 -> NA and 0-based -> 1-based

#' Point rfaiss to a specific Python environment (optional)
#' @export
use_rfaiss_python <- function(env) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Install with install.packages('reticulate').", call. = FALSE)
  }
  if (file.exists(env)) {
    reticulate::use_python(env, required = TRUE)
  } else {
    reticulate::use_condaenv(env, required = TRUE)
  }
  invisible(env)
}

# internal: import module with clearer error
.import_mod <- function(name) {
  tryCatch(
    reticulate::import(name, delay_load = TRUE),
    error = function(e) {
      stop(sprintf("Failed to import Python module '%s'.\nMake sure it is installed in the Python used by reticulate.\nOriginal error: %s",
                   name, conditionMessage(e)),
           call. = FALSE)
    }
  )
}

.faiss_mod <- function() .import_mod("faiss")
.np_mod    <- function() .import_mod("numpy")

# Ensure we have an R matrix (n x d). Accepts:
# - numeric vector (length d) -> 1 x d
# - matrix d x 1 (column) when length(texts)==1 -> transpose -> 1 x d
# - matrix d x n (embeddings as columns) -> transpose -> n x d
# On failure, raises a helpful error.
.ensure_matrix_d <- function(x, d, name = "x") {
  if (is.null(x)) stop(sprintf("%s is NULL", name))
  # coerce to matrix if vector
  if (is.null(dim(x))) {
    x <- matrix(x, nrow = 1)
  } else {
    x <- as.matrix(x)
  }
  
  # If already n x d -> done
  if (ncol(x) == d) {
    return(x)
  }
  
  # If d x n -> transpose
  if (nrow(x) == d && ncol(x) != d) {
    x <- t(x)
    if (ncol(x) == d) return(x)
  }
  
  # If 1d column vector (d x 1) and we expected d columns for a single-row add
  if (ncol(x) == 1 && nrow(x) == d) {
    x <- t(x)
    if (ncol(x) == d) return(x)
  }
  
  stop(sprintf("Dimension mismatch: %s has shape %d x %d but expected %d columns.", name, nrow(x), ncol(x), d))
}

# Convert an R matrix / vector to a NumPy float32 C-contiguous array.
# If `expect_d` provided, will attempt a transpose conversion if needed to match expected d.
.to_float32 <- function(x, expect_d = NULL) {
  np <- .np_mod()
  # coerce to matrix/array
  if (is.null(dim(x))) {
    x <- matrix(x, nrow = 1)
  } else {
    x <- as.matrix(x)
  }
  
  # If expect_d supplied, try to coerce shape (transpose if needed)
  if (!is.null(expect_d)) {
    if (ncol(x) != expect_d) {
      if (nrow(x) == expect_d) {
        x <- t(x)
      } else {
        stop(sprintf("to_float32: unexpected shape %d x %d and expected columns = %d", nrow(x), ncol(x), expect_d))
      }
    }
  }
  
  # ensure double before handing to numpy (safer)
  storage.mode(x) <- "double"
  
  # create numpy array with dtype float32 and C-order
  np$array(x, dtype = "float32", order = "C")
}

# Convert FAISS search tuple (D, I) into R-friendly list:
# - distances: numeric matrix (rows = queries, cols = k)
# - indices: integer matrix 1-based, with NA for missing (-1 in faiss)
.faiss_py_search_result <- function(di) {
  # di is the Python tuple (D, I)
  D_r <- reticulate::py_to_r(di[[1]])
  I_r <- reticulate::py_to_r(di[[2]])
  
  # Normalize shapes: if returned as 1-D (single query) convert to 1 x k
  if (is.null(dim(D_r))) {
    D <- matrix(D_r, nrow = 1)
  } else {
    D <- as.matrix(D_r)
  }
  
  if (is.null(dim(I_r))) {
    I0 <- matrix(I_r, nrow = 1)
  } else {
    I0 <- as.matrix(I_r)
  }
  
  # Sanity: shapes must match (same nrow, ncol)
  if (!all(dim(D) == dim(I0))) {
    stop(sprintf("faiss search returned mismatched shapes: distances %d x %d, indices %d x %d",
                 nrow(D), ncol(D), nrow(I0), ncol(I0)))
  }
  
  # FAISS uses -1 for missing indices; map to NA_integer_
  missing_mask <- (I0 < 0)
  I0[missing_mask] <- NA_integer_
  
  # Convert to 1-based indices for R (NA stays NA)
  I <- I0 + 1L
  
  # Set distances to NA where index is missing
  if (any(missing_mask)) {
    D[missing_mask] <- NA_real_
  }
  
  storage.mode(D) <- "double"
  storage.mode(I) <- "integer"
  
  list(distances = D, indices = I)
}
