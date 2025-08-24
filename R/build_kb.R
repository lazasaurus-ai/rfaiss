#' Build a KB (FAISS index + manifest) from a directory (convenience wrapper)
#'
#' A convenience wrapper around \code{build_flatl2_from_dir()} that chooses a
#' sensible default \code{out_prefix} (by default next to the source files),
#' writes a small README describing the KB, and returns the result invisibly.
#'
#' @param dir Directory containing source files to index.
#' @param out_prefix Optional path prefix for the generated index and manifest.
#'   If NULL (default) the wrapper will create a prefix next to \code{dir}:
#'   \code{file.path(dir, paste0("kb_", timestamp))}. If \code{use_tmp = TRUE},
#'   a temporary prefix (via \code{tempfile()}) will be used.
#' @param use_tmp Logical; if TRUE forces a tempfile() prefix (useful for quick tests).
#' @param overwrite Logical; if FALSE (default) and an index/manifest already exist at
#'   the computed prefix, this function will error. Set TRUE to replace.
#' @param readme Logical; write a small README markdown file beside the index/manifest.
#' @param ... Additional arguments are forwarded to \code{build_flatl2_from_dir()}:
#'   e.g. \code{pattern}, \code{chunk_chars}, \code{overlap}, \code{model}, \code{python}.
#' @return Invisibly returns the result produced by \code{build_flatl2_from_dir()},
#'   with an added element \code$readme_path (if \code{readme = TRUE}).
#' @export
#' @examples
#' \dontrun{
#' # build KB next to ./docs
#' build_kb("docs")
#'
#' # build KB into /tmp for quick test
#' build_kb("docs", use_tmp = TRUE)
#' }
build_kb <- function(dir,
                     out_prefix = NULL,
                     use_tmp = FALSE,
                     overwrite = FALSE,
                     readme = TRUE,
                     ...) {
  stopifnot(dir.exists(dir))
  # normalize dir
  dir <- normalizePath(dir)
  
  # choose out_prefix
  if (is.null(out_prefix)) {
    if (isTRUE(use_tmp)) {
      out_prefix <- tempfile("kb_")
    } else {
      ts <- gsub("[: -]", "_", format(Sys.time(), "%Y-%m-%d_%H-%M-%S"))
      # try to place next to source dir (dir/kb_TIMESTAMP)
      base <- paste0("kb_", ts)
      out_prefix <- file.path(dir, base)
    }
  } else {
    out_prefix <- normalizePath(out_prefix, mustWork = FALSE)
  }
  
  idx_path <- paste0(out_prefix, ".faiss")
  man_path <- paste0(out_prefix, "_manifest.rds")
  readme_path <- paste0(out_prefix, "_README.md")
  
  # guard against accidental overwrite
  if (!overwrite) {
    if (file.exists(idx_path) || file.exists(man_path) || file.exists(readme_path)) {
      stop("Output files already exist for prefix '", out_prefix, "'. Set overwrite = TRUE to replace.")
    }
  }
  
  # Ensure parent dir exists
  out_dir <- dirname(out_prefix)
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  # call underlying builder (propagate errors)
  res <- build_flatl2_from_dir(dir = dir, out_prefix = out_prefix, ...)
  
  # add README if requested
  if (isTRUE(readme)) {
    n_chunks <- if (!is.null(res$n_chunks)) res$n_chunks else NA_integer_
    d <- if (!is.null(res$dim)) res$dim else NA_integer_
    created <- format(Sys.time(), tz = "UTC")
    files_indexed <- unique(res$manifest$file)
    files_count <- length(files_indexed)
    
    readme_lines <- c(
      "# KB index (auto-generated)",
      "",
      paste0("- **source_dir:** ", dir),
      paste0("- **created_at (UTC):** ", created),
      paste0("- **index_path:** ", normalizePath(res$index_path, mustWork = FALSE)),
      paste0("- **manifest_path:** ", normalizePath(res$manifest_path, mustWork = FALSE)),
      paste0("- **embedding_dim:** ", d),
      paste0("- **n_chunks:** ", n_chunks),
      paste0("- **files_indexed (count):** ", files_count),
      "",
      "## Files indexed (top 50)",
      "",
      paste0(head(files_indexed, 50), collapse = "\n"),
      "",
      "## Notes",
      "",
      "- This KB was produced by `build_kb()` which calls `build_flatl2_from_dir()`.",
      "- You can search using `search_dir_index(index_path, manifest_path, queries, k)`.",
      "- To use this KB for RAG with Bedrock / ellmer, pass the paths into `rag_query_aws()`."
    )
    
    # attempt to write README; if it fails, warn but do not fail the whole build
    tryCatch({
      writeLines(readme_lines, con = readme_path, useBytes = TRUE)
    }, error = function(e) {
      warning("Failed to write README at '", readme_path, "': ", conditionMessage(e))
    })
    
    res$readme_path <- readme_path
  }
  
  invisible(res)
}
