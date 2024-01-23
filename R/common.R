nderiv <- function(y, x = NULL, dx = NULL,
                   end_method = "NA")
{
  assertthat::assert_that(!is.null(x) | !is.null(dt),
                          msg = "Either x or dt need to be non-NULL")
  assertthat::assert_that(end_method %in% c("NA", "forward_backward"))
  
  if (!is.null(x) && (length(x) == 1))
    dx <- x
  else if (!is.null(x) && (length(x) > 1))
    assertthat::are_equal(length(x), length(y))
  
  n <- length(y)
  
  dy <- (dplyr::lead(y) - dplyr::lag(y)) / 2
  if (end_method == "forward_backward") {
    dy[1] <- y[2] - y[1]
    dy[n] <- y[n] - y[n-1]
  }
  
  if (length(x) > 1) {
    dx <- (dplyr::lead(x) - dplyr::lag(x)) / 2
    
    if (end_method == "forward_backward") {
      dx[1] <- x[2] - x[1]
      dx[n] <- x[n] - x[n-1]
    }
  }
  
  return(dy / dx)
}