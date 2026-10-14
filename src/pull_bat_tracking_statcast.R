## Pull Statcast data via baseballr::statcast_search(), keep only rows that
## have bat tracking data (bat_speed and/or swing_length populated), covering
## the 2023 season (when MLB's Hawk-Eye bat tracking rolled out mid-season)
## through the most recent available day.
##
## IMPORTANT: Baseball Savant's CSV search endpoint silently truncates results
## at 25,000 rows per query. A single day's full slate tops out around ~5,000
## pitches league-wide, so we chunk DAY BY DAY (not week-by-week) to guarantee
## nothing gets cut off. We still log a loud warning if any single day ever
## hits the cap, as a sanity check.

suppressPackageStartupMessages({
  library(baseballr)
  library(dplyr)
  library(lubridate)
})

out_file <- "data/statcast_bat_tracking_data.csv"
log_file <- "results/logs/pull_log.txt"
ROW_CAP <- 25000

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = log_file, append = TRUE)
}

today <- Sys.Date()
last_available <- today - 1  # Statcast data is compiled overnight

# In-season date ranges only (skip off-season months to save requests).
# Bat tracking itself won't exist before mid-2023 anyway; starting the first
# window a bit before the confirmed mid-July-2023 boundary and filtering on
# non-NA bat_speed/swing_length lets the data itself show exactly when it
# starts, rather than relying on a guessed cutoff.
season_windows <- list(
  c("2023-06-15", "2023-11-05"),
  c("2024-03-01", "2024-11-05"),
  c("2025-03-01", "2025-11-05"),
  c("2026-03-01", as.character(last_available))
)

build_days <- function(start_date, end_date) {
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)
  if (start_date > end_date) return(NULL)
  seq(start_date, end_date, by = "1 day")
}

all_days <- unlist(lapply(season_windows, function(w) build_days(w[1], w[2])))
all_days <- as.Date(all_days, origin = "1970-01-01")
all_days <- sort(unique(all_days[all_days <= last_available]))

log_msg("Total days to pull: ", length(all_days))

first_write <- !file.exists(out_file)
total_bt_rows <- 0
total_rows_seen <- 0
capped_days <- character(0)

for (i in seq_along(all_days)) {
  d <- as.character(all_days[i])

  attempt <- 1
  max_attempts <- 4
  success <- FALSE

  while (!success && attempt <= max_attempts) {
    result <- tryCatch({
      statcast_search(start_date = d, end_date = d, player_type = "batter")
    }, error = function(e) {
      log_msg("  ERROR on ", d, " (attempt ", attempt, "): ", conditionMessage(e))
      NULL
    })

    if (!is.null(result) && is.data.frame(result) && nrow(result) > 0) {
      total_rows_seen <- total_rows_seen + nrow(result)

      if (nrow(result) >= ROW_CAP) {
        log_msg("  *** WARNING: ", d, " returned ", nrow(result), " rows -- possible truncation at cap! ***")
        capped_days <- c(capped_days, d)
      }

      has_bat_speed <- "bat_speed" %in% names(result)
      has_swing_len <- "swing_length" %in% names(result)

      if (has_bat_speed || has_swing_len) {
        bt_rows <- result %>%
          filter(
            (if (has_bat_speed) !is.na(bat_speed) else FALSE) |
            (if (has_swing_len) !is.na(swing_length) else FALSE)
          )

        if (nrow(bt_rows) > 0) {
          readr::write_csv(bt_rows, out_file, append = !first_write, col_names = first_write)
          first_write <- FALSE
          total_bt_rows <- total_bt_rows + nrow(bt_rows)
        }

        if (i %% 10 == 0 || nrow(bt_rows) > 0 && i <= 5) {
          log_msg(sprintf("  [%d/%d] %s: %d pitches, %d with bat tracking (running total: %d)",
                           i, length(all_days), d, nrow(result), nrow(bt_rows), total_bt_rows))
        }
      }
      success <- TRUE
    } else if (!is.null(result) && is.data.frame(result) && nrow(result) == 0) {
      success <- TRUE
    } else {
      attempt <- attempt + 1
      if (attempt <= max_attempts) Sys.sleep(3 * attempt)
    }
  }

  if (!success) {
    log_msg("  GAVE UP on ", d, " after ", max_attempts, " attempts")
  }

  Sys.sleep(0.6)  # be polite to Baseball Savant between requests

  if (i %% 50 == 0) {
    log_msg(sprintf("Progress: %d/%d days done, %d bat-tracking rows so far", i, length(all_days), total_bt_rows))
  }
}

log_msg("DONE. Total pitches scanned: ", total_rows_seen, " | Total rows with bat tracking data: ", total_bt_rows)
if (length(capped_days) > 0) {
  log_msg("*** DAYS THAT HIT THE ROW CAP (data may still be incomplete for these): ", paste(capped_days, collapse = ", "))
} else {
  log_msg("No single day hit the row cap -- data is complete.")
}
log_msg("Output written to: ", out_file)
