required_packages <- c("readr", "dplyr", "lavaan")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install required packages first: install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lavaan)
})

configure_lavaan_ncpus <- function() {
  cores <- suppressWarnings(parallel::detectCores())
  ncpus <- if (is.na(cores) || cores < 2L) 1L else max(1L, cores - 1L)

  cache_env <- get("lavaan_cache_env", envir = asNamespace("lavaan"))
  if (!exists("opt.default", envir = cache_env, inherits = FALSE)) {
    suppressWarnings(lavaan::lavOptions())
  }
  opt_default <- get("opt.default", envir = cache_env)
  opt_check <- get("opt.check", envir = cache_env)
  opt_default$ncpus <- ncpus
  opt_check$ncpus$nm$bounds <- c(1, ncpus)
  assign("opt.default", opt_default, envir = cache_env)
  assign("opt.check", opt_check, envir = cache_env)
}

configure_lavaan_ncpus()

parse_int_env <- function(name, default) {
  value <- Sys.getenv(name, unset = as.character(default))
  parsed <- suppressWarnings(as.integer(value))
  if (is.na(parsed) || parsed < 0) default else parsed
}

input_file <- Sys.getenv(
  "SEM_BINARY_INPUT_FILE",
  unset = Sys.getenv("SEM_INPUT_FILE", unset = "Results.csv")
)
seed <- parse_int_env("SEM_BINARY_SEED", parse_int_env("SEM_SEED", 42))
n_boot <- parse_int_env("SEM_BINARY_BOOTSTRAPS", 0)
write_outputs <- tolower(Sys.getenv("SEM_BINARY_WRITE_OUTPUTS", unset = "1")) %in%
  c("1", "true", "yes", "y")
output_dir <- Sys.getenv("SEM_BINARY_OUTPUT_DIR", unset = "sem_binary_outputs")

latent_indicator_map <- list(
  Emotional_Trust = c("ET_1", "ET_2", "ET_3"),
  Integrity_Trust = c("CTI_1", "CTI_2", "CTI_3")
)
trust_item_vars <- unlist(latent_indicator_map, use.names = FALSE)
structural_path_labels <- c(
  "a1_w", "a1_p", "a1_pre", "a2_w", "a2_p", "a2_pre",
  "c_w", "c_p", "c_wp", "c_pre", "b1", "b2"
)
structural_path_labels_without_pre <- setdiff(
  structural_path_labels,
  c("a1_pre", "a2_pre", "c_pre")
)

to_numeric <- function(x) suppressWarnings(as.numeric(x))

z_score <- function(x) {
  x <- as.numeric(x)
  x_sd <- sd(x, na.rm = TRUE)
  if (is.na(x_sd) || x_sd == 0) {
    return(rep(NA_real_, length(x)))
  }
  (x - mean(x, na.rm = TRUE)) / x_sd
}

select_existing <- function(data, cols) {
  data[, intersect(cols, names(data)), drop = FALSE]
}

set_value_where <- function(x, condition, value) {
  x[which(condition %in% TRUE)] <- value
  x
}

clean_survey_data <- function(path) {
  df <- readr::read_csv(
    path,
    col_types = readr::cols(.default = readr::col_character()),
    show_col_types = FALSE,
    progress = FALSE
  )

  # Python notebook: df.drop([0, 1]).reset_index(drop=True)
  if (nrow(df) >= 2) {
    df <- df[-c(1, 2), , drop = FALSE]
  }
  df$.python_index <- seq_len(nrow(df)) - 1L

  discussion_filter <- (df$DiscNoPers == "1") | (df$DiscPers == "1")
  attention_filter <- df$AC3 == "1"
  df <- df[which((discussion_filter %in% TRUE) & (attention_filter %in% TRUE)), , drop = FALSE]

  drop_cols <- c(
    "StartDate", "EndDate", "Status", "IPAddress", "Progress",
    "Finished", "RecordedDate", "DiscNoPers", "DiscPers",
    "RecipientLastName", "RecipientFirstName", "RecipientEmail",
    "ExternalReference", "LocationLatitude", "LocationLongitude",
    "DistributionChannel", "UserLanguage", "Q_RecaptchaScore",
    "__js_NextPage_MS", "ResponseId",
    "__js_Prompt1_Clicked", "__js_Prompt2_Clicked", "__js_Prompt3_Clicked",
    "__js_Prompt4_Clicked", "__js_AllPromptsClicked", "Gradio",
    "PROLIFIC_PID", "Q_RecaptchaStatus", "Q_RecaptchaError"
  )
  df <- df[, setdiff(names(df), drop_cols), drop = FALSE]

  warmth_columns <- c("W_Q1_1", "W_Q1_2", "W_Q1_3", "W_Q1_4")
  personalization_columns <- c("MC_Personal_1", "MC_Personal_2")
  emotional_trust_columns <- c("ET_1", "ET_2", "ET_3")
  integrity_trust_columns <- c("CTI_1", "CTI_2", "CTI_3")
  digital_literacy_columns <- c("DL_1", "DL_2", "DL_3", "DL_4", "DL_5", "DL_6", "DL_7")
  personality_columns <- paste0("Personality_", 1:10)

  numeric_cols <- c(
    warmth_columns,
    personalization_columns,
    emotional_trust_columns,
    integrity_trust_columns,
    digital_literacy_columns,
    personality_columns,
    "PostDV_Binary", "PreDV_Binary",
    "PreDV_Slider_1", "PostDV_Slider_1",
    "UF_Q1", "Personalization", "Warmth",
    "Duration (in seconds)"
  )
  numeric_cols <- intersect(numeric_cols, names(df))
  df[numeric_cols] <- lapply(df[numeric_cols], to_numeric)

  personality_columns <- intersect(personality_columns, names(df))
  df[personality_columns] <- lapply(df[personality_columns], function(x) {
    x[is.na(x)] <- 0
    x
  })

  df$Slider_Difference <- df$PostDV_Slider_1 - df$PreDV_Slider_1

  df$SD_ChangeDirection <- "No Change"
  df$SD_ChangeDirection <- set_value_where(
    df$SD_ChangeDirection,
    df$Slider_Difference > 0,
    "Increase"
  )
  df$SD_ChangeDirection <- set_value_where(
    df$SD_ChangeDirection,
    df$Slider_Difference < 0,
    "Decrease"
  )

  df$DV_Binary_Diff <- df$PostDV_Binary - df$PreDV_Binary
  df$DV_Change_Diff <- "ChangeTrust"
  df$DV_Change_Diff <- set_value_where(
    df$DV_Change_Diff,
    (df$PostDV_Binary == 2) & (df$PreDV_Binary == 2),
    "Distrust"
  )
  df$DV_Change_Diff <- set_value_where(
    df$DV_Change_Diff,
    (df$PostDV_Binary == 1) & (df$PreDV_Binary == 1),
    "Trust"
  )
  df$DV_Change_Diff <- set_value_where(
    df$DV_Change_Diff,
    (df$PostDV_Binary == 2) & (df$PreDV_Binary == 1),
    "ChangeDistrust"
  )

  df$Warmth_Check <- rowMeans(select_existing(df, warmth_columns), na.rm = TRUE)
  df$Personalization_Check <- rowMeans(select_existing(df, personalization_columns), na.rm = TRUE)
  df$Emotional_Trust <- rowMeans(select_existing(df, emotional_trust_columns), na.rm = TRUE)
  df$Integrity_Trust <- rowMeans(select_existing(df, integrity_trust_columns), na.rm = TRUE)
  df$Digital_Literacy <- rowMeans(select_existing(df, digital_literacy_columns), na.rm = TRUE)
  df$UF_Q1 <- as.integer(df$UF_Q1)

  df$IV_Congruence <- 1L
  df$IV_Congruence <- set_value_where(
    df$IV_Congruence,
    (df$Warmth + df$Personalization) == 1,
    0L
  )

  # TIPI reverse coding matches the Python notebook.
  df$P_Extraversion <- (6 - df$Personality_1) + df$Personality_6
  df$P_Agreeableness <- df$Personality_2 + (6 - df$Personality_7)
  df$P_Conscientiousness <- (6 - df$Personality_3) + df$Personality_8
  df$P_Neuroticism <- (6 - df$Personality_4) + df$Personality_9
  df$P_Openness <- (6 - df$Personality_5) + df$Personality_10

  duration <- df$`Duration (in seconds)`
  df <- df[which(!is.na(duration) & duration > 300 & duration < 2500), , drop = FALSE]

  df
}

remove_iqr_outliers_by_cell <- function(data,
                                        outcome,
                                        group_vars,
                                        cells,
                                        iqr_mult = 1.5,
                                        verbose = TRUE) {
  flagged <- rep(FALSE, nrow(data))
  summaries <- list()

  for (i in seq_len(nrow(cells))) {
    cell_mask <- rep(TRUE, nrow(data))
    cell_label_parts <- character(length(group_vars))

    for (j in seq_along(group_vars)) {
      group_var <- group_vars[j]
      group_value <- cells[[group_var]][i]
      cell_mask <- cell_mask & !is.na(data[[group_var]]) & data[[group_var]] == group_value
      cell_label_parts[j] <- paste0(group_var, "=", group_value)
    }

    idx <- which(cell_mask & !is.na(data[[outcome]]))
    if (length(idx) == 0) next

    x <- data[[outcome]][idx]
    q <- quantile(x, c(0.25, 0.75), na.rm = TRUE, type = 7)
    iqr <- q[[2]] - q[[1]]
    lower <- q[[1]] - iqr_mult * iqr
    upper <- q[[2]] + iqr_mult * iqr
    cell_flag <- x < lower | x > upper
    flagged[idx[cell_flag]] <- TRUE

    summaries[[length(summaries) + 1L]] <- data.frame(
      cell = paste(cell_label_parts, collapse = ", "),
      n = length(x),
      q1 = unname(q[[1]]),
      q3 = unname(q[[2]]),
      iqr = unname(iqr),
      lower = unname(lower),
      upper = unname(upper),
      flagged = sum(cell_flag),
      stringsAsFactors = FALSE
    )
  }

  summary_table <- dplyr::bind_rows(summaries)

  if (verbose) {
    cat("\n=== IQR outlier removal by Warmth x Personalization ===\n")
    print(summary_table, row.names = FALSE)
    cat(sprintf(
      "\nIQR step: flagged %d; N: %d -> %d\n",
      sum(flagged), nrow(data), nrow(data) - sum(flagged)
    ))
  }

  result <- data[!flagged, , drop = FALSE]
  attr(result, "iqr_summary") <- summary_table
  attr(result, "iqr_outliers") <- data[flagged, , drop = FALSE]
  result
}

remove_outliers_by_cooks_d <- function(data,
                                       formula,
                                       threshold_mult = 4,
                                       verbose = TRUE) {
  required <- all.vars(formula)
  d <- data[complete.cases(data[, required, drop = FALSE]), , drop = FALSE]
  fit <- lm(formula, data = d)
  cooks <- cooks.distance(fit)
  threshold <- threshold_mult / nrow(d)
  flagged <- cooks > threshold

  diagnostics <- data.frame(
    row_in_model = seq_len(nrow(d)),
    python_index = if (".python_index" %in% names(d)) d$.python_index else NA_integer_,
    cooks_d = unname(cooks),
    flagged = unname(flagged),
    stringsAsFactors = FALSE
  )

  if (verbose) {
    cat("\n=== Cook's D outlier removal from OLS influence ===\n")
    cat(sprintf(
      "Cook's D step: threshold = %d/n = %.4f; flagged %d; N: %d -> %d\n",
      threshold_mult,
      threshold,
      sum(flagged, na.rm = TRUE),
      nrow(d),
      sum(!flagged, na.rm = TRUE)
    ))
    if (any(flagged, na.rm = TRUE)) {
      print(diagnostics[flagged, , drop = FALSE], row.names = FALSE)
    }
  }

  result <- d[!flagged, , drop = FALSE]
  attr(result, "cooks_diagnostics") <- diagnostics
  attr(result, "initial_lm") <- fit
  result
}

prepare_binary_sem_data <- function(ancova_df) {
  df_sem <- ancova_df
  df_sem$Warmth_x_Personalization <- df_sem$Warmth * df_sem$Personalization
  df_sem$IV_Congruence <- as.integer(df_sem$Warmth == df_sem$Personalization)

  # Notebook logistic model recodes the Qualtrics 1/2 binary answers to 0/1.
  df_sem$PreDV_Binary <- df_sem$PreDV_Binary - 1
  df_sem$PostDV_Binary <- df_sem$PostDV_Binary - 1

  control_vars <- c(
    "Digital_Literacy",
    "P_Extraversion",
    "P_Agreeableness",
    "P_Openness",
    "P_Conscientiousness",
    "P_Neuroticism",
    "UF_Q1"
  )

  df_model <- df_sem
  df_model[control_vars] <- lapply(df_model[control_vars], z_score)

  model_vars <- c(
    "PostDV_Binary",
    trust_item_vars,
    "Warmth",
    "Personalization",
    "Warmth_x_Personalization",
    "PreDV_Binary",
    control_vars
  )

  df_model <- df_model[complete.cases(df_model[, model_vars, drop = FALSE]), model_vars, drop = FALSE]
  df_model[] <- lapply(df_model, as.numeric)

  invalid_post_values <- setdiff(stats::na.omit(unique(df_model$PostDV_Binary)), c(0, 1))
  if (length(invalid_post_values) > 0) {
    stop(
      "PostDV_Binary must be recoded to 0/1 for binary SEM; found: ",
      paste(invalid_post_values, collapse = ", "),
      call. = FALSE
    )
  }

  if (length(unique(df_model$PostDV_Binary)) < 2) {
    stop("PostDV_Binary has fewer than two observed categories after filtering.", call. = FALSE)
  }

  invalid_pre_values <- setdiff(stats::na.omit(unique(df_model$PreDV_Binary)), c(0, 1))
  if (length(invalid_pre_values) > 0) {
    stop(
      "PreDV_Binary must be recoded to 0/1 for binary SEM; found: ",
      paste(invalid_pre_values, collapse = ", "),
      call. = FALSE
    )
  }

  df_model
}

binary_sem_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3

  Emotional_Trust ~ a1_w*Warmth + a1_p*Personalization + a1_pre*PreDV_Binary +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Integrity_Trust ~ a2_w*Warmth + a2_p*Personalization + a2_pre*PreDV_Binary +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  PostDV_Binary ~ c_w*Warmth + c_p*Personalization + c_wp*Warmth_x_Personalization + c_pre*PreDV_Binary +
    b1*Emotional_Trust + b2*Integrity_Trust +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Emotional_Trust ~~ Integrity_Trust

  ind_w_emotional := a1_w*b1
  ind_w_integrity := a2_w*b2
  ind_w_total := a1_w*b1 + a2_w*b2
  direct_w := c_w
  total_w := c_w + ind_w_total

  ind_p_emotional := a1_p*b1
  ind_p_integrity := a2_p*b2
  ind_p_total := a1_p*b1 + a2_p*b2
  direct_p := c_p
  total_p := c_p + ind_p_total

  direct_wp := c_wp

'

binary_sem_without_pre_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3

  Emotional_Trust ~ a1_w*Warmth + a1_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Integrity_Trust ~ a2_w*Warmth + a2_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  PostDV_Binary ~ c_w*Warmth + c_p*Personalization + c_wp*Warmth_x_Personalization +
    b1*Emotional_Trust + b2*Integrity_Trust +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Emotional_Trust ~~ Integrity_Trust

  ind_w_emotional := a1_w*b1
  ind_w_integrity := a2_w*b2
  ind_w_total := a1_w*b1 + a2_w*b2
  direct_w := c_w
  total_w := c_w + ind_w_total

  ind_p_emotional := a1_p*b1
  ind_p_integrity := a2_p*b2
  ind_p_total := a1_p*b1 + a2_p*b2
  direct_p := c_p
  total_p := c_p + ind_p_total

  direct_wp := c_wp
'

trust_measurement_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3

  Emotional_Trust ~~ Integrity_Trust
'

trust_one_factor_model <- '
  Trust_Common =~ ET_1 + ET_2 + ET_3 + CTI_1 + CTI_2 + CTI_3
'

trust_common_method_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3
  Common_Method =~ 1*ET_1 + 1*ET_2 + 1*ET_3 + 1*CTI_1 + 1*CTI_2 + 1*CTI_3

  Emotional_Trust ~~ Integrity_Trust
  Common_Method ~~ 0*Emotional_Trust
  Common_Method ~~ 0*Integrity_Trust
'

binary_sem_common_method_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3
  Common_Method =~ 1*ET_1 + 1*ET_2 + 1*ET_3 + 1*CTI_1 + 1*CTI_2 + 1*CTI_3

  Emotional_Trust ~ a1_w*Warmth + a1_p*Personalization + a1_pre*PreDV_Binary +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Integrity_Trust ~ a2_w*Warmth + a2_p*Personalization + a2_pre*PreDV_Binary +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  PostDV_Binary ~ c_w*Warmth + c_p*Personalization + c_wp*Warmth_x_Personalization + c_pre*PreDV_Binary +
    b1*Emotional_Trust + b2*Integrity_Trust +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Emotional_Trust ~~ Integrity_Trust
  Common_Method ~~ 0*Emotional_Trust
  Common_Method ~~ 0*Integrity_Trust

  ind_w_emotional := a1_w*b1
  ind_w_integrity := a2_w*b2
  ind_w_total := a1_w*b1 + a2_w*b2
  direct_w := c_w
  total_w := c_w + ind_w_total

  ind_p_emotional := a1_p*b1
  ind_p_integrity := a2_p*b2
  ind_p_total := a1_p*b1 + a2_p*b2
  direct_p := c_p
  total_p := c_p + ind_p_total

  direct_wp := c_wp

'

binary_sem_without_pre_common_method_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3
  Common_Method =~ 1*ET_1 + 1*ET_2 + 1*ET_3 + 1*CTI_1 + 1*CTI_2 + 1*CTI_3

  Emotional_Trust ~ a1_w*Warmth + a1_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Integrity_Trust ~ a2_w*Warmth + a2_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  PostDV_Binary ~ c_w*Warmth + c_p*Personalization + c_wp*Warmth_x_Personalization +
    b1*Emotional_Trust + b2*Integrity_Trust +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Emotional_Trust ~~ Integrity_Trust
  Common_Method ~~ 0*Emotional_Trust
  Common_Method ~~ 0*Integrity_Trust

  ind_w_emotional := a1_w*b1
  ind_w_integrity := a2_w*b2
  ind_w_total := a1_w*b1 + a2_w*b2
  direct_w := c_w
  total_w := c_w + ind_w_total

  ind_p_emotional := a1_p*b1
  ind_p_integrity := a2_p*b2
  ind_p_total := a1_p*b1 + a2_p*b2
  direct_p := c_p
  total_p := c_p + ind_p_total

  direct_wp := c_wp
'

defined_effect_cols <- c(
  "lhs", "op", "rhs", "label", "est", "se", "z", "pvalue",
  "ci.lower", "ci.upper", "std.all", "std.nox"
)

extract_defined_effects <- function(parameter_estimates) {
  defined_effects <- parameter_estimates[parameter_estimates$op == ":=", , drop = FALSE]
  defined_effects[, intersect(defined_effect_cols, names(defined_effects)), drop = FALSE]
}

extract_fit_measures <- function(fit, fit_measure_names) {
  values <- lapply(fit_measure_names, function(measure) {
    tryCatch(
      unname(fitMeasures(fit, measure)),
      error = function(e) NA_real_
    )
  })

  data.frame(
    measure = fit_measure_names,
    value = unlist(values, use.names = FALSE),
    row.names = NULL
  )
}

fit_measures_wide <- function(fit, model_name, fit_measure_names) {
  fit_df <- extract_fit_measures(fit, fit_measure_names)
  values <- as.list(fit_df$value)
  names(values) <- fit_df$measure
  data.frame(
    model = model_name,
    values,
    check.names = FALSE,
    row.names = NULL
  )
}

calculate_ave_cr <- function(parameter_estimates, indicator_map) {
  rows <- lapply(names(indicator_map), function(construct) {
    indicators <- indicator_map[[construct]]
    loadings <- parameter_estimates[
      parameter_estimates$op == "=~" &
        parameter_estimates$lhs == construct &
        parameter_estimates$rhs %in% indicators,
      ,
      drop = FALSE
    ]
    loadings <- loadings[match(indicators, loadings$rhs), , drop = FALSE]
    lambda <- loadings$std.all

    residuals <- parameter_estimates[
      parameter_estimates$op == "~~" &
        parameter_estimates$lhs == parameter_estimates$rhs &
        parameter_estimates$lhs %in% indicators,
      ,
      drop = FALSE
    ]
    residuals <- residuals[match(indicators, residuals$lhs), , drop = FALSE]
    theta <- residuals$std.all
    missing_theta <- is.na(theta)
    theta[missing_theta] <- 1 - lambda[missing_theta]^2

    lambda_sq_sum <- sum(lambda^2, na.rm = TRUE)
    theta_sum <- sum(theta, na.rm = TRUE)
    lambda_sum <- sum(lambda, na.rm = TRUE)

    data.frame(
      construct = construct,
      indicators = paste(indicators, collapse = ", "),
      n_indicators = length(indicators),
      min_std_loading = min(lambda, na.rm = TRUE),
      max_std_loading = max(lambda, na.rm = TRUE),
      ave = lambda_sq_sum / (lambda_sq_sum + theta_sum),
      cr = lambda_sum^2 / (lambda_sum^2 + theta_sum),
      ave_ge_0_50 = lambda_sq_sum / (lambda_sq_sum + theta_sum) >= 0.50,
      cr_ge_0_70 = lambda_sum^2 / (lambda_sum^2 + theta_sum) >= 0.70,
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(rows)
}

calculate_discriminant_validity <- function(parameter_estimates, reliability_table) {
  constructs <- reliability_table$construct
  latent_covs <- parameter_estimates[
    parameter_estimates$op == "~~" &
      parameter_estimates$lhs != parameter_estimates$rhs &
      parameter_estimates$lhs %in% constructs &
      parameter_estimates$rhs %in% constructs,
    ,
    drop = FALSE
  ]

  if (nrow(latent_covs) == 0) {
    return(data.frame())
  }

  latent_covs$sqrt_ave_lhs <- sqrt(reliability_table$ave[match(latent_covs$lhs, reliability_table$construct)])
  latent_covs$sqrt_ave_rhs <- sqrt(reliability_table$ave[match(latent_covs$rhs, reliability_table$construct)])
  latent_covs$latent_correlation <- latent_covs$std.all
  latent_covs$squared_correlation <- latent_covs$std.all^2
  latent_covs$fornell_larcker_pass <- (
    latent_covs$sqrt_ave_lhs > abs(latent_covs$latent_correlation) &
      latent_covs$sqrt_ave_rhs > abs(latent_covs$latent_correlation)
  )

  latent_covs[, c(
    "lhs", "rhs", "latent_correlation", "squared_correlation",
    "sqrt_ave_lhs", "sqrt_ave_rhs", "fornell_larcker_pass"
  ), drop = FALSE]
}

extract_admissibility_checks <- function(fit, parameter_estimates = NULL) {
  if (is.null(parameter_estimates)) {
    parameter_estimates <- parameterEstimates(fit, standardized = TRUE)
  }

  latent_names <- tryCatch(lavNames(fit, type = "lv"), error = function(e) character())
  observed_names <- tryCatch(lavNames(fit, type = "ov"), error = function(e) character())

  variances <- parameter_estimates[
    parameter_estimates$op == "~~" &
      parameter_estimates$lhs == parameter_estimates$rhs,
    ,
    drop = FALSE
  ]

  if (nrow(variances) == 0) {
    return(data.frame())
  }

  variances$variable_type <- ifelse(
    variances$lhs %in% latent_names,
    "latent",
    ifelse(variances$lhs %in% observed_names, "observed", "unknown")
  )
  variances$negative_variance <- variances$est < 0
  variances$near_zero_variance <- variances$est >= 0 & variances$est < 1e-6
  variances$negative_std_variance <- variances$std.all < 0

  variances[, c(
    "lhs", "variable_type", "est", "se", "z", "pvalue", "std.all",
    "negative_variance", "near_zero_variance", "negative_std_variance"
  ), drop = FALSE]
}

harman_single_factor_check <- function(data, items) {
  d <- data[complete.cases(data[, items, drop = FALSE]), items, drop = FALSE]
  cor_mat <- cor(d)
  eigen_values <- eigen(cor_mat, symmetric = TRUE, only.values = TRUE)$values

  data.frame(
    n = nrow(d),
    n_items = length(items),
    first_eigenvalue = eigen_values[[1]],
    first_factor_variance_pct = 100 * eigen_values[[1]] / sum(eigen_values),
    below_50_pct_rule = (100 * eigen_values[[1]] / sum(eigen_values)) < 50,
    stringsAsFactors = FALSE
  )
}

run_measurement_diagnostics <- function(df_model,
                                        indicator_map,
                                        two_factor_model,
                                        one_factor_model,
                                        common_method_model) {
  items <- unlist(indicator_map, use.names = FALSE)
  fit_measure_names <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")

  fit_two_factor <- lavaan::cfa(two_factor_model, data = df_model, estimator = "ML")
  fit_one_factor <- lavaan::cfa(one_factor_model, data = df_model, estimator = "ML")
  fit_common_method <- lavaan::cfa(common_method_model, data = df_model, estimator = "ML")

  fit_comparison <- dplyr::bind_rows(
    fit_measures_wide(fit_two_factor, "Two-factor trust CFA", fit_measure_names),
    fit_measures_wide(fit_one_factor, "One-factor trust CFA", fit_measure_names),
    fit_measures_wide(fit_common_method, "Two-factor CFA + common method factor", fit_measure_names)
  )

  two_factor_cfi <- fit_comparison$cfi[fit_comparison$model == "Two-factor trust CFA"]
  two_factor_rmsea <- fit_comparison$rmsea[fit_comparison$model == "Two-factor trust CFA"]
  fit_comparison$delta_cfi_vs_two_factor <- fit_comparison$cfi - two_factor_cfi
  fit_comparison$delta_rmsea_vs_two_factor <- fit_comparison$rmsea - two_factor_rmsea

  method_pe <- parameterEstimates(fit_common_method, standardized = TRUE)
  method_loadings <- method_pe[
    method_pe$op == "=~" & method_pe$lhs == "Common_Method",
    c("lhs", "op", "rhs", "est", "se", "z", "pvalue", "std.all"),
    drop = FALSE
  ]
  method_variance_param <- method_pe[
    method_pe$op == "~~" &
      method_pe$lhs == "Common_Method" &
      method_pe$rhs == "Common_Method",
    ,
    drop = FALSE
  ]
  method_variance <- data.frame(
    method_variance_est = method_variance_param$est,
    method_variance_pvalue = method_variance_param$pvalue,
    mean_method_variance_pct = mean(method_loadings$std.all^2, na.rm = TRUE) * 100,
    max_method_variance_pct = max(method_loadings$std.all^2, na.rm = TRUE) * 100,
    stringsAsFactors = FALSE
  )

  list(
    harman = harman_single_factor_check(df_model, items),
    fit_comparison = fit_comparison,
    common_method_loadings = method_loadings,
    common_method_variance = method_variance
  )
}

extract_labelled_structural_paths <- function(parameter_estimates, path_labels) {
  paths <- parameter_estimates[
    parameter_estimates$op == "~" & parameter_estimates$label %in% path_labels,
    ,
    drop = FALSE
  ]

  paths[, c("label", "lhs", "rhs", "est", "se", "z", "pvalue", "std.all"), drop = FALSE]
}

compare_common_method_paths <- function(main_parameter_estimates,
                                        common_method_fit,
                                        path_labels) {
  method_parameter_estimates <- parameterEstimates(
    common_method_fit,
    standardized = TRUE,
    ci = TRUE,
    level = 0.95
  )

  main_paths <- extract_labelled_structural_paths(main_parameter_estimates, path_labels)
  method_paths <- extract_labelled_structural_paths(method_parameter_estimates, path_labels)

  names(main_paths)[names(main_paths) %in% c("est", "se", "z", "pvalue", "std.all")] <-
    paste0("main_", names(main_paths)[names(main_paths) %in% c("est", "se", "z", "pvalue", "std.all")])
  names(method_paths)[names(method_paths) %in% c("est", "se", "z", "pvalue", "std.all")] <-
    paste0("cmb_", names(method_paths)[names(method_paths) %in% c("est", "se", "z", "pvalue", "std.all")])

  comparison <- merge(
    main_paths,
    method_paths,
    by = c("label", "lhs", "rhs"),
    all = TRUE,
    sort = FALSE
  )
  comparison$delta_std_all <- comparison$cmb_std.all - comparison$main_std.all
  comparison
}

run_binary_sem <- function(df_model,
                           model,
                           label,
                           output_prefix,
                           n_boot,
                           write_outputs,
                           output_dir,
                           common_method_model = binary_sem_common_method_model,
                           path_labels = structural_path_labels) {
  cat(sprintf("\n=== Binary SEM data: %s ===\n", label))
  cat(sprintf("N = %d\n", nrow(df_model)))
  cat("\nPostDV_Binary counts after 0/1 recode:\n")
  print(table(df_model$PostDV_Binary, useNA = "ifany"))

  if ("PreDV_Binary" %in% names(df_model)) {
    cat("\nPreDV_Binary x PostDV_Binary counts after 0/1 recode:\n")
    print(table(df_model$PreDV_Binary, df_model$PostDV_Binary, useNA = "ifany"))
  }

  measurement_diagnostics <- run_measurement_diagnostics(
    df_model = df_model,
    indicator_map = latent_indicator_map,
    two_factor_model = trust_measurement_model,
    one_factor_model = trust_one_factor_model,
    common_method_model = trust_common_method_model
  )

  cat("\n=== Harman single-factor diagnostic for trust items ===\n")
  print(measurement_diagnostics$harman, row.names = FALSE)

  cat("\n=== Trust measurement and common-method CFA fit comparison ===\n")
  print(measurement_diagnostics$fit_comparison, row.names = FALSE)

  cat("\n=== Equal-loading common method factor variance ===\n")
  print(measurement_diagnostics$common_method_variance, row.names = FALSE)

  # lavaan does not allow bootstrap SEs with WLSMV; use DWLS only when
  # SEM_BINARY_BOOTSTRAPS is set.
  estimator <- if (n_boot > 0) "DWLS" else "WLSMV"
  cat(sprintf(
    "\nEstimator = %s; bootstrap draws = %d%s\n",
    estimator,
    n_boot,
    if (n_boot == 0) " (robust WLSMV SEs)" else " (bootstrap SEs/CIs)"
  ))

  sem_args <- list(
    model = model,
    data = df_model,
    ordered = "PostDV_Binary",
    estimator = estimator,
    fixed.x = TRUE
  )

  if (n_boot > 0) {
    sem_args$se <- "bootstrap"
    sem_args$bootstrap <- n_boot
  }

  fit <- do.call(lavaan::sem, sem_args)

  cat(sprintf("\n=== Binary SEM summary: %s ===\n", label))
  print(summary(
    fit,
    standardized = TRUE,
    ci = TRUE,
    rsquare = TRUE,
    fit.measures = TRUE
  ))

  parameter_estimates <- parameterEstimates(
    fit,
    standardized = TRUE,
    ci = TRUE,
    boot.ci.type = if (n_boot > 0) "perc" else "norm",
    level = 0.95
  )
  defined_effects <- extract_defined_effects(parameter_estimates)

  latent_reliability <- calculate_ave_cr(parameter_estimates, latent_indicator_map)
  discriminant_validity <- calculate_discriminant_validity(parameter_estimates, latent_reliability)
  sem_admissibility_checks <- extract_admissibility_checks(fit, parameter_estimates)

  cat(sprintf("\n=== Binary SEM latent reliability and convergent validity: %s ===\n", label))
  print(latent_reliability, row.names = FALSE)

  cat(sprintf("\n=== Binary SEM discriminant validity check: %s ===\n", label))
  print(discriminant_validity, row.names = FALSE)

  cat(sprintf("\n=== Binary SEM variance admissibility checks: %s ===\n", label))
  print(sem_admissibility_checks, row.names = FALSE)

  cat(sprintf("\n=== Binary SEM defined direct, indirect, and total effects: %s ===\n", label))
  print(defined_effects, row.names = FALSE)

  fit_measure_names <- if (n_boot > 0) {
    c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr")
  } else {
    c(
      "chisq.scaled", "df.scaled", "pvalue.scaled",
      "cfi.scaled", "tli.scaled", "rmsea.scaled", "srmr"
    )
  }
  fit_measures_df <- extract_fit_measures(fit, fit_measure_names)

  cat(sprintf("\n=== Binary SEM fit measures: %s ===\n", label))
  print(fit_measures_df, row.names = FALSE)

  common_method_sem_args <- list(
    model = common_method_model,
    data = df_model,
    ordered = "PostDV_Binary",
    estimator = estimator,
    fixed.x = TRUE
  )
  fit_common_method <- do.call(lavaan::sem, common_method_sem_args)
  common_method_fit_measures <- extract_fit_measures(fit_common_method, fit_measure_names)
  common_method_admissibility_checks <- extract_admissibility_checks(fit_common_method)
  common_method_path_comparison <- compare_common_method_paths(
    main_parameter_estimates = parameter_estimates,
    common_method_fit = fit_common_method,
    path_labels = path_labels
  )

  cat(sprintf("\n=== Binary common-method-adjusted SEM fit measures: %s ===\n", label))
  print(common_method_fit_measures, row.names = FALSE)

  cat(sprintf("\n=== Binary common-method-adjusted variance admissibility checks: %s ===\n", label))
  print(common_method_admissibility_checks, row.names = FALSE)

  cat(sprintf("\n=== Binary common-method-adjusted structural path comparison: %s ===\n", label))
  print(common_method_path_comparison, row.names = FALSE)

  if (write_outputs) {
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    readr::write_csv(df_model, file.path(output_dir, paste0(output_prefix, "_data.csv")))
    readr::write_csv(parameter_estimates, file.path(output_dir, paste0(output_prefix, "_parameter_estimates.csv")))
    readr::write_csv(defined_effects, file.path(output_dir, paste0(output_prefix, "_defined_effects.csv")))
    readr::write_csv(fit_measures_df, file.path(output_dir, paste0(output_prefix, "_fit_measures.csv")))
    readr::write_csv(
      latent_reliability,
      file.path(output_dir, paste0(output_prefix, "_latent_reliability_ave_cr.csv"))
    )
    readr::write_csv(
      discriminant_validity,
      file.path(output_dir, paste0(output_prefix, "_discriminant_validity.csv"))
    )
    readr::write_csv(
      sem_admissibility_checks,
      file.path(output_dir, paste0(output_prefix, "_admissibility_checks.csv"))
    )
    readr::write_csv(
      measurement_diagnostics$harman,
      file.path(output_dir, paste0(output_prefix, "_common_method_harman.csv"))
    )
    readr::write_csv(
      measurement_diagnostics$fit_comparison,
      file.path(output_dir, paste0(output_prefix, "_measurement_fit_comparison.csv"))
    )
    readr::write_csv(
      measurement_diagnostics$common_method_loadings,
      file.path(output_dir, paste0(output_prefix, "_common_method_loadings.csv"))
    )
    readr::write_csv(
      measurement_diagnostics$common_method_variance,
      file.path(output_dir, paste0(output_prefix, "_common_method_variance.csv"))
    )
    readr::write_csv(
      common_method_fit_measures,
      file.path(output_dir, paste0(output_prefix, "_common_method_fit_measures.csv"))
    )
    readr::write_csv(
      common_method_admissibility_checks,
      file.path(output_dir, paste0(output_prefix, "_common_method_admissibility_checks.csv"))
    )
    readr::write_csv(
      common_method_path_comparison,
      file.path(output_dir, paste0(output_prefix, "_common_method_path_comparison.csv"))
    )
  }

  invisible(list(
    fit = fit,
    parameter_estimates = parameter_estimates,
    defined_effects = defined_effects,
    fit_measures = fit_measures_df,
    latent_reliability = latent_reliability,
    discriminant_validity = discriminant_validity,
    admissibility_checks = sem_admissibility_checks,
    measurement_diagnostics = measurement_diagnostics,
    common_method_fit = fit_common_method,
    common_method_fit_measures = common_method_fit_measures,
    common_method_admissibility_checks = common_method_admissibility_checks,
    common_method_path_comparison = common_method_path_comparison
  ))
}

cat(sprintf("Reading data from %s\n", input_file))
df <- clean_survey_data(input_file)
cat(sprintf("After Python-equivalent cleaning: N = %d\n", nrow(df)))
cat("\nCell counts after initial cleaning:\n")
print(table(df$Warmth, df$Personalization, useNA = "ifany"))

condition_cells <- data.frame(
  Warmth = c(0, 0, 1, 1),
  Personalization = c(0, 1, 0, 1)
)

# Keep this aligned with the notebook's logistic section: binary models use
# ancova_df after the same slider IQR and Cook's D screens.
df_clean <- remove_iqr_outliers_by_cell(
  data = df,
  outcome = "Slider_Difference",
  group_vars = c("Warmth", "Personalization"),
  cells = condition_cells,
  iqr_mult = 1.5
)

ancova_required <- c(
  "Slider_Difference", "Warmth", "Personalization",
  "PostDV_Binary", "PreDV_Binary",
  "ET_1", "ET_2", "ET_3", "CTI_1", "CTI_2", "CTI_3",
  "Emotional_Trust", "Integrity_Trust", "Digital_Literacy", "UF_Q1",
  "P_Extraversion", "P_Agreeableness", "P_Openness",
  "P_Conscientiousness", "P_Neuroticism",
  "SD_ChangeDirection", "DV_Change_Diff"
)

ancova_df <- df_clean[
  complete.cases(df_clean[, ancova_required, drop = FALSE]),
  ,
  drop = FALSE
]
ancova_df$Warmth <- as.integer(ancova_df$Warmth)
ancova_df$Personalization <- as.integer(ancova_df$Personalization)

ols_formula <- as.formula(
  paste(
    "Slider_Difference ~ factor(Warmth) * factor(Personalization)",
    "+ Emotional_Trust + Integrity_Trust + Digital_Literacy + UF_Q1",
    "+ P_Extraversion + P_Agreeableness + P_Openness",
    "+ P_Conscientiousness + P_Neuroticism"
  )
)

ancova_df <- remove_outliers_by_cooks_d(
  data = ancova_df,
  formula = ols_formula,
  threshold_mult = 4
)

cat("\nCell counts after both outlier removal steps:\n")
print(table(ancova_df$Warmth, ancova_df$Personalization, useNA = "ifany"))

set.seed(seed)

df_model_binary_all <- prepare_binary_sem_data(ancova_df)
binary_sem <- run_binary_sem(
  df_model = df_model_binary_all,
  model = binary_sem_model,
  label = "All samples with PreDV_Binary as IV",
  output_prefix = "binary_sem_all",
  n_boot = n_boot,
  write_outputs = write_outputs,
  output_dir = output_dir
)

# Supplementary analysis: restrict to participants who initially approved.
# The raw Qualtrics binary coding uses 1 = Yes and 2 = No; prepare_binary_sem_data()
# then recodes this to 0/1 for the binary SEM.
ancova_df_initial_yes <- ancova_df[
  !is.na(ancova_df$PreDV_Binary) & ancova_df$PreDV_Binary == 1,
  ,
  drop = FALSE
]
cat("\n=== Supplementary binary SEM subset: initially yes ===\n")
cat(sprintf("N before SEM complete-case filtering = %d\n", nrow(ancova_df_initial_yes)))
cat("\nRaw PreDV_Binary x PostDV_Binary counts before 0/1 recode:\n")
print(table(
  ancova_df_initial_yes$PreDV_Binary,
  ancova_df_initial_yes$PostDV_Binary,
  useNA = "ifany"
))

df_model_binary_initial_yes <- prepare_binary_sem_data(ancova_df_initial_yes)
binary_sem_initial_yes <- run_binary_sem(
  df_model = df_model_binary_initial_yes,
  model = binary_sem_without_pre_model,
  label = "Initially yes participants only",
  output_prefix = "binary_sem_initial_yes",
  n_boot = n_boot,
  write_outputs = write_outputs,
  output_dir = output_dir,
  common_method_model = binary_sem_without_pre_common_method_model,
  path_labels = structural_path_labels_without_pre
)

if (write_outputs) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(df_clean, file.path(output_dir, "df_after_iqr.csv"))
  readr::write_csv(ancova_df, file.path(output_dir, "ancova_after_iqr_and_cooksd.csv"))
  readr::write_csv(attr(df_clean, "iqr_summary"), file.path(output_dir, "iqr_summary.csv"))
  readr::write_csv(attr(ancova_df, "cooks_diagnostics"), file.path(output_dir, "cooks_diagnostics.csv"))

  cat(sprintf("\nWrote binary SEM output tables to %s\n", output_dir))
}
