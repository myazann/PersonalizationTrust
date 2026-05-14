# =============================================================================
# StormShield SEM - Monte Carlo power analysis (self-contained, hybrid engine)
# =============================================================================
# Purpose: estimate statistical power for the focal effects in the parallel
# mediation SEM, using the fitted model as the data-generating population.
#
# -----------------------------------------------------------------------------
# IMPORTANT - THIS SCRIPT IS SELF-CONTAINED
# -----------------------------------------------------------------------------
# It carries its own copy of the data-cleaning pipeline, the outlier-removal
# steps, and the SEM model syntax, copied from stormshield_sem.R. It does NOT
# source the main script. If you revise the cleaning pipeline, the outlier
# rules, the covariate set, or the model syntax in stormshield_sem.R, you MUST
# mirror those edits here or the power analysis will describe a different model
# than the one you report. The blocks copied from the main script are fenced
# with "### --- BEGIN/END copied from stormshield_sem.R --- ###".
#
# -----------------------------------------------------------------------------
# WHY THIS REWRITE EXISTS
# -----------------------------------------------------------------------------
# A previous version generated data with lavExport() + simulateData(). That
# pipeline does not work for a fitted, fixed.x = TRUE model: lavExport() does
# not reliably round-trip to a simulable lavaan population model, and with
# fixed.x = TRUE the fitted object contains no distribution for the exogenous
# variables, so simulateData() has nothing to draw them from. The result was
# 0/8000 successful fits reported (misleadingly) as "0% power".
#
# -----------------------------------------------------------------------------
# HYBRID SIMULATION ENGINE
# -----------------------------------------------------------------------------
# Because Warmth, Personalization, the interaction term, and all covariates are
# treated as fixed (fixed.x = TRUE), they are not modelled distributionally.
# The correct design is therefore hybrid:
#   (1) EXOGENOUS variables (Warmth, Personalization, Warmth_x_Personalization,
#       Digital_Literacy, UF_Q1, the five personality scores) are RESAMPLED
#       with replacement, as whole rows, from the real analysis data. This
#       preserves the true 0/1 structure of the manipulations, the real cell
#       balance, and the real covariate covariances exactly - no normal
#       approximation.
#   (2) ENDOGENOUS variables (the six trust indicators ET_1..3 / CTI_1..3 and
#       Slider_Difference) are GENERATED from the fitted measurement and
#       structural equations, adding draws of the latent variables and the
#       residuals from their fitted (co)variances.
#
# Caveat to report: because exogenous rows are resampled from the observed
# data, the covariate DISTRIBUTION is fixed at what was observed. Simulating
# N > analysis_n samples those rows with replacement; it does not extrapolate
# to covariate values outside the observed range. This is standard for
# conditional / fixed.x SEM power analysis and should be stated as such.
#
# Dependencies: lavaan, readr, dplyr, MASS.
# =============================================================================

required_packages <- c("readr", "dplyr", "lavaan", "MASS")
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
# MASS is used only for mvrnorm(); referenced as MASS::mvrnorm to avoid masking.

# -----------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# -----------------------------------------------------------------------------
parse_int_env <- function(name, default) {
  value <- Sys.getenv(name, unset = as.character(default))
  parsed <- suppressWarnings(as.integer(value))
  if (is.na(parsed) || parsed < 0) default else parsed
}
parse_num_csv_env <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) return(default)
  parsed <- suppressWarnings(as.numeric(strsplit(value, ",")[[1]]))
  if (any(is.na(parsed))) default else parsed
}

input_file <- Sys.getenv("SEM_INPUT_FILE", unset = "Results.csv")

# Number of Monte Carlo replications per N. 1000 gives a power estimate with a
# Monte Carlo SE of at most ~1.6 percentage points (worst case at power = .5).
# Use 200-500 for a quick exploratory run; 1000-2000 for a reportable one.
n_reps <- parse_int_env("POWER_REPS", 1000)

# Sample sizes for the sensitivity curve. The analysis N is added in
# automatically once it is known, so the post-hoc read is directly comparable.
n_grid <- parse_num_csv_env(
  "POWER_N_GRID",
  c(150, 200, 250, 350, 400, 500, 600)
)

alpha <- as.numeric(Sys.getenv("POWER_ALPHA", unset = "0.05"))
seed <- parse_int_env("POWER_SEED", 2024)
output_dir <- Sys.getenv("POWER_OUTPUT_DIR", unset = "power_outputs")
write_outputs <- tolower(Sys.getenv("POWER_WRITE_OUTPUTS", unset = "1")) %in%
  c("1", "true", "yes", "y")

# Significance test for simulated fits. "standard" (Wald z) is used inside the
# simulation loop because bootstrapping every replication is infeasible. For
# the indirect effects this means the normal-theory test of a product term is
# slightly conservative relative to the percentile bootstrap used in the main
# analysis; treat indirect-effect power as a mild LOWER bound. Direct effects
# and the interaction are single coefficients and are unaffected.
sim_se <- "standard"

# =============================================================================
# ### --- BEGIN copied from stormshield_sem.R --- ###
# Keep this block in sync with the main analysis script.
# =============================================================================

latent_indicator_map <- list(
  Emotional_Trust = c("ET_1", "ET_2", "ET_3"),
  Integrity_Trust = c("CTI_1", "CTI_2", "CTI_3")
)
trust_item_vars <- unlist(latent_indicator_map, use.names = FALSE)

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
  
  # Personality item missingness: mean imputation (raw 1-7 responses, before
  # reverse-coding).
  personality_columns <- intersect(personality_columns, names(df))
  df[personality_columns] <- lapply(df[personality_columns], function(x) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    x
  })
  
  df$Slider_Difference <- df$PostDV_Slider_1 - df$PreDV_Slider_1
  
  df$SD_ChangeDirection <- "No Change"
  df$SD_ChangeDirection <- set_value_where(df$SD_ChangeDirection, df$Slider_Difference > 0, "Increase")
  df$SD_ChangeDirection <- set_value_where(df$SD_ChangeDirection, df$Slider_Difference < 0, "Decrease")
  
  df$DV_Binary_Diff <- df$PostDV_Binary - df$PreDV_Binary
  df$DV_Change_Diff <- "ChangeTrust"
  df$DV_Change_Diff <- set_value_where(df$DV_Change_Diff, (df$PostDV_Binary == 2) & (df$PreDV_Binary == 2), "Distrust")
  df$DV_Change_Diff <- set_value_where(df$DV_Change_Diff, (df$PostDV_Binary == 1) & (df$PreDV_Binary == 1), "Trust")
  df$DV_Change_Diff <- set_value_where(df$DV_Change_Diff, (df$PostDV_Binary == 2) & (df$PreDV_Binary == 1), "ChangeDistrust")
  
  df$Warmth_Check <- rowMeans(select_existing(df, warmth_columns), na.rm = TRUE)
  df$Personalization_Check <- rowMeans(select_existing(df, personalization_columns), na.rm = TRUE)
  df$Emotional_Trust <- rowMeans(select_existing(df, emotional_trust_columns), na.rm = TRUE)
  df$Integrity_Trust <- rowMeans(select_existing(df, integrity_trust_columns), na.rm = TRUE)
  df$Digital_Literacy <- rowMeans(select_existing(df, digital_literacy_columns), na.rm = TRUE)
  df$UF_Q1 <- as.integer(df$UF_Q1)
  
  df$IV_Congruence <- 1L
  df$IV_Congruence <- set_value_where(df$IV_Congruence, (df$Warmth + df$Personalization) == 1, 0L)
  
  df$P_Extraversion <- (6 - df$Personality_1) + df$Personality_6
  df$P_Agreeableness <- df$Personality_2 + (6 - df$Personality_7)
  df$P_Conscientiousness <- (6 - df$Personality_3) + df$Personality_8
  df$P_Neuroticism <- (6 - df$Personality_4) + df$Personality_9
  df$P_Openness <- (6 - df$Personality_5) + df$Personality_10
  
  duration <- df$`Duration (in seconds)`
  df <- df[which(!is.na(duration) & duration > 300 & duration < 2500), , drop = FALSE]
  
  df
}

remove_iqr_outliers_by_cell <- function(data, outcome, group_vars, cells,
                                        iqr_mult = 1.5) {
  flagged <- rep(FALSE, nrow(data))
  for (i in seq_len(nrow(cells))) {
    cell_mask <- rep(TRUE, nrow(data))
    for (j in seq_along(group_vars)) {
      group_var <- group_vars[j]
      group_value <- cells[[group_var]][i]
      cell_mask <- cell_mask & !is.na(data[[group_var]]) & data[[group_var]] == group_value
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
  }
  data[!flagged, , drop = FALSE]
}

remove_outliers_by_cooks_d <- function(data, formula, threshold_mult = 4) {
  required <- all.vars(formula)
  d <- data[complete.cases(data[, required, drop = FALSE]), , drop = FALSE]
  fit <- lm(formula, data = d)
  cooks <- cooks.distance(fit)
  threshold <- threshold_mult / nrow(d)
  flagged <- cooks > threshold
  d[!flagged, , drop = FALSE]
}

prepare_sem_data <- function(ancova_df) {
  df_sem <- ancova_df
  df_sem$Warmth_x_Personalization <- df_sem$Warmth * df_sem$Personalization
  df_sem$IV_Congruence <- as.integer(df_sem$Warmth == df_sem$Personalization)
  
  continuous_vars <- c(
    "Slider_Difference", "Digital_Literacy",
    "P_Extraversion", "P_Agreeableness", "P_Openness",
    "P_Conscientiousness", "P_Neuroticism", "UF_Q1"
  )
  df_model <- df_sem
  df_model[continuous_vars] <- lapply(df_model[continuous_vars], z_score)
  
  model_vars <- c(
    "Slider_Difference", trust_item_vars,
    "Warmth", "Personalization", "Warmth_x_Personalization",
    "Digital_Literacy", "P_Extraversion", "P_Agreeableness", "P_Openness",
    "P_Conscientiousness", "P_Neuroticism", "UF_Q1"
  )
  df_model <- df_model[complete.cases(df_model[, model_vars, drop = FALSE]), model_vars, drop = FALSE]
  df_model[] <- lapply(df_model, as.numeric)
  df_model
}

# Latent-variable PARALLEL mediation SEM.
sem_model <- '
  Emotional_Trust =~ ET_1 + ET_2 + ET_3
  Integrity_Trust =~ CTI_1 + CTI_2 + CTI_3

  Emotional_Trust ~ a1_w*Warmth + a1_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Integrity_Trust ~ a2_w*Warmth + a2_p*Personalization +
    Digital_Literacy + P_Extraversion + P_Agreeableness + P_Openness +
    P_Conscientiousness + P_Neuroticism + UF_Q1

  Slider_Difference ~ c_w*Warmth + c_p*Personalization + c_wp*Warmth_x_Personalization +
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

# =============================================================================
# ### --- END copied from stormshield_sem.R --- ###
# =============================================================================

# -----------------------------------------------------------------------------
# Build the analysis data and fit the model once
# -----------------------------------------------------------------------------
cat(sprintf("Reading data from %s\n", input_file))
if (!file.exists(input_file)) {
  stop("Could not find input data file '", input_file,
       "'. Set SEM_INPUT_FILE to its path.", call. = FALSE)
}

df <- clean_survey_data(input_file)

condition_cells <- data.frame(
  Warmth = c(0, 0, 1, 1),
  Personalization = c(0, 1, 0, 1)
)
df_clean <- remove_iqr_outliers_by_cell(
  data = df, outcome = "Slider_Difference",
  group_vars = c("Warmth", "Personalization"),
  cells = condition_cells, iqr_mult = 1.5
)

ancova_required <- c(
  "Slider_Difference", "Warmth", "Personalization",
  "ET_1", "ET_2", "ET_3", "CTI_1", "CTI_2", "CTI_3",
  "Emotional_Trust", "Integrity_Trust", "Digital_Literacy", "UF_Q1",
  "P_Extraversion", "P_Agreeableness", "P_Openness",
  "P_Conscientiousness", "P_Neuroticism",
  "SD_ChangeDirection", "DV_Change_Diff"
)
ols_formula <- as.formula(
  paste(
    "Slider_Difference ~ factor(Warmth) * factor(Personalization)",
    "+ Emotional_Trust + Integrity_Trust + Digital_Literacy + UF_Q1",
    "+ P_Extraversion + P_Agreeableness + P_Openness",
    "+ P_Conscientiousness + P_Neuroticism"
  )
)

ancova_df_iqr_only <- df_clean[
  complete.cases(df_clean[, ancova_required, drop = FALSE]), , drop = FALSE
]
ancova_df_iqr_only$Warmth <- as.integer(ancova_df_iqr_only$Warmth)
ancova_df_iqr_only$Personalization <- as.integer(ancova_df_iqr_only$Personalization)

ancova_df <- remove_outliers_by_cooks_d(
  data = ancova_df_iqr_only, formula = ols_formula, threshold_mult = 4
)
df_model <- prepare_sem_data(ancova_df)
analysis_n <- nrow(df_model)

cat(sprintf("Analysis data prepared. N = %d.\n", analysis_n))

fit_sem <- lavaan::sem(
  sem_model, data = df_model, estimator = "ML", fixed.x = TRUE, se = "standard"
)
if (!lavInspect(fit_sem, "converged")) {
  stop("The base SEM did not converge on the real data; cannot run power analysis.",
       call. = FALSE)
}
cat("Base SEM fitted successfully on the real data.\n")

# Ensure the analysis N is on the grid so the post-hoc read is comparable.
n_grid <- sort(unique(c(n_grid, analysis_n)))

cat(sprintf(
  "Monte Carlo settings: %d replications per N, alpha = %.3f, seed = %d.\n",
  n_reps, alpha, seed
))
cat(sprintf("Sample sizes on the sensitivity grid: %s\n",
            paste(n_grid, collapse = ", ")))

# -----------------------------------------------------------------------------
# Extract the population parameters from the fitted model
# -----------------------------------------------------------------------------
# These fitted estimates are treated as the data-generating population values.
pe_all <- parameterEstimates(fit_sem)

get_path <- function(lhs_var, rhs_var) {
  row <- pe_all[pe_all$op == "~" & pe_all$lhs == lhs_var & pe_all$rhs == rhs_var, , drop = FALSE]
  if (nrow(row) == 1) row$est else stop("Path not found: ", lhs_var, " ~ ", rhs_var)
}
get_loading <- function(lv, indicator) {
  row <- pe_all[pe_all$op == "=~" & pe_all$lhs == lv & pe_all$rhs == indicator, , drop = FALSE]
  if (nrow(row) == 1) row$est else stop("Loading not found: ", lv, " =~ ", indicator)
}
get_var <- function(v) {
  row <- pe_all[pe_all$op == "~~" & pe_all$lhs == v & pe_all$rhs == v, , drop = FALSE]
  if (nrow(row) == 1) row$est else stop("Variance not found: ", v)
}
get_intercept <- function(v) {
  row <- pe_all[pe_all$op == "~1" & pe_all$lhs == v, , drop = FALSE]
  if (nrow(row) == 1) row$est else 0  # intercepts may be fixed at 0
}

# Covariates shared by all three structural equations.
covariate_vars <- c(
  "Digital_Literacy", "P_Extraversion", "P_Agreeableness", "P_Openness",
  "P_Conscientiousness", "P_Neuroticism", "UF_Q1"
)
# All exogenous columns the simulated data must carry, in a fixed order.
exo_vars <- c("Warmth", "Personalization", "Warmth_x_Personalization", covariate_vars)

# --- Structural coefficients for Emotional_Trust -----------------------------
et_coef <- c(
  Warmth = get_path("Emotional_Trust", "Warmth"),
  Personalization = get_path("Emotional_Trust", "Personalization")
)
for (cv in covariate_vars) et_coef[cv] <- get_path("Emotional_Trust", cv)
et_resid_var <- get_var("Emotional_Trust")  # residual variance of the latent

# --- Structural coefficients for Integrity_Trust -----------------------------
it_coef <- c(
  Warmth = get_path("Integrity_Trust", "Warmth"),
  Personalization = get_path("Integrity_Trust", "Personalization")
)
for (cv in covariate_vars) it_coef[cv] <- get_path("Integrity_Trust", cv)
it_resid_var <- get_var("Integrity_Trust")

# Residual covariance between the two trust latents.
et_it_resid_cov_row <- pe_all[
  pe_all$op == "~~" &
    ((pe_all$lhs == "Emotional_Trust" & pe_all$rhs == "Integrity_Trust") |
       (pe_all$lhs == "Integrity_Trust" & pe_all$rhs == "Emotional_Trust")),
  , drop = FALSE
]
et_it_resid_cov <- if (nrow(et_it_resid_cov_row) == 1) et_it_resid_cov_row$est else 0

# --- Structural coefficients for Slider_Difference ---------------------------
sd_coef <- c(
  Warmth = get_path("Slider_Difference", "Warmth"),
  Personalization = get_path("Slider_Difference", "Personalization"),
  Warmth_x_Personalization = get_path("Slider_Difference", "Warmth_x_Personalization"),
  Emotional_Trust = get_path("Slider_Difference", "Emotional_Trust"),
  Integrity_Trust = get_path("Slider_Difference", "Integrity_Trust")
)
for (cv in covariate_vars) sd_coef[cv] <- get_path("Slider_Difference", cv)
sd_resid_var <- get_var("Slider_Difference")
sd_intercept <- get_intercept("Slider_Difference")

# --- Measurement model: loadings, indicator intercepts, residual variances ---
measurement <- lapply(names(latent_indicator_map), function(lv) {
  indicators <- latent_indicator_map[[lv]]
  list(
    latent = lv,
    indicators = indicators,
    loadings = vapply(indicators, function(ind) get_loading(lv, ind), numeric(1)),
    intercepts = vapply(indicators, function(ind) get_intercept(ind), numeric(1)),
    resid_var = vapply(indicators, function(ind) get_var(ind), numeric(1))
  )
})
names(measurement) <- names(latent_indicator_map)

# -----------------------------------------------------------------------------
# Hybrid data generator
# -----------------------------------------------------------------------------
# For a target N:
#   1. Resample N whole rows of the EXOGENOUS variables from df_model (with
#      replacement). This fixes the covariate distribution at the observed one.
#   2. Generate the two trust latent scores from their structural equations
#      plus a bivariate-normal residual draw using (et_resid_var, it_resid_var,
#      et_it_resid_cov).
#   3. Generate the six trust indicators from the measurement model:
#      indicator = intercept + loading * latent + N(0, residual variance).
#   4. Generate Slider_Difference from its structural equation plus a normal
#      residual draw.
# Returns a data.frame with exactly the columns sem_model expects.
generate_dataset <- function(target_n, source_data) {
  idx <- sample.int(nrow(source_data), size = target_n, replace = TRUE)
  exo <- source_data[idx, exo_vars, drop = FALSE]
  rownames(exo) <- NULL
  
  # --- Latent trust scores -------------------------------------------------
  et_linpred <- as.numeric(
    et_coef["Warmth"] * exo$Warmth +
      et_coef["Personalization"] * exo$Personalization
  )
  it_linpred <- as.numeric(
    it_coef["Warmth"] * exo$Warmth +
      it_coef["Personalization"] * exo$Personalization
  )
  for (cv in covariate_vars) {
    et_linpred <- et_linpred + et_coef[cv] * exo[[cv]]
    it_linpred <- it_linpred + it_coef[cv] * exo[[cv]]
  }
  
  resid_sigma <- matrix(
    c(et_resid_var, et_it_resid_cov,
      et_it_resid_cov, it_resid_var),
    nrow = 2, byrow = TRUE
  )
  latent_resid <- MASS::mvrnorm(n = target_n, mu = c(0, 0), Sigma = resid_sigma)
  
  emotional_trust <- et_linpred + latent_resid[, 1]
  integrity_trust <- it_linpred + latent_resid[, 2]
  
  # --- Trust indicators from the measurement model -------------------------
  out <- data.frame(row.names = seq_len(target_n))
  latent_scores <- list(
    Emotional_Trust = emotional_trust,
    Integrity_Trust = integrity_trust
  )
  for (lv in names(measurement)) {
    m <- measurement[[lv]]
    score <- latent_scores[[lv]]
    for (k in seq_along(m$indicators)) {
      ind_name <- m$indicators[k]
      out[[ind_name]] <- m$intercepts[k] +
        m$loadings[k] * score +
        rnorm(target_n, mean = 0, sd = sqrt(m$resid_var[k]))
    }
  }
  
  # --- Slider_Difference from its structural equation ----------------------
  sd_linpred <- rep(sd_intercept, target_n) +
    sd_coef["Warmth"] * exo$Warmth +
    sd_coef["Personalization"] * exo$Personalization +
    sd_coef["Warmth_x_Personalization"] * exo$Warmth_x_Personalization +
    sd_coef["Emotional_Trust"] * emotional_trust +
    sd_coef["Integrity_Trust"] * integrity_trust
  for (cv in covariate_vars) {
    sd_linpred <- sd_linpred + sd_coef[cv] * exo[[cv]]
  }
  out$Slider_Difference <- sd_linpred + rnorm(target_n, mean = 0, sd = sqrt(sd_resid_var))
  
  # --- Attach exogenous variables ------------------------------------------
  for (v in exo_vars) out[[v]] <- exo[[v]]
  
  out
}

# -----------------------------------------------------------------------------
# Focal effects
# -----------------------------------------------------------------------------
focal_effects <- list(
  list(name = "Warmth x Personalization interaction (c_wp)", type = "label", key = "c_wp"),
  list(name = "Warmth direct effect (c_w)", type = "label", key = "c_w"),
  list(name = "Personalization direct effect (c_p)", type = "label", key = "c_p"),
  list(name = "Emotional Trust -> Slider (b1)", type = "label", key = "b1"),
  list(name = "Integrity Trust -> Slider (b2)", type = "label", key = "b2"),
  list(name = "Indirect: Warmth via Emotional Trust", type = "defined", key = "ind_w_emotional"),
  list(name = "Indirect: Warmth via Integrity Trust", type = "defined", key = "ind_w_integrity"),
  list(name = "Indirect: Warmth total", type = "defined", key = "ind_w_total"),
  list(name = "Indirect: Personalization via Emotional Trust", type = "defined", key = "ind_p_emotional"),
  list(name = "Indirect: Personalization via Integrity Trust", type = "defined", key = "ind_p_integrity"),
  list(name = "Indirect: Personalization total", type = "defined", key = "ind_p_total")
)

fitted_pe_std <- parameterEstimates(fit_sem, standardized = TRUE)
focal_population_values <- vapply(focal_effects, function(eff) {
  if (eff$type == "label") {
    row <- fitted_pe_std[fitted_pe_std$op == "~" & fitted_pe_std$label == eff$key, , drop = FALSE]
  } else {
    row <- fitted_pe_std[fitted_pe_std$op == ":=" & fitted_pe_std$lhs == eff$key, , drop = FALSE]
  }
  if (nrow(row) == 1) row$est else NA_real_
}, numeric(1))
focal_population_std <- vapply(focal_effects, function(eff) {
  if (eff$type == "label") {
    row <- fitted_pe_std[fitted_pe_std$op == "~" & fitted_pe_std$label == eff$key, , drop = FALSE]
  } else {
    row <- fitted_pe_std[fitted_pe_std$op == ":=" & fitted_pe_std$lhs == eff$key, , drop = FALSE]
  }
  if (nrow(row) == 1) row$std.all else NA_real_
}, numeric(1))

extract_focal_pvalue <- function(pe_table, eff) {
  if (eff$type == "label") {
    row <- pe_table[pe_table$op == "~" & pe_table$label == eff$key, , drop = FALSE]
  } else {
    row <- pe_table[pe_table$op == ":=" & pe_table$lhs == eff$key, , drop = FALSE]
  }
  if (nrow(row) == 1) row$pvalue else NA_real_
}

# -----------------------------------------------------------------------------
# Core simulation for one target N
# -----------------------------------------------------------------------------
# Failure reasons are recorded and surfaced so a broken run can never again be
# reported as a tidy "0% power".
simulate_power_for_n <- function(target_n, source_data, model_syntax, effects,
                                 reps, alpha_level, sim_se_type, verbose = TRUE) {
  detect_matrix <- matrix(
    NA, nrow = reps, ncol = length(effects),
    dimnames = list(NULL, vapply(effects, function(e) e$key, character(1)))
  )
  converged <- logical(reps)
  admissible <- logical(reps)
  failure_reasons <- character(0)
  
  for (r in seq_len(reps)) {
    sim_data <- tryCatch(
      generate_dataset(target_n, source_data),
      error = function(e) {
        failure_reasons <<- c(failure_reasons, paste("generate:", conditionMessage(e)))
        NULL
      }
    )
    if (is.null(sim_data)) { converged[r] <- FALSE; next }
    
    sim_fit <- tryCatch(
      lavaan::sem(
        model_syntax, data = sim_data,
        estimator = "ML", fixed.x = TRUE, se = sim_se_type, warn = FALSE
      ),
      error = function(e) {
        failure_reasons <<- c(failure_reasons, paste("fit:", conditionMessage(e)))
        NULL
      }
    )
    if (is.null(sim_fit)) { converged[r] <- FALSE; next }
    
    converged[r] <- isTRUE(lavInspect(sim_fit, "converged"))
    admissible[r] <- isTRUE(tryCatch(lavInspect(sim_fit, "post.check"),
                                     error = function(e) FALSE))
    if (!converged[r]) {
      failure_reasons <- c(failure_reasons, "fit: did not converge")
      next
    }
    if (!admissible[r]) {
      failure_reasons <- c(failure_reasons, "fit: inadmissible solution (post.check failed)")
      next
    }
    
    pe <- tryCatch(parameterEstimates(sim_fit), error = function(e) NULL)
    if (is.null(pe)) {
      converged[r] <- FALSE
      failure_reasons <- c(failure_reasons, "parameterEstimates: failed")
      next
    }
    for (j in seq_along(effects)) {
      detect_matrix[r, j] <- isTRUE(extract_focal_pvalue(pe, effects[[j]]) < alpha_level)
    }
  }
  
  usable <- converged & admissible
  n_usable <- sum(usable)
  
  power_est <- vapply(seq_along(effects), function(j) {
    if (n_usable == 0) NA_real_ else mean(detect_matrix[usable, j], na.rm = TRUE)
  }, numeric(1))
  mc_se <- vapply(seq_along(effects), function(j) {
    p <- power_est[j]
    if (is.na(p) || n_usable == 0) NA_real_ else sqrt(p * (1 - p) / n_usable)
  }, numeric(1))
  
  if (verbose) {
    cat(sprintf(
      "  N = %d: %d/%d replications converged & admissible (%.1f%%)\n",
      target_n, n_usable, reps, 100 * n_usable / reps
    ))
    if (n_usable < reps && length(failure_reasons) > 0) {
      reason_tab <- sort(table(failure_reasons), decreasing = TRUE)
      cat("    Failure reasons (top):\n")
      for (k in seq_len(min(3, length(reason_tab)))) {
        cat(sprintf("      [%d x] %s\n", reason_tab[k], names(reason_tab)[k]))
      }
    }
  }
  
  list(
    table = data.frame(
      n = target_n,
      effect_key = vapply(effects, function(e) e$key, character(1)),
      effect_name = vapply(effects, function(e) e$name, character(1)),
      power = power_est,
      mc_se = mc_se,
      n_reps_usable = n_usable,
      n_reps_requested = reps,
      stringsAsFactors = FALSE
    ),
    n_usable = n_usable,
    failure_reasons = failure_reasons
  )
}

# -----------------------------------------------------------------------------
# Run across the N grid
# -----------------------------------------------------------------------------
set.seed(seed)
cat("\nRunning Monte Carlo power simulation (hybrid resampling engine)...\n")
cat("(this can take several minutes depending on POWER_REPS and grid size)\n\n")

sim_runs <- lapply(n_grid, function(target_n) {
  simulate_power_for_n(
    target_n = target_n, source_data = df_model, model_syntax = sem_model,
    effects = focal_effects, reps = n_reps, alpha_level = alpha,
    sim_se_type = sim_se
  )
})

power_results <- dplyr::bind_rows(lapply(sim_runs, function(x) x$table))

# Hard stop if the simulation produced nothing usable anywhere - this is the
# guardrail that was missing before.
total_usable <- sum(vapply(sim_runs, function(x) x$n_usable, integer(1)))
if (total_usable == 0) {
  all_reasons <- unlist(lapply(sim_runs, function(x) x$failure_reasons))
  reason_tab <- sort(table(all_reasons), decreasing = TRUE)
  cat("\n!!! SIMULATION FAILED: 0 usable replications across the entire grid.\n")
  cat("Most common failure reasons:\n")
  for (k in seq_len(min(5, length(reason_tab)))) {
    cat(sprintf("  [%d x] %s\n", reason_tab[k], names(reason_tab)[k]))
  }
  stop("Power analysis aborted - see failure reasons above. ",
       "Do not interpret any 0% / NA values as power estimates.",
       call. = FALSE)
}

effect_size_lookup <- data.frame(
  effect_key = vapply(focal_effects, function(e) e$key, character(1)),
  population_est = focal_population_values,
  population_std = focal_population_std,
  stringsAsFactors = FALSE
)
power_results <- merge(power_results, effect_size_lookup,
                       by = "effect_key", all.x = TRUE, sort = FALSE)

effect_order <- vapply(focal_effects, function(e) e$key, character(1))
power_results$effect_key <- factor(power_results$effect_key, levels = effect_order)
power_results <- power_results[order(power_results$effect_key, power_results$n), ]
power_results$effect_key <- as.character(power_results$effect_key)

# -----------------------------------------------------------------------------
# Report 1: post-hoc power at the analysis N
# -----------------------------------------------------------------------------
posthoc <- power_results[power_results$n == analysis_n, , drop = FALSE]
posthoc_print <- data.frame(
  Effect = posthoc$effect_name,
  `Pop. b` = round(posthoc$population_est, 3),
  `Pop. std` = round(posthoc$population_std, 3),
  Power = round(posthoc$power, 3),
  `MC SE` = round(posthoc$mc_se, 3),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
cat(sprintf(
  "\n=== Post-hoc power at the analysis N (N = %d), alpha = %.3f ===\n",
  analysis_n, alpha
))
cat("Power = P(detect effect | true effect = the fitted estimate)\n\n")
print(posthoc_print, row.names = FALSE)

# -----------------------------------------------------------------------------
# Report 2: sensitivity curve
# -----------------------------------------------------------------------------
sensitivity_wide <- reshape(
  power_results[, c("effect_name", "n", "power")],
  idvar = "effect_name", timevar = "n", direction = "wide"
)
names(sensitivity_wide) <- sub("^power\\.", "N=", names(sensitivity_wide))
sensitivity_wide <- sensitivity_wide[
  match(vapply(focal_effects, function(e) e$name, character(1)),
        sensitivity_wide$effect_name), , drop = FALSE
]
numeric_cols <- vapply(sensitivity_wide, is.numeric, logical(1))
sensitivity_wide[numeric_cols] <- lapply(sensitivity_wide[numeric_cols], round, 3)

cat("\n=== Power sensitivity curve across sample sizes ===\n")
cat("Each cell is estimated power at that N for that effect.\n\n")
print(sensitivity_wide, row.names = FALSE)

# -----------------------------------------------------------------------------
# Report 3: minimum N to reach power thresholds
# -----------------------------------------------------------------------------
min_n_for_threshold <- function(effect_key_value, threshold) {
  rows <- power_results[power_results$effect_key == effect_key_value, , drop = FALSE]
  rows <- rows[order(rows$n), ]
  hit <- rows$n[which(rows$power >= threshold)]
  if (length(hit) == 0) paste0("> ", max(n_grid)) else as.character(min(hit))
}
min_n_table <- data.frame(
  Effect = vapply(focal_effects, function(e) e$name, character(1)),
  `Min N for 80% power` = vapply(focal_effects, function(e) min_n_for_threshold(e$key, 0.80), character(1)),
  `Min N for 90% power` = vapply(focal_effects, function(e) min_n_for_threshold(e$key, 0.90), character(1)),
  check.names = FALSE, stringsAsFactors = FALSE
)
cat("\n=== Minimum sample size to reach power thresholds (grid-based) ===\n")
cat(sprintf("Grid range tested: %d to %d.\n\n", min(n_grid), max(n_grid)))
print(min_n_table, row.names = FALSE)

# -----------------------------------------------------------------------------
# Plain-language summary
# -----------------------------------------------------------------------------
cat("\n=== Summary for the limitations / power section ===\n")
gp <- function(key) {
  v <- posthoc$power[posthoc$effect_key == key]
  if (length(v) == 1 && !is.na(v)) 100 * v else NA_real_
}
fmt <- function(x) if (is.na(x)) "NA (no usable reps)" else sprintf("~%.0f%%", x)
cat(sprintf(
  paste0(
    "At the analysis sample size (N = %d), assuming the population effects equal\n",
    "the fitted estimates:\n",
    "  - Warmth x Personalization interaction: estimated power %s\n",
    "  - Warmth direct effect: estimated power %s\n",
    "  - b1 (Emotional Trust -> outcome): estimated power %s\n",
    "  - Total indirect effect of Warmth: estimated power %s\n",
    "  - Total indirect effect of Personalization: estimated power %s\n"
  ),
  analysis_n,
  fmt(gp("c_wp")), fmt(gp("c_w")), fmt(gp("b1")),
  fmt(gp("ind_w_total")), fmt(gp("ind_p_total"))
))
cat(paste0(
  "Interpretation: low power for the indirect effects is driven primarily by\n",
  "the small fitted a-paths (manipulation -> trust). A null indirect effect at\n",
  "this N should be read as 'not powered to detect a small indirect effect',\n",
  "not as positive evidence of no mediation. Exogenous variables were resampled\n",
  "from the observed data, so the covariate distribution is fixed at the\n",
  "observed one. Significance inside the simulation uses the normal-theory test,\n",
  "a mild lower bound for the indirect effects relative to the percentile\n",
  "bootstrap used in the main analysis.\n"
))

# -----------------------------------------------------------------------------
# Write outputs
# -----------------------------------------------------------------------------
if (write_outputs) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(power_results, file.path(output_dir, "power_results_long.csv"))
  readr::write_csv(sensitivity_wide, file.path(output_dir, "power_sensitivity_wide.csv"))
  readr::write_csv(posthoc_print, file.path(output_dir, "power_posthoc_at_analysis_n.csv"))
  readr::write_csv(min_n_table, file.path(output_dir, "power_min_n_for_thresholds.csv"))
  cat(sprintf("\nWrote power-analysis tables to %s\n", output_dir))
}