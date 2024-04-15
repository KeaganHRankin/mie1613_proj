### Keagan H Rankin
### 4/05/2024

### THIS FILE explores fitting baseline forecast models to the toronto housing data
rm(list = ls())
#Ctrl L to clear console.
dev.off(dev.list()["RStudioGD"]) #Use to clear graphics

# Load required packages
#install.packages("fpp3")
library(fpp3)
library(GGally)
library(gridExtra)

#-----------------------------------------------------------------------------
## CLEANING AND LOADING DATA
# Toronto starts: not a lot of data
#starts_csv <- readr::read_csv("C:/Users/Keagan Rankin/OneDrive - University of Toronto/PhD 1st Year/Winter Courses/stochastic sim/project/data/starts_historic_toronto.csv")
# Canada housing starts: more data, could scale simulation up to national level
# with simplifying assumption
starts_csv <- readr::read_csv("C:/Users/Keagan Rankin/OneDrive - University of Toronto/PhD 1st Year/Winter Courses/stochastic sim/project/data/starts_historic_canada.csv")

# group by year
starts_year <- starts_csv |> 
  group_by(`year`) |> 
  summarise(`All` = sum(`All`),
  ) |> as_tsibble(index=year)

starts_year


#starts_year <- starts_year |> filter_index("1948" ~ "2023")
starts_year <- starts_year |> filter_index(~"2023")

autoplot(starts_year)

# Setting for conf interval in plots
# evaluate models for long term (20 year) forecast
conf_int = 80
forecast_years = 20


## Cross validation
startsy_cv <- starts_year |> stretch_tsibble(.init=3, .step=1)

## Training set
starts_train <- starts_year |> filter_index( ~ '2017')

#-----------------------------------------------------------------------------
## BASELINE MODELS

startsy_cv_models <-  startsy_cv |>
  model(
    Mean = MEAN(`All`),
    `Na�ve` = NAIVE(`All`),
    Drift = RW(`All` ~ drift())
  )


# Which forecast looks best for h step forward forecast.
# look at the performance of naive models 30 years ahead
fc <- startsy_cv_models |>
  forecast(h = 30) |>
  group_by(.id) |>
  mutate(h = row_number()) |>
  ungroup() |>
  as_fable(response = "All", distribution = `All`)

fc |> accuracy(starts_year, by = c("h", ".model")) |>
  ggplot(aes(x = h, y = RMSE, colour=.model)) +
  geom_point() + geom_line()


# Fit and Plot
starts_fit_baseline <- starts_year |> model(`Naive` = NAIVE(`All`))

starts_fc_baseline <- starts_fit_baseline |> forecast(h=forecast_years)


baseline_naive <- starts_fc_baseline |> filter(.model == 'Naive') |> 
  autoplot(size=1.5, level=conf_int, color='cornflowerblue') +
  autolayer(starts_year, color='grey', size=1.5) + 
  ylim(70000,400000) +
  theme_classic(base_size=20) +
  labs(title='Naive Model', y='Housing starts', x='Year') +
  theme(plot.title = element_text(face="bold", size=22),
        legend.position='none')

baseline_naive

## EVALUATING BASELINE MODELS (without a test set)
# Check innovation residual properties: uncorrelated, zero mean
baseline_aug <- starts_fit_baseline |> augment()

# Mean check as per 5.4
starts_fit_baseline |> select(`Naive`) |> gg_tsresiduals()

# ljung test, l = 2m = 2*4 = 8 or l = T/5 which is > 8
baseline_aug |> filter(.model=='Naive') |> features(.innov, ljung_box, lag=8)


# modelling housing as a naive forward walk works okay; we can try and
# see if we can do better with some slightly more complicated models.
#-------------------------------------------------------------------------------
## ARIMA MODEL
# Check out residuals - they are correlated, 1 difference eliminates it
starts_year |> gg_tsdisplay(`All`)
starts_year |> gg_tsdisplay(difference(`All`), plot_type='partial')

# LOG: THIS ENSURES THAT THE FORECAST STAYS POSITIVE.
starts_year_log <- starts_year
starts_year_log['All'] <- log(starts_year_log['All'])
starts_year_log
starts_train_log <- starts_year_log |> filter_index( ~ '2015')

# Training set - train many different ARIMA models to see which may be best
starts_train_arima <- starts_train_log |> 
  model(
    noise = ARIMA(`All` ~ pdq(0,0,0)),
    rand_walk = ARIMA(`All` ~ pdq(0,1,0)),
    exp_smooth = ARIMA(`All` ~ pdq(0,1,1)),
    holt_ets = ARIMA(`All` ~ pdq(0,1,2)),
    ar_1 = ARIMA(`All` ~ pdq(1,0,0)),
    ma_1 = ARIMA(`All` ~ pdq(0,0,1)),
    auto_ar = ARIMA(`All`, stepwise = FALSE, approx = FALSE),
  )

starts_train_arima |> pivot_longer(!1, names_to = "Model name",
                                   values_to = "Orders")


# Choose best model from training: evaluate for long (20 year) forecast
"long forecast results"
starts_train_arima |> forecast(h=forecast_years) |> accuracy(starts_year_log) |> arrange(RMSE)
glance(starts_train_arima) |> arrange(AICc)

# Choose best model from training: evaluate for 10 year ability
"short forecast results"
starts_train_arima |> forecast(h=5) |> accuracy(starts_year) |> arrange(RMSE)
glance(starts_train_arima) |> arrange(AICc)



starts_train_arima |> select(exp_smooth) |> gg_tsresiduals()
arima_res <- augment(starts_train_arima) |> select(.resid)
qqnorm(arima_res$.resid, frame=FALSE)
qqline(arima_res$.resid, col='steelblue', lwd=2)
augment(starts_train_arima) |> features(.innov, ljung_box, lag = 10, dof=2)
# all pass the ljung, quantiles normal but residuals are not

# Train final model
starts_fit_arima <- starts_year |> 
  model(ARIMA(log(`All`)~pdq(0,1,1))) #,stepwise=FALSE, approx=FALSE))

# Forecast final model
starts_fc_arima <- starts_fit_arima |> forecast(h=forecast_years, bootstrap=TRUE)

# Plot final model
arima_plot <-starts_fc_arima |>
  autoplot(size=1.5, level=conf_int, color='darkseagreen') +
  autolayer(starts_year, color='grey', size=1.5) + 
  ylim(75000,400000) +
  theme_classic(base_size=20) +
  labs(title="010 ARIMA Model", y='Housing starts', x='Year') +
  theme(plot.title = element_text(face="bold", size=22),
        legend.position='none')

arima_plot


#-----------------------------------------------------------------------------
## GENERATING SAMPLE PATHS -> usiong statsmodel in python
# We can generate sample paths from the walk model for future housing growth.
path <- "C:/Users/Keagan Rankin/Downloads"
write.csv(starts_year, file.path(path, "my_file.csv"), row.names=FALSE)

