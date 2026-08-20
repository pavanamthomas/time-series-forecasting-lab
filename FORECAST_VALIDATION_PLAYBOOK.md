# Forecast validation playbook

This is the protocol I use for a univariate forecasting problem. It is written in the first person because that is how I would walk it in a technical interview. The laboratory implements the protocol on **simulated** series; the same sequence applies when the series is observational, with the additional burden of saying what in the world the DGP is approximating.

I do not treat a more heavily parameterised model as superior by default. I always keep a naive benchmark. I never report in-sample fit as if it were out-of-sample forecast performance.

---

## I inspect the series

I plot the path, then I look at rolling means and rolling standard deviations. I ask whether the dominant feature is a trend, a repeating seasonal shape, a level shift, mean reversion, or clustered volatility.

I compute a short numerical summary (location, scale, quantiles, missingness). I do not select a model from a gallery of plots alone. Inspection generates hypotheses about the second-moment structure and about breaks. Those hypotheses have to survive a withheld window.

If the series is simulated, I read the DGP before I pretend to discover it. In this repository the DGP is documented in `src/tsforecast/dgp.py`. If the series were observational, I would write the analogue: sampling frequency, revisions, calendar effects, and known policy dates.

## I define the forecast horizon

I state h, the decision that depends on ŷ_{t+h|t}, and the loss. A one-step RMSE ranking is not a licence to claim skill at h = 12.

I also state the origin scheme. Expanding origin uses all data up to t. Rolling origin uses a fixed window ending at t. Both are causal only if the window's last index is t.

I refuse a design that estimates on 1…T and then scores fitted values on 1…T under the name “forecast evaluation.”

## I establish a naive benchmark

Before ARIMA, Holt–Winters, or GARCH, I compute:

- naive (last value),
- seasonal naive when the decision is seasonal,
- mean (expanding or rolling, matching the origin scheme).

If a named model cannot beat the relevant naive forecast on the same origins and the same h, I do not advertise it as a forecasting improvement. It may still be a useful description of the in-sample dynamics.

## I identify transformations

I ask whether the working series should be levels, logs, or differences. Logarithms require a strictly positive series. First differences are a modelling choice, not a ritual.

I run an ADF regression whose deterministic terms match the alternative I care about. A trending series tested with a constant and no trend is a misspecified ADF. I treat p-values as diagnostics. Failure to reject a unit root is not a proof that the series “is I(1).”

I inspect ACF and PACF on the working series (and on squares if volatility is the object). These plots inform candidate orders. They do not close the validation question.

## I estimate candidate models

I fit on the information set of the origin, not on the full sample.

Candidates in this laboratory:

- OLS linear trend, when the DGP or the inspection suggests a drift in mean;
- ARIMA(p,d,q) and SARIMA(p,d,q)(P,D,Q)s;
- Holt–Winters (additive trend/seasonal as specified);
- Gaussian GARCH(1,1) QMLE when the object is conditional variance.

I record the exact order, the trend specification, and the optimiser status. If estimation fails to converge, I do not silently fall back to a different sample.

## I inspect residuals

On the estimation window I look at residual plots, residual ACF, and a Ljung–Box statistic with degrees of freedom that respect the estimated ARMA orders.

A quiet Ljung–Box is consistent with uncorrelated residuals under the test's assumptions. It is not evidence that the forecast will beat naive out of sample. A loud Ljung–Box is a specification warning, especially if I am about to quote analytic Gaussian intervals.

For GARCH, the analogue is the ACF of squared standardised residuals, which I treat with the same caution.

## I perform rolling evaluation

I move the origin forward. At each origin t I:

1. slice y_s for s ≤ t (expanding) or for t−w+1 ≤ s ≤ t (rolling);
2. fit a **new** instance of the model on that slice;
3. forecast h steps;
4. score against y_{t+1}, …, y_{t+h}.

I never reuse a model fitted on a later origin. I never standardise, impute, or encode using statistics computed after t.

The tests in `tests/test_forecast_no_leakage.py` exist because this step is where leakage is introduced. If I plant a sentinel in the future and the fitted mean moves, the pipeline is wrong.

## I compare errors

I report RMSE and MAE on the held-out windows. I report MAPE only when every realised value is bounded away from zero. I report MASE against the matching in-sample naive or seasonal naive scale.

I compare models on the **same** origins. Averaging RMSE across incompatible windows is not a ranking.

I look at the distribution of origin-wise errors, not only the mean. A method that wins on average but collapses after a break is not stable.

## I assess stability

I ask whether coefficients, residual variance, and rolling errors are stable across origins.

For a suspected level shift I do not “test for a break” by peeking at the whole sample and then pretending the date was known—unless, as in this laboratory, the date **is** known because it is part of the DGP. Given a known T_b I split the sample and I compute a Chow statistic for a linear specification. I also forecast the post-break window using only pre-break information, which is the honest pre-break predictor.

If the DGP did not give me T_b, I treat break search as a multiple-testing problem. `sup_chow` maximises a Chow statistic on a trimmed grid inside the estimation window. A pointwise p < 0.05 anywhere on that grid is not the sup test. `sup_chow_null_critical_value` is a Monte Carlo quantile under iid N(0,1) and no break, for the n and trim used in the call; it is not an Andrews (1993) table.

## I report forecast uncertainty

I publish the interval rule next to the interval:

- ARIMA: statsmodels' analytic Gaussian prediction intervals;
- naive: random-walk scaling of the in-sample difference standard deviation;
- mean: iid Gaussian predictive interval;
- linear trend: OLS predictive interval under homoskedasticity;
- Holt–Winters: Monte Carlo intervals from residual simulation.

Coverage is a property of the assumed model, not a nonparametric guarantee. I do not thicken bands after seeing the test window.

## I state limitations

I close with what the exercise cannot support:

- synthetic DGPs do not validate a production forecasting system;
- Gaussian intervals fail under heavy tails and unmodelled breaks;
- beating naive on one simulated seasonal series does not generalise to other seasonal shapes;
- GARCH(1,1) QMLE is not a complete volatility model;
- computational convergence of MLE is not identification of the DGP.

If I had to make a decision with the forecast, I would state the decision, the loss, and the cost of being wrong in the tail, not only a point RMSE.

---

The implementation of this protocol is `src/tsforecast/validation.py` for causal slicing, `scripts/run_all.py` for a full laboratory run, and `notebooks/01_forecast_workflow.ipynb` for a worked path through one simulated series.
