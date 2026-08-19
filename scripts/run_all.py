"""Generate simulated series, figures, and tables.

Random seed: 42 (documented laboratory seed).
Matplotlib backend: Agg (non-interactive).

Numeric summaries are written under outputs/. They are regenerated, not
hand-edited, and they are not scientific findings about observational data.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsforecast.arima_models import (  # noqa: E402
    ARIMAForecaster,
    ar_lag1_coefficient,
    fit_arima,
    residual_diagnostics,
)
from tsforecast.dgp import DEFAULT_SEED, generate_catalog  # noqa: E402
from tsforecast.eda import series_summary, transform_report  # noqa: E402
from tsforecast.plots import (  # noqa: E402
    plot_acf_pacf,
    plot_break_split,
    plot_forecast,
    plot_garch,
    plot_misleading_trend_arima,
    plot_residuals,
    plot_rolling_rmse,
    plot_series_gallery,
)
from tsforecast.smoothing import HoltWintersForecaster  # noqa: E402
from tsforecast.stationarity import adf_to_row, adf_unit_root  # noqa: E402
from tsforecast.validation import (  # noqa: E402
    LinearTrendForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    chow_test,
    evaluate_forecaster,
    forecast_from_origin,
)
from tsforecast.volatility import fit_garch11  # noqa: E402

SEED = DEFAULT_SEED
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
DATA = ROOT / "data" / "simulated"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def _write_series(catalog: dict) -> None:
    for name, item in catalog.items():
        item.values.to_csv(DATA / f"{name}.csv", header=True)
    meta_rows = []
    for name, item in catalog.items():
        scalars = {
            k: v
            for k, v in item.parameters.items()
            if np.isscalar(v) or isinstance(v, (str, int, float, bool))
        }
        meta_rows.append(
            {
                "name": name,
                "simulated": True,
                "dgp": item.dgp,
                "notes": item.notes,
                **scalars,
            }
        )
    pd.DataFrame(meta_rows).to_csv(TAB / "dgp_metadata.csv", index=False)


def _summaries(catalog: dict) -> pd.DataFrame:
    frames = []
    for name, item in catalog.items():
        s = series_summary(item.values)
        row = s.to_frame().T
        row.insert(0, "series", name)
        row.insert(1, "simulated", True)
        tr = transform_report(item.values, period=12 if name != "simulated_volatility_clustering" else 5)
        for k, v in tr.items():
            row[k] = v
        frames.append(row)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(TAB / "series_summaries.csv", index=False)
    return out


def _adf_table(catalog: dict) -> pd.DataFrame:
    rows = []
    spec = {
        "simulated_trend_only": "ct",
        "simulated_seasonal": "c",
        "simulated_stationary_ar": "c",
        "simulated_structural_break": "c",
        "simulated_volatility_clustering": "n",
    }
    for name, item in catalog.items():
        result = adf_unit_root(item.values, regression=spec[name])
        rows.append(adf_to_row(name, result))
    # Misspecified ADF on the trend series: constant only, no trend term.
    miss = adf_unit_root(catalog["simulated_trend_only"].values, regression="c")
    row = adf_to_row("simulated_trend_only_adf_constant_only", miss)
    row["regression_note"] = "misspecified_for_trending_DGP"
    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "adf_diagnostics.csv", index=False)
    return out


def _acf_plots(catalog: dict) -> None:
    for name, item in catalog.items():
        nlags = 24 if name != "simulated_volatility_clustering" else 20
        plot_acf_pacf(
            item.values,
            FIG / f"acf_pacf_{name}.png",
            nlags=nlags,
            title=f"ACF/PACF: {name} (simulated)",
        )
        if name == "simulated_volatility_clustering":
            sq = item.values**2
            sq.name = f"{name}_squared"
            plot_acf_pacf(
                sq,
                FIG / "acf_pacf_simulated_squared_returns.png",
                nlags=20,
                title="ACF/PACF of squared simulated GARCH returns",
            )


def _one_origin_forecasts(catalog: dict) -> None:
    y = catalog["simulated_seasonal"].values
    origin = len(y) - 13
    fc, train, test = forecast_from_origin(
        y, SeasonalNaiveForecaster(period=12), origin, horizon=12
    )
    plot_forecast(
        train.iloc[-48:],
        test,
        fc,
        FIG / "forecast_simulated_seasonal_seasonal_naive.png",
        title="Seasonal naive, simulated seasonal series, last origin",
    )
    hw = HoltWintersForecaster(trend=None, seasonal="add", seasonal_periods=12)
    fc_hw, train_hw, test_hw = forecast_from_origin(y, hw, origin, horizon=12)
    plot_forecast(
        train_hw.iloc[-48:],
        test_hw,
        fc_hw,
        FIG / "forecast_simulated_seasonal_holt_winters.png",
        title="Holt-Winters additive seasonal, simulated series, last origin",
    )

    ar = catalog["simulated_stationary_ar"].values
    origin_ar = len(ar) - 13
    fc_ar, tr_ar, te_ar = forecast_from_origin(
        ar, ARIMAForecaster(order=(1, 0, 0)), origin_ar, horizon=12
    )
    plot_forecast(
        tr_ar.iloc[-80:],
        te_ar,
        fc_ar,
        FIG / "forecast_simulated_stationary_ar_arima100.png",
        title="ARIMA(1,0,0) on simulated AR(1), last origin",
    )
    diag = residual_diagnostics(fit_arima(tr_ar, order=(1, 0, 0)), lags=12)
    diag.to_csv(TAB / "ljung_box_residuals.csv", index=False)
    plot_residuals(
        fit_arima(tr_ar, order=(1, 0, 0)).result.resid,
        FIG / "residual_diagnostics_simulated_stationary_ar.png",
        title="In-sample residuals: ARIMA(1,0,0) on simulated AR(1)",
    )


def _rolling_tables(catalog: dict) -> pd.DataFrame:
    blocks = []

    y_ar = catalog["simulated_stationary_ar"].values
    for factory in (
        NaiveForecaster,
        MeanForecaster,
        lambda: ARIMAForecaster(order=(1, 0, 0)),
    ):
        tbl = evaluate_forecaster(
            y_ar,
            factory,
            min_train=120,
            horizon=6,
            step=24,
            scheme="expanding",
            seasonality=1,
        )
        tbl.insert(0, "series", "simulated_stationary_ar")
        blocks.append(tbl)

    y_s = catalog["simulated_seasonal"].values
    for factory in (
        NaiveForecaster,
        lambda: SeasonalNaiveForecaster(period=12),
        MeanForecaster,
        lambda: HoltWintersForecaster(trend=None, seasonal="add", seasonal_periods=12),
        lambda: ARIMAForecaster(order=(1, 0, 0)),
    ):
        tbl = evaluate_forecaster(
            y_s,
            factory,
            min_train=72,
            horizon=12,
            step=24,
            scheme="expanding",
            seasonality=12,
        )
        tbl.insert(0, "series", "simulated_seasonal")
        blocks.append(tbl)

    y_t = catalog["simulated_trend_only"].values
    for factory in (
        NaiveForecaster,
        MeanForecaster,
        LinearTrendForecaster,
        lambda: ARIMAForecaster(order=(1, 0, 0), name="ARIMA(1,0,0)_no_difference"),
        lambda: ARIMAForecaster(order=(0, 1, 1), name="ARIMA(0,1,1)_differenced"),
    ):
        tbl = evaluate_forecaster(
            y_t,
            factory,
            min_train=80,
            horizon=12,
            step=24,
            scheme="expanding",
            seasonality=1,
        )
        tbl.insert(0, "series", "simulated_trend_only")
        blocks.append(tbl)

    out = pd.concat(blocks, ignore_index=True)
    out.to_csv(TAB / "rolling_forecast_errors.csv", index=False)
    summary = (
        out.groupby(["series", "model"], as_index=False)[["rmse", "mae", "mase"]]
        .mean()
        .rename(columns={"rmse": "mean_rmse", "mae": "mean_mae", "mase": "mean_mase"})
    )
    summary.to_csv(TAB / "rolling_error_means.csv", index=False)

    trend_roll = out[out["series"] == "simulated_trend_only"]
    if not trend_roll.empty:
        plot_rolling_rmse(
            trend_roll,
            FIG / "rolling_rmse_simulated_trend_only.png",
            title="Expanding-origin RMSE on simulated trend-only series",
        )
    seas_roll = out[out["series"] == "simulated_seasonal"]
    if not seas_roll.empty:
        plot_rolling_rmse(
            seas_roll,
            FIG / "rolling_rmse_simulated_seasonal.png",
            title="Expanding-origin RMSE on simulated seasonal series",
        )
    return out


def _misspecification_trend(catalog: dict) -> None:
    y = catalog["simulated_trend_only"].values
    origin = len(y) - 25
    fit_bad = fit_arima(y.iloc[: origin + 1], order=(2, 0, 2))
    fc_bad, train, test = forecast_from_origin(
        y,
        ARIMAForecaster(order=(2, 0, 2), name="ARIMA(2,0,2)_no_difference"),
        origin,
        horizon=24,
    )
    fitted = np.asarray(fit_bad.result.fittedvalues, dtype=float)
    plot_misleading_trend_arima(
        y, fitted, fc_bad, origin, FIG / "misleading_arima_on_trend.png"
    )
    fc_tr, _, _ = forecast_from_origin(y, LinearTrendForecaster(), origin, horizon=24)
    fc_nv, _, _ = forecast_from_origin(y, NaiveForecaster(), origin, horizon=24)
    fc_d, _, _ = forecast_from_origin(
        y, ARIMAForecaster(order=(0, 1, 1), name="ARIMA(0,1,1)"), origin, horizon=24
    )
    from tsforecast.metrics import error_summary

    rows = []
    for fc in (fc_bad, fc_tr, fc_nv, fc_d):
        rows.append(
            {
                "series": "simulated_trend_only",
                "origin": origin,
                "model": fc.name,
                "ar_lag1": (
                    ar_lag1_coefficient(fit_bad) if "2, 0, 2" in fc.name else np.nan
                ),
                **error_summary(test.to_numpy(), fc.point, train.to_numpy()),
                "note": (
                    "Undifferenced ARIMA on a linear trend can fit in sample "
                    "while losing to a correctly specified trend extrapolation."
                ),
            }
        )
    pd.DataFrame(rows).to_csv(TAB / "misspecification_trend.csv", index=False)
    plot_forecast(
        train.iloc[-60:],
        test,
        fc_tr,
        FIG / "forecast_simulated_trend_linear.png",
        title="Linear trend extrapolation on simulated trend-only series",
    )


def _misspecification_break(catalog: dict) -> None:
    item = catalog["simulated_structural_break"]
    y = item.values
    break_index = int(item.parameters["break_index"])
    plot_break_split(
        y,
        break_index,
        FIG / "break_split_illustration.png",
        title="Known mean shift in the simulated structural-break DGP",
    )
    chow_level = chow_test(y, break_index, trend=False)
    chow_tr = chow_test(y, break_index, trend=True)
    chow_df = pd.DataFrame(
        [
            {
                "series": "simulated_structural_break",
                "specification": r.specification,
                "break_index": r.break_index,
                "f_stat": r.f_stat,
                "p_value": r.p_value,
                "rss_pooled": r.rss_pooled,
                "rss_split": r.rss_split,
                "df_num": r.df_num,
                "df_den": r.df_den,
                "note": r.note,
            }
            for r in (chow_level, chow_tr)
        ]
    )
    chow_df.to_csv(TAB / "chow_known_break.csv", index=False)

    # Forecast the post-break window from a pre-break origin, using only pre-break data.
    origin = break_index - 1
    horizon = min(24, len(y) - break_index)
    fc_pre, train_pre, test_post = forecast_from_origin(
        y, MeanForecaster(), origin, horizon=horizon
    )
    # Causal analogue of "ignore the break": expanding mean that has not yet seen the shift
    # is already fc_pre. The full-sample mean mixes regimes and uses post-break data.
    from tsforecast.metrics import error_summary

    post = y.iloc[break_index : break_index + horizon]
    ignore_break_point = np.repeat(float(y.mean()), horizon)
    rows = [
        {
            "model": "mean_trained_only_before_break",
            **error_summary(test_post.to_numpy(), fc_pre.point, train_pre.to_numpy()),
            "note": "Causal: origin is the last pre-break observation.",
        },
        {
            "model": "oracle_post_break_mean",
            **error_summary(
                post.to_numpy(),
                np.repeat(float(post.mean()), horizon),
                y.iloc[:break_index].to_numpy(),
            ),
            "note": "In-sample description of the post-break regime; not a forecast.",
        },
        {
            "model": "full_sample_mean_uses_future_regime",
            **error_summary(
                post.to_numpy(),
                ignore_break_point,
                y.to_numpy(),
            ),
            "note": (
                "This calculation uses the whole series, including the post-break "
                "window itself. It is shown only to illustrate leakage and regime mixing."
            ),
        },
    ]
    pd.DataFrame(rows).to_csv(TAB / "misspecification_break.csv", index=False)
    plot_forecast(
        train_pre.iloc[-40:],
        test_post,
        fc_pre,
        FIG / "forecast_simulated_break_prebreak_mean.png",
        title="Pre-break mean forecasting the post-break simulated regime",
    )


def _garch_section(catalog: dict) -> None:
    item = catalog["simulated_volatility_clustering"]
    r = item.values
    true_var = item.parameters["conditional_variance"].to_numpy()
    est = fit_garch11(r.to_numpy())
    pd.DataFrame(
        [
            {
                "series": "simulated_volatility_clustering",
                "omega_hat": est.omega,
                "alpha_hat": est.alpha,
                "beta_hat": est.beta,
                "persistence_hat": est.persistence,
                "nll": est.nll,
                "success": est.success,
                "n_obs": est.n_obs,
                "omega_dgp": item.parameters["omega"],
                "alpha_dgp": item.parameters["alpha"],
                "beta_dgp": item.parameters["beta"],
                "assumption": (
                    "Gaussian GARCH(1,1) quasi-likelihood, zero conditional mean, "
                    "covariance-stationary constraint alpha+beta<1."
                ),
            }
        ]
    ).to_csv(TAB / "garch11_estimates.csv", index=False)
    plot_garch(
        r,
        est.sigma2,
        FIG / "garch_filtered_variance.png",
        title="Simulated GARCH(1,1) returns and QMLE filtered volatility",
    )
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(r.index, np.sqrt(true_var), color="#1f4e79", linewidth=1.0, label="DGP sigma_t")
    ax.plot(r.index, np.sqrt(est.sigma2), color="#c45911", linewidth=0.9, label="filtered sigma_t")
    ax.set_title("Simulated GARCH(1,1): DGP volatility versus filtered volatility")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "garch_dgp_versus_filtered.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    _ensure_dirs()
    catalog = generate_catalog(seed=SEED)
    plot_series_gallery(
        {k: v.values for k, v in catalog.items()},
        FIG / "series_gallery.png",
    )
    _write_series(catalog)
    _summaries(catalog)
    _adf_table(catalog)
    _acf_plots(catalog)
    _one_origin_forecasts(catalog)
    _rolling_tables(catalog)
    _misspecification_trend(catalog)
    _misspecification_break(catalog)
    _garch_section(catalog)
    print(f"Seed={SEED}. Wrote figures under {FIG} and tables under {TAB}.")
    print("All series are simulated. Tables are regenerated artifacts, not observational findings.")


if __name__ == "__main__":
    main()
