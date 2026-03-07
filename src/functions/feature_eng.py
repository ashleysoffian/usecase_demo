from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

# Color scheme
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Accent purple
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'box_fill': '#E8F4F8',     # Light blue for boxes
    'hist_fill': '#2E86AB',    # Blue for histograms
}

class Feature_Engineering:
    """Feature engineering utilities for DataFrames."""

    ImputeMethod = Literal["median", "mean", "mode", "knn"]

    @staticmethod
    def _resolve_columns(
        df: pd.DataFrame,
        columns: str | Iterable[str] | None,
        *,
        default: Literal["numeric", "all"] = "numeric",
    ) -> list[str]:
        if columns is None:
            target_cols = list(df.columns) if default == "all" else [
                c for c in df.columns if is_numeric_dtype(df[c])
            ]
        elif isinstance(columns, str):
            target_cols = [columns]
        else:
            target_cols = list(columns)

        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"columns not found in DataFrame: {missing_cols}")

        return target_cols

    @staticmethod
    def impute_missing_values(
        df: pd.DataFrame,
        columns: str | Iterable[str] | None = None,
        method: ImputeMethod = "median",
        *,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Impute missing values in a DataFrame.

        Args:
            df: Input DataFrame.
            columns: Column name or iterable of column names to impute.
                - If None: uses all numeric columns.
            method: Imputation method.
                - "median" (default): fill missing values with the column median (numeric only).
                - "mean": fill missing values with the column mean (numeric only).
                - "mode": fill missing values with the most frequent value (works for numeric or non-numeric).
                - "knn": use sklearn's KNNImputer (numeric columns only).
            n_neighbors: Number of neighbors to use for KNN imputation.
            weights: Weight function for KNN imputation.
            inplace: If True, modify `df` in place and return it.

        Returns:
            DataFrame with imputed values.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: For invalid columns or incompatible dtypes/method.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        normalized_method = str(method).lower().strip()

        # Default to numeric columns, except for mode where categorical often makes sense too.
        target_cols = Feature_Engineering._resolve_columns(
            df,
            columns,
            default="all" if normalized_method == "mode" else "numeric",
        )

        if not target_cols:
            # Nothing to do
            return df if inplace else df.copy()

        out = df if inplace else df.copy()

        method = normalized_method
        if method == "median":
            # Median only makes sense for numeric data.
            non_numeric = [c for c in target_cols if not is_numeric_dtype(out[c])]
            if non_numeric:
                raise ValueError(
                    f"median imputation requires numeric columns; non-numeric: {non_numeric}"
                )

            medians = out[target_cols].median(numeric_only=True)
            out[target_cols] = out[target_cols].fillna(medians)
            return out

        if method == "mean":
            non_numeric = [c for c in target_cols if not is_numeric_dtype(out[c])]
            if non_numeric:
                raise ValueError(
                    f"mean imputation requires numeric columns; non-numeric: {non_numeric}"
                )

            means = out[target_cols].mean(numeric_only=True)
            out[target_cols] = out[target_cols].fillna(means)
            return out

        if method == "mode":
            # Mode works for most dtypes (including object/category/bool/numeric).
            for c in target_cols:
                if out[c].isna().any():
                    modes = out[c].mode(dropna=True)
                    if len(modes) == 0:
                        # Column is entirely missing; nothing sensible to impute.
                        continue
                    out[c] = out[c].fillna(modes.iloc[0])
            return out

        if method == "knn":
            non_numeric = [c for c in target_cols if not is_numeric_dtype(out[c])]
            if non_numeric:
                raise ValueError(
                    f"KNN imputation requires numeric columns; non-numeric: {non_numeric}"
                )

            from sklearn.impute import KNNImputer

            # KNNImputer works on a numeric matrix; keep column alignment.
            original_dtypes = {c: out[c].dtype for c in target_cols}

            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imputed = imputer.fit_transform(out[target_cols])
            imputed_df = pd.DataFrame(imputed, columns=target_cols, index=out.index)

            # Best-effort dtype restoration for integer-like columns.
            for c in target_cols:
                if is_integer_dtype(original_dtypes[c]):
                    # If values are very close to integers, cast back.
                    vals = imputed_df[c].to_numpy(dtype=float)
                    if np.all(np.isfinite(vals)) and np.all(np.isclose(vals, np.round(vals))):
                        imputed_df[c] = np.round(imputed_df[c]).astype("Int64")

            out[target_cols] = imputed_df[target_cols]
            return out

        raise ValueError(
            f"Unknown imputation method: {method!r}. Use 'median', 'mean', 'mode', or 'knn'."
        )

    @staticmethod
    def log_transform_skewed_features(
        df: pd.DataFrame,
        columns: str | Iterable[str] | None = None,
        *,
        skew_threshold: float = 1.0,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Apply a log transform to columns with high absolute skewness.

        A column is transformed when `abs(skew) >= skew_threshold`.

        Notes:
            - Uses `np.log1p()` to safely handle zeros.
            - If a selected column has negative values, it is shifted so the minimum is 0
            before applying `log1p`.

        Args:
            df: Input DataFrame.
            columns: Column name or iterable of column names to consider.
                - If None: uses all numeric columns.
            skew_threshold: Minimum absolute skewness to trigger transformation.
            inplace: If True, modify `df` in place and return it.

        Returns:
            DataFrame with log-transformed skewed columns.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If columns are missing or non-numeric.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        target_cols = Feature_Engineering._resolve_columns(df, columns, default="numeric")

        non_numeric = [c for c in target_cols if not is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(
                f"log transform requires numeric columns; non-numeric: {non_numeric}"
            )

        out = df if inplace else df.copy()

        for c in target_cols:
            col_data = out[c].dropna()
            if col_data.empty:
                continue

            skew_val = col_data.skew()
            if not np.isfinite(skew_val):
                continue

            if abs(float(skew_val)) >= float(skew_threshold):
                min_val = float(col_data.min())
                shift = -min_val if min_val < 0 else 0.0
                out[c] = np.log1p(out[c] + shift)

        return out

    @staticmethod
    def plot_distribution_before_after_log_transform(
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        columns: str | Iterable[str],
        *,
        bins: int = 30,
        combined: bool = False,
    ):
        """Plot distributions before/after transformation.

        This is styled to match the histogram + KDE approach used in `EDA_Analysis.hist_plot()`.

        Args:
            before_df: DataFrame before transformation.
            after_df: DataFrame after transformation.
            columns: Column name or iterable of columns to plot (must exist in both DataFrames).
            bins: Histogram bins.
            combined: If True and multiple columns are provided, display all columns in one
                figure (one row per column, two panels: before/after). If False, returns a
                single figure for a single column or a list of figures for multiple columns.

        Returns:
            Matplotlib Figure (single plot) or list of Figures (multiple, non-combined).
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if not isinstance(before_df, pd.DataFrame) or not isinstance(after_df, pd.DataFrame):
            raise TypeError("before_df and after_df must be pandas DataFrames")

        if isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)

        missing_before = [c for c in cols if c not in before_df.columns]
        if missing_before:
            raise ValueError(f"columns not found in before_df: {missing_before}")
        missing_after = [c for c in cols if c not in after_df.columns]
        if missing_after:
            raise ValueError(f"columns not found in after_df: {missing_after}")

        non_numeric = [c for c in cols if not is_numeric_dtype(before_df[c]) or not is_numeric_dtype(after_df[c])]
        if non_numeric:
            raise ValueError(f"distribution plots require numeric columns; non-numeric: {non_numeric}")

        plt.style.use("seaborn-v0_8-darkgrid")

        def _set_plot_style(ax, title: str):
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.tick_params(labelsize=9)

        def _plot_one(ax, series: pd.Series, title: str):
            col_data = series.dropna()
            if col_data.empty:
                ax.set_visible(False)
                return

            mean_val = col_data.mean()
            median_val = col_data.median()

            ax.hist(
                col_data,
                bins=bins,
                color=COLORS.get("hist_fill", "#2E86AB"),
                edgecolor="white",
                alpha=0.8,
                linewidth=0.4,
            )

            ax2 = ax.twinx()
            col_data.plot(
                kind="kde",
                ax=ax2,
                color=COLORS.get("danger", "#C73E1D"),
                linewidth=1.2,
                alpha=0.7,
            )
            ax2.set_ylabel("")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.set_yticks([])

            ax.axvline(
                mean_val,
                color=COLORS.get("danger", "#C73E1D"),
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Mean: {mean_val:.3g}",
            )
            ax.axvline(
                median_val,
                color=COLORS.get("success", "#06A77D"),
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Median: {median_val:.3g}",
            )

            kde_line = Line2D(
                [0],
                [0],
                color=COLORS.get("danger", "#C73E1D"),
                linewidth=1.2,
                alpha=0.7,
                label="KDE",
            )
            handles, _ = ax.get_legend_handles_labels()
            handles.append(kde_line)

            ax.text(
                0.98,
                0.98,
                f"Skew: {col_data.skew():.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            ax.legend(handles=handles, loc="center right", fontsize=8, framealpha=0.9)
            ax.set_ylabel("Frequency", fontsize=10, fontweight="medium")
            ax.set_xlabel("")
            _set_plot_style(ax, title)

        # Combined grid: one row per feature, two columns (before/after).
        if combined and len(cols) > 1:
            fig_h = max(2.8, 2.6 * len(cols))
            fig, axes = plt.subplots(len(cols), 2, figsize=(12, fig_h))
            axes = np.atleast_2d(axes)

            for i, c in enumerate(cols):
                _plot_one(axes[i, 0], before_df[c], f"Before: {c}")
                _plot_one(axes[i, 1], after_df[c], f"After: {c}")

            fig.suptitle(
                "Distribution Before/After Transformation",
                fontsize=14,
                fontweight="bold",
                y=1.00,
            )
            fig.tight_layout()
            return fig

        # Single column (or non-combined multiple): return one fig per column.
        if len(cols) == 1:
            c = cols[0]
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            _plot_one(axes[0], before_df[c], f"Before: {c}")
            _plot_one(axes[1], after_df[c], f"After: {c}")
            fig.suptitle("Distribution Before/After Transformation", fontsize=14, fontweight="bold", y=1.02)
            fig.tight_layout()
            return fig

        figs = []
        for c in cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            _plot_one(axes[0], before_df[c], f"Before: {c}")
            _plot_one(axes[1], after_df[c], f"After: {c}")
            fig.suptitle("Distribution Before/After Transformation", fontsize=14, fontweight="bold", y=1.02)
            fig.tight_layout()
            figs.append(fig)
        return figs

    @staticmethod
    def plot_boxplot_before_after_transform(
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        columns: str | Iterable[str],
        *,
        combined: bool = False,
    ):
        """Plot boxplots before/after transformation.

        Styled to match the general look of `EDA_Analysis.box_plot()`.

        Args:
            before_df: DataFrame before transformation.
            after_df: DataFrame after transformation.
            columns: Column name or iterable of columns to plot (must exist in both DataFrames).
            combined: If True and multiple columns are provided, display all columns in one
                figure (one row per column, two panels: before/after). If False, returns a
                single figure for a single column or a list of figures for multiple columns.

        Returns:
            Matplotlib Figure (single plot) or list of Figures (multiple, non-combined).
        """
        import matplotlib.pyplot as plt

        if not isinstance(before_df, pd.DataFrame) or not isinstance(after_df, pd.DataFrame):
            raise TypeError("before_df and after_df must be pandas DataFrames")

        if isinstance(columns, str):
            cols = [columns]
        else:
            cols = list(columns)

        missing_before = [c for c in cols if c not in before_df.columns]
        if missing_before:
            raise ValueError(f"columns not found in before_df: {missing_before}")
        missing_after = [c for c in cols if c not in after_df.columns]
        if missing_after:
            raise ValueError(f"columns not found in after_df: {missing_after}")

        non_numeric = [c for c in cols if not is_numeric_dtype(before_df[c]) or not is_numeric_dtype(after_df[c])]
        if non_numeric:
            raise ValueError(f"boxplots require numeric columns; non-numeric: {non_numeric}")

        plt.style.use("seaborn-v0_8-darkgrid")

        def _outlier_count(series: pd.Series) -> int:
            s = series.dropna()
            if s.empty:
                return 0
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                return 0
            return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        def _boxplot(ax, series: pd.Series, title: str, *, compact: bool):
            s = series.dropna()
            if s.empty:
                ax.set_visible(False)
                return

            ax.boxplot(
                s,
                vert=False,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": COLORS["danger"],
                    "markeredgecolor": COLORS["danger"],
                    "markersize": 4 if compact else 7,
                },
                medianprops={
                    "color": COLORS["success"],
                    "linewidth": 1.5 if compact else 2,
                },
                boxprops={
                    "facecolor": COLORS["box_fill"],
                    "edgecolor": COLORS["primary"],
                    "linewidth": 1.0 if compact else 1.5,
                },
                whiskerprops={
                    "color": COLORS["primary"],
                    "linewidth": 1.0 if compact else 1.5,
                    "linestyle": "--",
                },
                capprops={
                    "color": COLORS["primary"],
                    "linewidth": 1.0 if compact else 1.5,
                },
                flierprops={
                    "marker": "o",
                    "markerfacecolor": COLORS["warning"],
                    "markeredgecolor": COLORS["warning"],
                    "markersize": 2.5 if compact else 5,
                    "alpha": 0.6,
                },
            )

            mean_val = float(s.mean())
            median_val = float(s.median())
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            outliers = _outlier_count(s)

            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.tick_params(labelsize=9)
            ax.set_yticks([])

            stats_text = (
                f"μ={mean_val:.2g} | med={median_val:.2g} | IQR={iqr:.2g} | Out={outliers}"
                if compact
                else f"Mean: {mean_val:.3g} | Median: {median_val:.3g} | IQR: {iqr:.3g} | Outliers: {outliers}"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8 if compact else 9,
                style="italic",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3),
            )

        # Combined grid: one row per feature, two columns (before/after).
        if combined and len(cols) > 1:
            fig_h = max(2.8, 2.2 * len(cols))
            fig, axes = plt.subplots(len(cols), 2, figsize=(12, fig_h))
            axes = np.atleast_2d(axes)

            for i, c in enumerate(cols):
                _boxplot(axes[i, 0], before_df[c], f"Before: {c}", compact=True)
                _boxplot(axes[i, 1], after_df[c], f"After: {c}", compact=True)

            fig.suptitle("Boxplot Before/After Transformation", fontsize=14, fontweight="bold", y=1.00)
            fig.tight_layout()
            return fig

        # Single column (or non-combined multiple): return one fig per column.
        if len(cols) == 1:
            c = cols[0]
            fig, axes = plt.subplots(1, 2, figsize=(12, 2.8))
            _boxplot(axes[0], before_df[c], f"Before: {c}", compact=False)
            _boxplot(axes[1], after_df[c], f"After: {c}", compact=False)
            fig.suptitle("Boxplot Before/After Transformation", fontsize=14, fontweight="bold", y=1.02)
            fig.tight_layout()
            return fig

        figs = []
        for c in cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 2.8))
            _boxplot(axes[0], before_df[c], f"Before: {c}", compact=False)
            _boxplot(axes[1], after_df[c], f"After: {c}", compact=False)
            fig.suptitle("Boxplot Before/After Transformation", fontsize=14, fontweight="bold", y=1.02)
            fig.tight_layout()
            figs.append(fig)
        return figs

    EncodingType = Literal["onehot", "ordinal"]

    @staticmethod
    def encode_categorical_features(
        df: pd.DataFrame,
        columns: str | Iterable[str] | None = None,
        encoding: EncodingType = "onehot",
        *,
        drop_first: bool = False,
        dummy_na: bool = False,
        onehot_dtype: type | None = np.int8,
        categories: dict[str, list] | None = None,
        handle_unknown: Literal["ignore", "error"] = "ignore",
        return_mapping: bool = False,
        inplace: bool = False,
    ):
        """Encode categorical features.

        Args:
            df: Input DataFrame.
            columns: Column name or iterable of columns to encode.
                - If None: uses all categorical columns (object/category/bool/string).
            encoding: Encoding strategy: "onehot" or "ordinal".
            drop_first: For one-hot, drop the first level to reduce collinearity.
            dummy_na: For one-hot, add a NaN indicator column.
            onehot_dtype: For one-hot, dtype of created dummy columns. Defaults to `np.int8`
                to produce 0/1 integer columns.
            categories: For ordinal encoding, optional explicit category order per column.
                ex: {"fuel": ["Diesel", "Petrol", "Hybrid"]}
            handle_unknown: For ordinal encoding, behavior when unseen categories appear.
                - "ignore": map unknowns to NaN
                - "error": raise ValueError
            return_mapping: If True and encoding="ordinal", return (df_encoded, mapping).
            inplace: If True, modify `df` in place and return it.
            df_enc = FE.encode_categorical_features(df_transformed, columns=["fuelType"], encoding="ordinal", categories={"fuelType": ["Diesel", "Petrol", "Hybrid"]})

        Returns:
            Encoded DataFrame, or (DataFrame, mapping) if return_mapping=True for ordinal.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if columns is None:
            target_cols = (
                df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
            )
        elif isinstance(columns, str):
            target_cols = [columns]
        else:
            target_cols = list(columns)

        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"columns not found in DataFrame: {missing_cols}")

        if not target_cols:
            out = df if inplace else df.copy()
            return out

        out = df if inplace else df.copy()
        encoding = str(encoding).lower().strip()

        if encoding == "onehot":
            # pd.get_dummies handles object/category/bool/string.
            before_cols = set(out.columns)
            try:
                out = pd.get_dummies(
                    out,
                    columns=target_cols,
                    drop_first=drop_first,
                    dummy_na=dummy_na,
                    dtype=onehot_dtype if onehot_dtype is not None else None,
                )
            except TypeError:
                # Fallback for older pandas versions without the `dtype` parameter.
                out = pd.get_dummies(
                    out,
                    columns=target_cols,
                    drop_first=drop_first,
                    dummy_na=dummy_na,
                )
                if onehot_dtype is not None:
                    new_cols = [c for c in out.columns if c not in before_cols]
                    out[new_cols] = out[new_cols].astype(onehot_dtype)
            return out

        if encoding == "ordinal":
            mapping: dict[str, dict] = {}
            for c in target_cols:
                series = out[c]
                col_categories = None
                if categories is not None and c in categories:
                    col_categories = list(categories[c])
                else:
                    # Stable default: sort unique non-null values as strings.
                    vals = series.dropna().unique().tolist()
                    col_categories = sorted(vals, key=lambda v: str(v))

                col_map = {v: i for i, v in enumerate(col_categories)}
                mapping[c] = col_map

                def _map_value(v):
                    if pd.isna(v):
                        return np.nan
                    if v in col_map:
                        return col_map[v]
                    if handle_unknown == "ignore":
                        return np.nan
                    raise ValueError(f"Unknown category {v!r} in column {c!r}")

                out[c] = series.map(_map_value).astype("Int64")

            if return_mapping:
                return out, mapping
            return out

        raise ValueError(f"Unknown encoding: {encoding!r}. Use 'onehot' or 'ordinal'.")

