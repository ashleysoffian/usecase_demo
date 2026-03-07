import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import missingno as msno
import rfpimp
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

class EDA_Analysis:
    """
    EDA visualization toolkit for data analysis.
    
    Provides functions and visualizations for exploratory data analysis
    including missing data analysis, distributions, outliers, and correlations.
    """

    @staticmethod
    def _table_col_widths(headers, rows, *, min_col: float = 0.04, max_col: float = 0.30):
        """Compute relative column widths based on rendered text lengths."""
        if not headers:
            return []

        def _cell_len(v) -> int:
            s = "" if v is None else str(v)
            return max(len(s), 1)

        col_max = [len(str(h)) for h in headers]
        for row in rows:
            for j in range(min(len(headers), len(row))):
                col_max[j] = max(col_max[j], _cell_len(row[j]))

        # Convert lengths to weights; sqrt dampens very long text dominating.
        weights = np.sqrt(np.array(col_max, dtype=float))
        weights = np.maximum(weights, 1.0)
        widths = weights / weights.sum()

        # Enforce min/max per column and re-normalize.
        widths = np.clip(widths, min_col, max_col)
        widths = widths / widths.sum()
        return widths.tolist()

    @staticmethod
    def _table_figsize(n_rows: int, n_cols: int, *, row_height: float = 0.30, col_width: float = 1.05,
                      min_w: float = 6.5, max_w: float = 16.0, min_h: float = 1.6, pad_h: float = 1.0):
        """Compute a tight-ish figure size for table-like plots."""
        # +1 for header row
        h = max(min_h, (n_rows + 1) * row_height + pad_h)
        w = max(min_w, min(max_w, n_cols * col_width))
        return (w, h)
    
    @staticmethod
    def _set_plot_style(ax, title=None, xlabel=None, ylabel=None):
        """
        Apply consistent professional styling to plots.
        
        Args:
            ax: Matplotlib axis object
            title (str, optional): Plot title
            xlabel (str, optional): X-axis label
            ylabel (str, optional): Y-axis label
        """
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11, fontweight='medium')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11, fontweight='medium')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=10)

    @staticmethod
    def _get_numeric_columns(data, variables=None):
        """
        Helper method to get numeric columns from data.
        
        Args:
            data (DataFrame): Input dataframe
            variables (list, optional): Specific variables to filter
            
        Returns:
            list: List of numeric column names
            
        Raises:
            TypeError: If variables is not a list
        """
        numeric_cols = [col for col in data.columns if is_numeric_dtype(data[col])]
        
        if variables is None:
            return numeric_cols
        
        if not isinstance(variables, list):
            raise TypeError("Expected variables to be a list")
            
        return [col for col in variables if col in numeric_cols]

    @staticmethod
    def data_info(data):
        """
        Generate data info table as matplotlib figure.

        Args:
            data (DataFrame): Input dataframe

        Returns:
            Figure: Matplotlib figure with formatted table
        """
        n_rows = len(data)
        n_cols = len(data.columns)

        def _null_pct(null_count: int) -> str:
            if n_rows == 0:
                return "0.0%"
            return f"{(null_count / n_rows * 100):.1f}%"

        info_df = pd.DataFrame(
            {
                '#': range(n_cols),
                'Column': data.columns,
                'Non-Null Count': data.notna().sum().astype(int).to_numpy(),
                'Null Count': data.isna().sum().astype(int).to_numpy(),
                'Null %': [
                    _null_pct(int(x)) for x in data.isna().sum().astype(int).to_numpy()
                ],
                'Dtype': data.dtypes.astype(str).to_numpy(),
            }
        )

        info_data = info_df.to_numpy().tolist()
        headers = info_df.columns.tolist()
        col_widths = EDA_Analysis._table_col_widths(headers, info_data)

        fig_w, fig_h = EDA_Analysis._table_figsize(len(info_data), len(headers))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')

        table = ax.table(
            cellText=info_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=col_widths,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        # Header row
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(COLORS['primary'])
            cell.set_text_props(weight='bold', color='white')

        # Body cells
        for i in range(1, len(info_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor('white')
                cell.set_edgecolor('#CCCCCC')

        title_text = (
            "DataFrame Info\n"
            f"RangeIndex: {n_rows} entries, 0 to {max(n_rows - 1, 0)} | Data columns: {n_cols} total"
        )
        ax.set_title(title_text, fontsize=9, fontweight='bold', pad=20, loc='left')

        fig.tight_layout()
        return fig
    
    @staticmethod
    def descriptive_stats(data, top_value_maxlen: int = 30):
        """
        Generate descriptive statistics table as a matplotlib figure.

        - Numeric columns: count, missing, unique, mean, std, min, 25%, 50%, 75%, max
        - Non-numeric columns: count, missing, unique, top, freq

        Args:
            data (DataFrame): Input dataframe
            top_value_maxlen (int): Max length for 'Top' (categorical) value display

        Returns:
            Figure: Matplotlib figure with formatted table
        """
        n_rows = len(data)
        n_cols = len(data.columns)

        def _fmt(v) -> str:
            """Format values for table cells."""
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if isinstance(v, (np.integer, int)):
                return str(int(v))
            if isinstance(v, (np.floating, float)):
                # keep readable, avoid long floats
                return f"{float(v):.1f}"
            s = str(v)
            s = s.replace("\n", " ")
            if len(s) > top_value_maxlen:
                s = s[: top_value_maxlen - 3] + "..."
            return s

        numeric_cols = [c for c in data.columns if is_numeric_dtype(data[c])]
        categorical_cols = [c for c in data.columns if c not in numeric_cols]

        ordered_cols = numeric_cols + categorical_cols

        rows = []
        for i, col in enumerate(ordered_cols):
            s = data[col]
            non_null = int(s.notna().sum())
            missing = int(s.isna().sum())

            try:
                unique_count = int(s.nunique(dropna=True))
            except TypeError:
                # Fallback for unhashable values (e.g., lists/dicts in cells)
                unique_count = int(s.astype("string").nunique(dropna=True))

            row = {
                "#": i,
                "Column": col,
                "Count": non_null,
                "Missing": missing,
                "Mean": "",
                "Std": "",
                "Min": "",
                "25%": "",
                "50%": "",
                "75%": "",
                "Max": "",
                "Unique": unique_count,
                "Top": "",
                "Freq": "",
                "Dtype": str(s.dtype),
            }

            if is_numeric_dtype(s):
                d = s.describe(percentiles=[0.25, 0.5, 0.75])
                row.update(
                    {
                        "Mean": _fmt(d.get("mean")),
                        "Std": _fmt(d.get("std")),
                        "Min": _fmt(d.get("min")),
                        "25%": _fmt(d.get("25%")),
                        "50%": _fmt(d.get("50%")),
                        "75%": _fmt(d.get("75%")),
                        "Max": _fmt(d.get("max")),
                    }
                )
            else:
                # object/category/bool/datetime show top/unique/freq via describe
                d = s.astype("object").describe()
                row.update(
                    {
                        "Top": _fmt(d.get("top")),
                        "Freq": _fmt(d.get("freq")),
                    }
                )

            rows.append(row)

        stats_df = pd.DataFrame(
            rows,
            columns=[
                "#",
                "Column",
                "Count",
                "Missing",
                "Mean",
                "Std",
                "Min",
                "25%",
                "50%",
                "75%",
                "Max",
                "Unique",
                "Top",
                "Freq",
                "Dtype",
            ],
        )

        headers = stats_df.columns.tolist()
        table_data = stats_df.astype(str).to_numpy().tolist()

        col_widths = EDA_Analysis._table_col_widths(headers, table_data, min_col=0.03, max_col=0.22)
        fig_w, fig_h = EDA_Analysis._table_figsize(len(table_data), len(headers), col_width=0.95, min_w=9.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)

        # Header styling
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(weight="bold", color="white")

        # Body styling
        for r in range(1, len(table_data) + 1):
            for c in range(len(headers)):
                cell = table[(r, c)]
                cell.set_facecolor("white")
                cell.set_edgecolor("#CCCCCC")

        title_text = (
            "Descriptive Statistics\n"
            f"Rows: {n_rows} | Columns: {n_cols}"
        )
        ax.set_title(title_text, fontsize=9, fontweight="bold", pad=20, loc="left")

        fig.tight_layout()
        return fig

    @staticmethod
    def missing_plot(data):
        """
        Generate missing data visualization.

        Args:
            data (DataFrame): Input dataframe

        Returns:
            Figure: Missing data bar plot
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        msno.bar(data, ax=ax, color=COLORS['primary'], fontsize=9)
        ax.set_title('Missing Data Analysis', fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def hist_plot(data, variables=None, bins=30, combined=False):
        """
        Generate histogram plots for distribution analysis.

        Args:
            data (DataFrame): Input dataframe
            variables (list, optional): Specific variables to plot. Defaults to all numeric columns.
            bins (int, optional): Number of bins for histogram. Defaults to 30.
            combined (bool, optional): If True, combine all plots into one figure. Defaults to False.

        Returns:
            Figure or list: Single figure if combined=True, list of figures otherwise

        Raises:
            TypeError: If variables is not a list
        """
        cols = [c for c in EDA_Analysis._get_numeric_columns(data, variables) if not data[c].isnull().all()]
        if not cols:
            return [] if not combined else None

        from matplotlib.lines import Line2D

        def _plot_one(ax, series, title, *, show_density_label: bool):
            col_data = series.dropna()
            if col_data.empty:
                ax.set_visible(False)
                return

            mean_val = col_data.mean()
            median_val = col_data.median()

            ax.hist(
                col_data,
                bins=bins,
                color=COLORS['hist_fill'],
                edgecolor='white',
                alpha=0.8,
                linewidth=0.4,
            )

            ax2 = ax.twinx()
            col_data.plot(kind='kde', ax=ax2, color=COLORS['danger'], linewidth=1.2, alpha=0.7)
            ax2.set_ylabel('Density' if show_density_label else '')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_yticks([])

            ax.axvline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=1.0, alpha=0.7,
                       label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color=COLORS['success'], linestyle='--', linewidth=1.0, alpha=0.7,
                       label=f'Median: {median_val:.1f}')

            kde_line = Line2D([0], [0], color=COLORS['danger'], linewidth=1.2, alpha=0.7, label='KDE')
            handles, _ = ax.get_legend_handles_labels()
            handles.append(kde_line)

            ax.text(0.98, 0.98, f"Skew: {col_data.skew():.2f}", transform=ax.transAxes,
                    ha='right', va='top', fontsize=8)
            ax.legend(handles=handles, loc='center right', fontsize=8, framealpha=0.9)

            EDA_Analysis._set_plot_style(ax, title=title, xlabel='', ylabel='Frequency')
            ax.tick_params(labelsize=9)

        if combined:
            n_plots = len(cols)
            grid_cols = min(2, n_plots)
            grid_rows = int(np.ceil(n_plots / grid_cols))
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 3 * grid_rows))
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(cols):
                try:
                    _plot_one(axes[idx], data[col], col, show_density_label=False)
                except (TypeError, ValueError):
                    axes[idx].set_visible(False)

            for idx in range(len(cols), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle('Distribution Analysis', fontsize=14, fontweight='bold', y=1.00)
            fig.tight_layout()
            return fig

        figs = []
        for col in cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 2.5))
                _plot_one(ax, data[col], f'Distribution: {col}', show_density_label=True)
                fig.tight_layout()
                figs.append(fig)
            except (TypeError, ValueError):
                continue

        return figs

    @staticmethod
    def box_plot(data, variables=None, combined=False):
        """
        Generate box plots for outlier analysis.

        Args:
            data (DataFrame): Input dataframe
            variables (list, optional): Specific variables to plot. Defaults to all numeric columns.
            combined (bool, optional): If True, combine all plots into one figure. Defaults to False.

        Returns:
            Figure or list: Single figure if combined=True, list of figures otherwise

        Raises:
            TypeError: If variables is not a list
        """
        cols = [c for c in EDA_Analysis._get_numeric_columns(data, variables) if not data[c].isnull().all()]
        if not cols:
            return [] if not combined else None

        def _outlier_count(series: pd.Series) -> int:
            s = series.dropna()
            if s.empty:
                return 0
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                return 0
            return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        def _boxplot(ax, series: pd.Series, *, compact: bool):
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
                    "markerfacecolor": COLORS['danger'],
                    "markeredgecolor": COLORS['danger'],
                    "markersize": 4 if compact else 8,
                    **({} if compact else {"label": "Mean"}),
                },
                medianprops={
                    "color": COLORS['success'],
                    "linewidth": 1.5 if compact else 2,
                    **({} if compact else {"label": "Median"}),
                },
                boxprops={
                    "facecolor": COLORS['box_fill'],
                    "edgecolor": COLORS['primary'],
                    "linewidth": 1.0 if compact else 1.5,
                },
                whiskerprops={
                    "color": COLORS['primary'],
                    "linewidth": 1.0 if compact else 1.5,
                    "linestyle": "--",
                },
                capprops={
                    "color": COLORS['primary'],
                    "linewidth": 1.0 if compact else 1.5,
                },
                flierprops={
                    "marker": "o",
                    "markerfacecolor": COLORS['warning'],
                    "markeredgecolor": COLORS['warning'],
                    "markersize": 2.5 if compact else 5,
                    "alpha": 0.6,
                },
            )

            mean_val = float(s.mean())
            median_val = float(s.median())
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            outliers = _outlier_count(s)

            if compact:
                stats_text = f'μ={mean_val:.1f} | med={median_val:.1f} | IQR={iqr:.1f} | Out={outliers}'
                ax.text(
                    0.98,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=8,
                    style='italic',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3),
                )
                ax.set_yticks([])
                ax.tick_params(labelsize=9)
            else:
                stats_text = (
                    f'Mean: {mean_val:.2f} | Median: {median_val:.2f} | '
                    f'IQR: {iqr:.2f} | Outliers: {outliers}'
                )
                ax.text(0.5, 1.05, stats_text, transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=10, style='italic')

        if combined:
            n_plots = len(cols)
            grid_cols = min(2, n_plots)
            grid_rows = int(np.ceil(n_plots / grid_cols))

            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 2.5 * grid_rows))
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(cols):
                try:
                    ax = axes[idx]
                    _boxplot(ax, data[col], compact=True)
                    if ax.get_visible():
                        EDA_Analysis._set_plot_style(ax, title=col, xlabel='')
                except (TypeError, ValueError):
                    axes[idx].set_visible(False)

            for idx in range(len(cols), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle('Box Plot Analysis', fontsize=14, fontweight='bold', y=1.00)
            fig.tight_layout()
            plt.close("all")
            return fig

        figs = []
        for col in cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 1.5))
                _boxplot(ax, data[col], compact=False)
                if ax.get_visible():
                    EDA_Analysis._set_plot_style(ax, title=f'Box Plot: {col}', xlabel=col)
                fig.tight_layout()
                figs.append(fig)
                plt.close("all")
            except (TypeError, ValueError):
                continue

        return figs
    

    @staticmethod
    def categorical_bar_plot(data, variables=None, combined=False):
        """
        Generate bar plots for categorical variable distribution.

        Args:
            data (DataFrame): Input dataframe
            variables (list, optional): Specific categorical variables to plot. Defaults to all object columns.
            combined (bool, optional): If True, combine all plots into one figure. Defaults to False.

        Returns:
            Figure or list: Single figure if combined=True, list of figures otherwise
        """
        if variables is not None and not isinstance(variables, (list, tuple, set)):
            raise TypeError("variables must be a list/tuple/set of column names or None")

        # Get categorical columns
        if variables is None:
            cat_cols = data.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
        else:
            cat_cols = [col for col in variables if col in data.columns]

        if not cat_cols:
            return [] if not combined else None

        def _plot_one(ax, series: pd.Series, title: str, *, compact: bool):
            s = series.astype("object").where(series.notna(), "<missing>")
            value_counts = s.value_counts(dropna=False)
            if value_counts.empty:
                ax.set_visible(False)
                return

            x = np.arange(len(value_counts))
            bars = ax.bar(
                x,
                value_counts.values,
                color=COLORS["primary"],
                edgecolor="white",
                alpha=0.8,
            )

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8 if compact else 9,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(value_counts.index.astype(str), rotation=45, ha="right", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")
            missing = int(series.isna().sum())
            unique = int(series.dropna().nunique())
            total = int(len(series))
            top_share = float(value_counts.iloc[0] / total) if total else 0.0

            ax.text(
                0.98,
                0.98,
                f"Unique: {unique} | Missing: {missing} | Top: {top_share:.0%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3),
            )

            EDA_Analysis._set_plot_style(ax, title=title, xlabel="", ylabel="Frequency")
            ax.tick_params(labelsize=9)

        if combined:
            n_plots = len(cat_cols)
            grid_cols = min(2, n_plots)
            grid_rows = (n_plots + grid_cols - 1) // grid_cols

            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 4 * grid_rows))
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(cat_cols):
                try:
                    _plot_one(axes[idx], data[col], col, compact=True)
                except (TypeError, ValueError):
                    axes[idx].set_visible(False)

            for idx in range(len(cat_cols), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle("Distribution Analysis", fontsize=14, fontweight="bold", y=1.00)
            fig.tight_layout()
            plt.close(fig)
            return fig

        figs = []
        for col in cat_cols:
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                _plot_one(ax, data[col], f"Distribution: {col}", compact=False)
                fig.tight_layout()
                figs.append(fig)
                plt.close(fig)
            except (TypeError, ValueError):
                continue

        return figs

    @staticmethod
    def binary_bar_plot(data, variables=None, combined=False):
        """Generate bar plots for binary variable distribution.

        A "binary" variable is defined here as having exactly 2 unique non-null
        values (missing values are ignored for the binary check).

        Args:
            data (DataFrame): Input dataframe
            variables (list/tuple/set, optional): Specific columns to consider. Defaults to all columns.
            combined (bool, optional): If True, combine all plots into one figure. Defaults to False.

        Returns:
            Figure or list: Single figure if combined=True, list of figures otherwise
        """
        if variables is not None and not isinstance(variables, (list, tuple, set)):
            raise TypeError("variables must be a list/tuple/set of column names or None")

        cols = data.columns.tolist() if variables is None else [c for c in variables if c in data.columns]
        if not cols:
            return [] if not combined else None

        binary_cols = []
        for col in cols:
            s = data[col]
            try:
                nunique = int(s.dropna().nunique())
            except TypeError:
                nunique = int(s.dropna().astype("string").nunique())
            if nunique == 2:
                binary_cols.append(col)

        if not binary_cols:
            return [] if not combined else None

        def _ordered_index(idx):
            values = list(idx)
            # Prefer sensible ordering for common binary patterns.
            try:
                if all(isinstance(v, (bool, np.bool_)) for v in values):
                    ordered = [v for v in [False, True] if v in values]
                    return ordered if len(ordered) == len(values) else values
                # Try numeric ordering (e.g., 0/1)
                return sorted(values, key=lambda x: float(x))
            except Exception:
                return values

        def _plot_one(ax, series: pd.Series, title: str, *, compact: bool):
            non_null = series.dropna()
            if non_null.empty:
                ax.set_visible(False)
                return

            s = non_null.astype("object")
            value_counts = s.value_counts(dropna=False)
            if value_counts.empty:
                ax.set_visible(False)
                return

            order = _ordered_index(value_counts.index)
            value_counts = value_counts.reindex(order)

            x = np.arange(len(value_counts))
            colors = [COLORS["primary"], COLORS["secondary"]]
            bar_colors = [colors[i % len(colors)] for i in range(len(value_counts))]
            bars = ax.bar(
                x,
                value_counts.values,
                color=bar_colors,
                edgecolor="white",
                alpha=0.85,
            )

            total_non_null = int(non_null.shape[0])
            for bar in bars:
                height = bar.get_height()
                pct = (height / total_non_null) if total_non_null else 0.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)} ({pct:.0%})",
                    ha="center",
                    va="bottom",
                    fontsize=8 if compact else 9,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in value_counts.index], fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")

            missing = int(series.isna().sum())
            ax.text(
                0.98,
                0.98,
                f"Non-null: {total_non_null} | Missing: {missing}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3),
            )

            EDA_Analysis._set_plot_style(ax, title=title, xlabel="", ylabel="Count")

        if combined:
            n_plots = len(binary_cols)
            grid_cols = min(2, n_plots)
            grid_rows = (n_plots + grid_cols - 1) // grid_cols

            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 4 * grid_rows))
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(binary_cols):
                try:
                    _plot_one(axes[idx], data[col], col, compact=True)
                except (TypeError, ValueError):
                    axes[idx].set_visible(False)

            for idx in range(len(binary_cols), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle("Binary Variable Distribution", fontsize=14, fontweight="bold", y=1.00)
            fig.tight_layout()
            plt.close(fig)
            return fig

        figs = []
        for col in binary_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                _plot_one(ax, data[col], f"Binary Distribution: {col}", compact=False)
                fig.tight_layout()
                figs.append(fig)
                plt.close(fig)
            except (TypeError, ValueError):
                continue

        return figs

    @staticmethod
    def binary_crosstab_with_target(
        data: pd.DataFrame,
        target: str,
        variables=None,
        *,
        normalize: str | bool = "index",
        include_missing: bool = False,
    ) -> pd.DataFrame:
        """Create crosstabs for all binary variables against a target.

        Equivalent to running, for each binary column `col`:
            `pd.crosstab(df[col], df[target], normalize='index')`

        A "binary" variable is defined as a column having exactly 2 unique non-null
        values (missing values are ignored for the binary check).

        Args:
            data: Input dataframe
            target: Target column name
            variables: Optional list/tuple/set of columns to consider. Defaults to all columns.
            normalize: Passed through to `pd.crosstab(normalize=...)`.
                Common values: 'index' (row-wise), 'columns', 'all', or False/None.
            include_missing: If True, treats missing values as a '<missing>' category
                for both feature and target.

        Returns:
            A single DataFrame formed by concatenating crosstabs for each binary column.
            The index is a MultiIndex: (feature, feature_value).
        """
        if target not in data.columns:
            raise ValueError(f"Target '{target}' not found in dataframe")

        if variables is not None and not isinstance(variables, (list, tuple, set)):
            raise TypeError("variables must be a list/tuple/set of column names or None")

        cols = data.columns.tolist() if variables is None else [c for c in variables if c in data.columns]
        cols = [c for c in cols if c != target]
        if not cols:
            return pd.DataFrame()

        binary_cols: list[str] = []
        for col in cols:
            s = data[col]
            try:
                nunique = int(s.dropna().nunique())
            except TypeError:
                nunique = int(s.dropna().astype("string").nunique())
            if nunique == 2:
                binary_cols.append(col)

        if not binary_cols:
            return pd.DataFrame()

        tables: dict[str, pd.DataFrame] = {}
        for col in binary_cols:
            x = data[col]
            y = data[target]
            if include_missing:
                x = x.astype("object").where(x.notna(), "<missing>")
                y = y.astype("object").where(y.notna(), "<missing>")

            tables[col] = pd.crosstab(x, y, normalize=normalize)

        out = pd.concat(tables, axis=0, names=["feature", "value"])
        return out

    @staticmethod
    def categorical_target_boxplot(data, categorical_var, target='price', combined=False):
        """
        Generate box plots showing target distribution across categorical values.

        Args:
            data (DataFrame): Input dataframe
            categorical_var (str or list): Categorical variable(s) to analyze
            target (str): Target variable name. Defaults to 'price'.
            combined (bool): If True and categorical_var is a list, combine into one figure.

        Returns:
            Figure or list: Box plot figure(s)
        """
        if target not in data.columns:
            raise ValueError(f"Target '{target}' not found in dataframe")

        if isinstance(categorical_var, str):
            categorical_vars = [categorical_var]
        elif isinstance(categorical_var, (list, tuple, set)):
            categorical_vars = list(categorical_var)
        else:
            raise TypeError("categorical_var must be a column name or a list/tuple/set of column names")

        categorical_vars = [c for c in categorical_vars if c in data.columns]
        if not categorical_vars:
            return [] if not combined else None

        def _plot_one(ax, cat_var: str, *, compact: bool):
            cat_series = data[cat_var].astype("object").where(data[cat_var].notna(), "<missing>")
            order = cat_series.value_counts(dropna=False).index.tolist()

            box_data = [data.loc[cat_series == cat, target].dropna() for cat in order]
            if not any(len(s) for s in box_data):
                ax.set_visible(False)
                return

            ax.boxplot(
                box_data,
                labels=[str(x) for x in order],
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": COLORS["danger"],
                    "markeredgecolor": COLORS["danger"],
                    "markersize": 3 if compact else 5,
                },
                medianprops={"color": COLORS["success"], "linewidth": 1.5},
                boxprops={
                    "facecolor": COLORS["box_fill"],
                    "edgecolor": COLORS["primary"],
                    "linewidth": 1.5,
                },
                whiskerprops={"color": COLORS["primary"], "linewidth": 1.5},
                capprops={"color": COLORS["primary"], "linewidth": 1.5},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": COLORS["warning"],
                    "markersize": 2.5 if compact else 4,
                    "alpha": 0.5,
                },
            )

            ax.tick_params(axis="x", rotation=45, labelsize=9 if compact else 10)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")
            EDA_Analysis._set_plot_style(ax, title=f"{target} by {cat_var}", xlabel=cat_var, ylabel=target)

        if combined and len(categorical_vars) > 1:
            n_plots = len(categorical_vars)
            grid_cols = min(2, n_plots)
            grid_rows = (n_plots + grid_cols - 1) // grid_cols

            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 5 * grid_rows))
            axes = np.atleast_1d(axes).ravel()

            for idx, cat_var in enumerate(categorical_vars):
                try:
                    _plot_one(axes[idx], cat_var, compact=True)
                except (TypeError, ValueError, KeyError):
                    axes[idx].set_visible(False)

            for idx in range(len(categorical_vars), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(f"{target} Distribution by Categories", fontsize=14, fontweight="bold", y=1.00)
            fig.tight_layout()
            plt.close(fig)
            return fig

        figs = []
        for cat_var in categorical_vars:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                _plot_one(ax, cat_var, compact=False)
                fig.tight_layout()
                figs.append(fig)
                plt.close(fig)
            except (TypeError, ValueError, KeyError):
                continue

        if len(figs) == 1:
            return figs[0]
        return figs

    @staticmethod
    def corr_matrix(data, variables=None, method='pearson', figsize=(7, 5)):
        """
        Generate correlation matrix heatmap.

        Args:
            data (DataFrame): Input dataframe
            variables (list, optional): Specific variables to analyze. Defaults to all numeric columns.
            method (str, optional): Correlation method - 'pearson', 'kendall', or 'spearman'. 
                                   Defaults to 'pearson'.

        Returns:
            Figure: Correlation heatmap

        Raises:
            TypeError: If variables is not a list
        """
        cols = [c for c in EDA_Analysis._get_numeric_columns(data, variables) if not data[c].isnull().all()]
        if not cols:
            return None

        corr = data[cols].corr(method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        
        # Create heatmap with styling
        sns.heatmap(
            corr, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            square=True, 
            linewidths=1,
            linecolor='white',
            cbar_kws={
                "shrink": 0.8,
                "label": "Correlation Coefficient"
            },
            ax=ax,
            vmin=-1,
            vmax=1,
            annot_kws={"size": 10}
        )
        
        # Styling
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        plt.tight_layout()
        plt.close("all")
        
        return fig
    
    @staticmethod
    def mutual_information_with_target(
        data: pd.DataFrame,
        target: str,
        variables=None,
        *,
        dropna: str = "any",
        min_rows: int = 30,
        random_state: int = 42,
    ):
        """Compute mutual information between features and a target.

        - If target is numeric: uses `mutual_info_regression`
        - Else: uses `mutual_info_classif`

        Returns a DataFrame with columns:
            - feature
            - mutual_information
        """
        if target not in data.columns:
            raise ValueError(f"Target '{target}' not found in dataframe")

        if variables is not None and not isinstance(variables, (list, tuple, set)):
            raise TypeError("variables must be a list/tuple/set of column names or None")

        if variables is None:
            features = [c for c in data.columns if c != target]
        else:
            features = [c for c in variables if c in data.columns and c != target]

        X = data[features]
        y = data[target]

        df = pd.concat([X, y], axis=1)
        if dropna == "any":
            df = df.dropna(axis=0, how="any")
        elif dropna == "all":
            df = df.dropna(axis=0, how="all")
        else:
            raise ValueError("dropna must be 'any' or 'all'")

        # For MI classification, target must be present.
        df = df[df[target].notna()]

        if df.shape[0] < min_rows:
            return pd.DataFrame(columns=["feature", "mutual_information"])

        X_clean = df[features]
        y_clean = df[target]

        is_regression = bool(is_numeric_dtype(y_clean))

        # mutual_info_* requires numeric arrays; encode non-numeric columns.
        X_enc = pd.DataFrame(index=X_clean.index)
        discrete_features = []
        for col in X_clean.columns:
            s = X_clean[col]
            if is_numeric_dtype(s):
                X_enc[col] = s
                kind = getattr(s.dtype, "kind", "")
                if is_regression:
                    # Regression MI is unstable if high-cardinality integer features are treated as discrete.
                    # Default numeric features to continuous; only bool is discrete.
                    discrete_features.append(bool(kind in ("b",)))
                else:
                    # Classification: ints/bools discrete; floats continuous.
                    discrete_features.append(bool(kind in ("i", "u", "b")))
            else:
                X_enc[col] = s.astype("category").cat.codes
                discrete_features.append(True)

        if is_regression:
            mi = mutual_info_regression(
                X_enc,
                y_clean,
                discrete_features=discrete_features,
                random_state=random_state,
            )
        else:
            y_enc = y_clean.astype("category").cat.codes
            mi = mutual_info_classif(
                X_enc,
                y_enc,
                discrete_features=discrete_features,
                random_state=random_state,
            )

        return (
            pd.DataFrame({"feature": X_clean.columns, "mutual_information": mi})
            .sort_values("mutual_information", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def rfpimp_dependency_with_target(
        data: pd.DataFrame,
        target: str,
        variables=None,
        *,
        dropna: str = "any",
        min_rows: int = 30,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        """
        Compute model-based feature dependency with respect to a target
        using Random Forest permutation importance (rfpimp).

        Returns:
            DataFrame with columns:
            - Feature
            - RFPIMP Importance
        """

        if target not in data.columns:
            raise ValueError(f"Target '{target}' not found in dataframe")

        if variables is not None and not isinstance(variables, (list, tuple, set)):
            raise TypeError("variables must be a list/tuple/set of column names or None")

        features = (
            [c for c in data.columns if c != target]
            if variables is None
            else [c for c in variables if c in data.columns and c != target]
        )

        X = data[features]
        y = data[target]

        df = pd.concat([X, y], axis=1)
        if dropna == "any":
            df = df.dropna(axis=0, how="any")
        elif dropna == "all":
            df = df.dropna(axis=0, how="all")
        else:
            raise ValueError("dropna must be 'any' or 'all'")

        # Target must be present
        df = df[df[target].notna()]

        if df.shape[0] < min_rows:
            return pd.DataFrame(columns=["Feature", "RFPIMP Importance"])

        X_clean = df[features]
        y_clean = df[target]

        # Encode non-numeric features (trees accept integer codes)
        X_enc = pd.DataFrame(index=X_clean.index)
        for col in X_clean.columns:
            s = X_clean[col]
            X_enc[col] = s if is_numeric_dtype(s) else s.astype("category").cat.codes

        # Choose model type: categorical target -> classifier, numeric target -> regressor
        model = (
            RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            if is_numeric_dtype(y_clean)
            else RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        )

        model.fit(X_enc, y_clean)

        imp = rfpimp.importances(model, X_enc, y_clean)

        # rfpimp returns a DataFrame indexed by feature
        return (
            imp.reset_index()
            .rename(columns={"index": "Feature", "Importance": "RFPIMP Importance"})
            .sort_values("RFPIMP Importance", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def importance_with_target(
        data: pd.DataFrame,
        target: str,
        variables=None,
        *,
        dropna: str = "any",
        min_rows: int = 30,
        n_estimators: int = 200,
        random_state: int = 42,
        top_n: int | None = 30,
        sort_by: str = "avg_rank",
        return_df: bool = False,
    ):
        """Combine MI + rfpimp importance and render a styled table figure.

        Args:
            data: Input dataframe
            target: Target column name
            variables: Optional feature list (defaults to all columns except target)
            dropna: 'any' or 'all' (row-wise), applied consistently to both methods
            min_rows: Minimum rows required after cleaning
            n_estimators: RandomForest trees for rfpimp
            random_state: RNG seed for MI and RF
            top_n: If set, limit to top N features after sorting
            sort_by: 'avg_rank', 'rfpimp', or 'mi'
            return_df: If True, returns (fig, combined_df); else returns fig

        Returns:
            Figure (or Figure, DataFrame)
        """
        mi_df = EDA_Analysis.mutual_information_with_target(
            data,
            target,
            variables,
            dropna=dropna,
            min_rows=min_rows,
            random_state=random_state,
        )

        rf_df = EDA_Analysis.rfpimp_dependency_with_target(
            data,
            target,
            variables,
            dropna=dropna,
            min_rows=min_rows,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        mi_norm = mi_df.rename(columns={"feature": "Feature", "mutual_information": "Mutual Information"})
        rf_norm = rf_df.rename(columns={"RFPIMP Importance": "RFPIMP Importance"})

        combined = pd.merge(mi_norm, rf_norm, on="Feature", how="outer")
        combined["Mutual Information"] = combined["Mutual Information"].astype(float)
        combined["RFPIMP Importance"] = combined["RFPIMP Importance"].astype(float)

        # Fill missing with 0 for ranking/display consistency
        combined[["Mutual Information", "RFPIMP Importance"]] = combined[["Mutual Information", "RFPIMP Importance"]].fillna(0.0)

        def _fmt(v) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if isinstance(v, (np.integer, int)):
                return str(int(v))
            if isinstance(v, (np.floating, float)):
                return f"{float(v):.4f}"
            return str(v)

        table_df = pd.DataFrame(
            {
                "#": range(len(combined)),
                "Feature": combined["Feature"].astype(str).to_numpy(),
                "Mutual Information": [_fmt(x) for x in combined["Mutual Information"].to_numpy()],
                "RFPIMP Importance": [_fmt(x) for x in combined["RFPIMP Importance"].to_numpy()],
            }
        )

        table_data = table_df.to_numpy().tolist()
        headers = table_df.columns.tolist()
        col_widths = EDA_Analysis._table_col_widths(headers, table_data, min_col=0.05, max_col=0.35)

        fig_w, fig_h = EDA_Analysis._table_figsize(len(table_data), len(headers), col_width=1.1, min_w=8.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(weight="bold", color="white")

        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor("white")
                cell.set_edgecolor("#CCCCCC")

        title_text = (
            "Feature Importance Summary\n"
            f"Target: {target} | Rows: {len(data)} | Features: {len(table_df)}"
        )
        ax.set_title(title_text, fontsize=9, fontweight="bold", pad=20, loc="left")

        fig.tight_layout()
        if return_df:
            return fig, combined
        return fig


