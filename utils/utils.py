import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
import time
from sklearn.metrics import (average_precision_score, roc_curve, roc_auc_score, precision_recall_curve, log_loss)
import numpy as np
import matplotlib.pyplot as plt

class _StepTimer:
    def __init__(self, name, step, total_steps):
        self.name = name
        self.step = step
        self.total_steps = total_steps

    def __enter__(self):
        print(
            f"[Pipeline] .... (step {self.step} of {self.total_steps}) "
            f"Processing {self.name}",
            flush=True
        )
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        print(
            f"[Pipeline] .... (step {self.step} of {self.total_steps}) "
            f"Processing {self.name}, total={elapsed:6.2f}s",
            flush=True
        )



class PolarsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        impute_zeros=None,
        impute_minus_one=None,
        categorical_features=None,
        winsorize_variables=None,
        rare_tol=0.012,
        winsor_fold=3,
        verbose=True,
    ):
        self.impute_zeros = impute_zeros or []
        self.impute_minus_one = impute_minus_one or []
        self.categorical_features = categorical_features or []
        self.winsorize_variables = winsorize_variables or []
        self.rare_tol = rare_tol
        self.winsor_fold = winsor_fold
        self.verbose = verbose

        self.winsor_caps_ = {}
        self.frequent_categories_ = {}
        self.ordinal_mappings_ = {}

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    def fit(self, X, y):
        if not isinstance(X, pl.DataFrame):
            X = pl.DataFrame(X)

        y = pl.Series(y).alias("__target__")
        df = X.with_columns(y)

        steps = [
            "winsor_stats",
            "rare_labels",
            "ordinal_encoder",
        ]

        total_steps = len(steps)
        step_id = 1

        # ---- Winsor stats ----
        if self.verbose:
            with _StepTimer("winsor_stats", step_id, total_steps):
                self._fit_winsor(df)
        else:
            self._fit_winsor(df)
        step_id += 1

        # ---- Rare labels ----
        if self.verbose:
            with _StepTimer("rare_label_encoder", step_id, total_steps):
                self._fit_rare_labels(df)
        else:
            self._fit_rare_labels(df)
        step_id += 1

        # ---- Ordinal encoder ----
        if self.verbose:
            with _StepTimer("ordinal_encoder", step_id, total_steps):
                self._fit_ordinal(df)
        else:
            self._fit_ordinal(df)

        return self

    # --------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------
    def transform(self, X):
        if not isinstance(X, pl.DataFrame):
            X = pl.DataFrame(X)

        df = X.clone()

        steps = [
            "zero_imputer",
            "minus_one_imputer",
            "max_winsorizer",
            "categorical_imputer",
            "rare_label_encoder",
            "ordinal_encoder",
        ]

        total_steps = len(steps)
        step_id = 1

        # ---- Zero imputer ----
        if self.impute_zeros:
            if self.verbose:
                with _StepTimer("zero_imputer", step_id, total_steps):
                    df = self._zero_imputer(df)
            else:
                df = self._zero_imputer(df)
        step_id += 1

        # ---- Minus one imputer ----
        if self.impute_minus_one:
            if self.verbose:
                with _StepTimer("minus_one_imputer", step_id, total_steps):
                    df = self._minus_one_imputer(df)
            else:
                df = self._minus_one_imputer(df)
        step_id += 1

        # ---- Winsorizer ----
        if self.winsorize_variables:
            if self.verbose:
                with _StepTimer("max_winsorizer", step_id, total_steps):
                    df = self._winsorize(df)
            else:
                df = self._winsorize(df)
        step_id += 1

        # ---- Categorical imputer ----
        if self.categorical_features:
            if self.verbose:
                with _StepTimer("categorical_imputer", step_id, total_steps):
                    df = self._categorical_imputer(df)
            else:
                df = self._categorical_imputer(df)
        step_id += 1

        # ---- Rare labels ----
        if self.categorical_features:
            if self.verbose:
                with _StepTimer("rare_label_encoder", step_id, total_steps):
                    df = self._apply_rare_labels(df)
            else:
                df = self._apply_rare_labels(df)
        step_id += 1

        # ---- Ordinal encoder ----
        if self.categorical_features:
            if self.verbose:
                with _StepTimer("ordinal_encoder", step_id, total_steps):
                    df = self._apply_ordinal(df)
            else:
                df = self._apply_ordinal(df)

        return df

    # ================= INTERNAL METHODS =================

    def _fit_winsor(self, df):
        for c in self.winsorize_variables:
            q1, q3 = (
                df.select([
                    pl.col(c).quantile(0.25).alias("q1"),
                    pl.col(c).quantile(0.75).alias("q3"),
                ])
                .row(0)
            )
            self.winsor_caps_[c] = q3 + self.winsor_fold * (q3 - q1)

    def _fit_rare_labels(self, df):
        n = df.height
        for c in self.categorical_features:
            df = df.with_columns(pl.col(c).cast(pl.Utf8))

            vc = (
                df.get_column(c)
                .value_counts()
                .with_columns((pl.col("count") / n).alias("freq"))
            )

            self.frequent_categories_[c] = (
                vc.filter(pl.col("freq") >= self.rare_tol)
                .select(c)
            )

    def _fit_ordinal(self, df):
        for c in self.categorical_features:
            df = df.with_columns(pl.col(c).cast(pl.Utf8))
            self.ordinal_mappings_[c] = (
                df.group_by(c)
                  .agg(pl.mean("__target__").alias("mean_target"))
                  .sort("mean_target")
                  .with_row_index("ordinal")
                  .select(c, "ordinal")
            )

    def _zero_imputer(self, df):
        for c in self.impute_zeros:
            df = df.with_columns(pl.col(c).fill_null(0))
        return df

    def _minus_one_imputer(self, df):
        for c in self.impute_minus_one:
            # Primero manejamos la conversión de booleanos
            if df.schema[c] == pl.Boolean:
                df = df.with_columns(
                    pl.col(c).cast(pl.Int8).fill_null(-1)
                )
            else:
                # Para el resto (int, float), solo rellenamos nulos
                df = df.with_columns(
                    pl.col(c).fill_null(-1)
                )
        return df

    def _winsorize(self, df):
        for c, cap in self.winsor_caps_.items():
            df = df.with_columns(
                pl.when(pl.col(c) > cap)
                  .then(cap)
                  .otherwise(pl.col(c))
                  .alias(c)
            )
        return df

    def _categorical_imputer(self, df):
        for c in self.categorical_features:
            df = df.with_columns(pl.col(c).fill_null("MISSING"))
        return df

    def _apply_rare_labels(self, df):
        for c, frequent_df in self.frequent_categories_.items():

            df = df.with_columns(pl.col(c).cast(pl.Utf8))
            frequent_df = frequent_df.with_columns(pl.col(c).cast(pl.Utf8))

            df = df.join(
                frequent_df.with_columns(pl.lit(1).alias("__keep__")),
                on=c,
                how="left"
            )

            df = (
                df.with_columns(
                    pl.when(pl.col("__keep__").is_null())
                    .then(pl.lit("OTHER"))
                    .otherwise(pl.col(c))
                    .alias(c)
                )
                .drop("__keep__")
            )

        return df

    def _apply_ordinal(self, df):
        for c, mapping in self.ordinal_mappings_.items():
            df = df.with_columns(pl.col(c).cast(pl.Utf8))
            df = (
                df.join(mapping, on=c, how="left")
                  .with_columns(pl.col("ordinal").fill_null(-1).alias(c))
                  .drop("ordinal")
            )
        return df

####### Utils for model evaluation

def plot_roc(labels, prediction_scores, legend, color):
    '''
    Function to plot ROC curve
    '''
    fpr, tpr, _   = roc_curve(labels, prediction_scores, pos_label=1)
    auc           = roc_auc_score(labels, prediction_scores)
    legend_string = legend + ' ($AUC = {:0.4f}$)'.format(auc)  
    plt.plot(fpr, tpr, label=legend_string, color=color)
    pass

def plot_prc(labels, prediction_scores, legend, color):
    '''
    Function to plot PRC curve
    '''
    precision, recall, thresholds = precision_recall_curve(labels, prediction_scores)
    average_precision = average_precision_score(labels, prediction_scores)
    legend_string = legend + ' ($AP = {:0.4f}$)'.format(average_precision)  
    plt.plot(recall, precision, label=legend_string, color=color)
    pass

def plot_ks(labels, prediction_scores, color):
    '''
    Function to plot KS plot
    '''
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1
    plt.plot(thresholds, fnr, label='FNR (Class 1 Cum. Dist.)', color=color[0], lw=1.5, alpha=0.2)
    plt.plot(thresholds, tnr, label='TNR (Class 0 Cum. Dist.)', color=color[1], lw=1.5, alpha=0.2)

    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]
    
    return ks, t_

def format_plot(title, xlabel, ylabel):
    '''
    Function to add format to plot
    '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')
    plt.axis('square')
    plt.ylim((-0.05, 1.05))
    plt.legend()
    plt.tight_layout()
    pass


def plot_ks_2(labels, prediction_scores, legend, color):
    '''
    Function to plot KS plot
    '''
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores, pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    thresholds[0] = 1
    
    kss = tnr - fnr
    ks = kss[np.argmax(np.abs(kss))]
    t_ = thresholds[np.argmax(np.abs(kss))]
    legend_string = f'{legend} ($KS = {ks:0.4f}$; $x = {t_:0.4f}$)'
    plt.plot(thresholds, kss, label=legend_string, color=color, lw=1.5)
    plt.vlines(t_, ks, 0, colors=color, linestyles='dashed', alpha=0.4)
    
    return ks, t_