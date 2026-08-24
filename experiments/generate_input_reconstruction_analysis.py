"""Build the report-ready input-versus-reconstruction analysis on 30 models."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "outputs" / "final_windowed4096" / "eval"
TW_ROOT = ROOT / "slurm_runs" / "outputs" / "tw_input_reconstruction_eval_windowed4096"
TABLES = ROOT / "outputs" / "final" / "tables"
REPORT_TABLES = ROOT / "report" / "tables"
REPORT_SECTION = ROOT / "report" / "input_reconstruction_analysis.tex"
MAX_BATCHES = 10
METRICS = {"chamfer": "CD", "hausdorff": "HD95"}
CORRUPTIONS = {"gaussian_noise": "Noise", "temporal_shuffle": "Shuffle", "random_drop": "Drop"}
BASE_LOSSES = {"chamfer": "Chamfer", "density_aware_chamfer": "DCD", "hausdorff": "Hausdorff"}


def read_csvs(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("*_corruptions.csv"))
    if not paths:
        raise FileNotFoundError(f"No corruption CSVs found in {root}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "training_objective", "metric", "corruption", "corruption_level"]
    return raw.groupby(keys, as_index=False).agg(mean_reconstruction_value=("reconstruction_value", "mean"), mean_corrupted_input_value=("corrupted_input_value", "mean"), num_batches=("batch", "nunique"))


def main() -> None:
    base = read_csvs(BASE_ROOT)
    base = base[base.trained_loss.isin(BASE_LOSSES) & base.metric.isin(METRICS) & (base.batch < MAX_BATCHES)].copy()
    base["training_objective"] = base.trained_loss.map(BASE_LOSSES)
    tw = read_csvs(TW_ROOT)
    tw = tw[tw.metric.isin(METRICS) & (tw.batch < MAX_BATCHES)].copy()
    tw["training_objective"] = tw.trained_loss_time_weight.map(lambda value: f"TW-{value:g}")
    raw = pd.concat((base, tw), ignore_index=True)
    expected = {"Chamfer", "DCD", "Hausdorff", "TW-2", "TW-5"}
    if set(raw.training_objective.unique()) != expected:
        raise ValueError(f"Expected {expected}, found {set(raw.training_objective.unique())}")
    if raw.groupby(["dataset", "model", "training_objective"]).ngroups != 30:
        raise ValueError("The combined evaluation does not contain 30 configurations.")

    summary = aggregate(raw)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "input_reconstruction_30_summary.csv", index=False, float_format="%.8g")
    endpoints = summary.loc[summary.groupby(["dataset", "model", "training_objective", "metric", "corruption"])["corruption_level"].idxmax()].copy()
    endpoints["absolute_gain"] = endpoints.mean_corrupted_input_value - endpoints.mean_reconstruction_value
    endpoints["relative_gain_percent"] = 100 * endpoints.absolute_gain / endpoints.mean_corrupted_input_value
    endpoints["improved"] = endpoints.absolute_gain > 0
    gains = endpoints.groupby(["metric", "corruption"], as_index=False).agg(configurations=("improved", "size"), improved_configurations=("improved", "sum"), improved_percent=("improved", lambda values: 100 * values.mean()), median_relative_gain_percent=("relative_gain_percent", "median"))
    gains.to_csv(TABLES / "input_reconstruction_gain_30_summary.csv", index=False, float_format="%.4f")

    rows = []
    for metric in ("chamfer", "hausdorff"):
        for corruption in ("gaussian_noise", "temporal_shuffle", "random_drop"):
            item = gains[(gains.metric == metric) & (gains.corruption == corruption)].iloc[0]
            rows.append(f"{METRICS[metric]} & {CORRUPTIONS[corruption]} & {int(item.improved_configurations)}/{int(item.configurations)} ({item.improved_percent:.1f}\\%) & {item.median_relative_gain_percent:+.1f}\\% \\\\")

    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    table = [r"\begin{table}[t]", r"\centering", r"\caption{Direct comparison between the corrupted input and its reconstruction at the strongest corruption severity (noise $\sigma=0.10$, shuffle $\rho=1.00$, drop $\delta=0.50$), across all 30 main-protocol configurations. The median relative gain is $100(E_{\mathrm{input}}-E_{\mathrm{rec}})/E_{\mathrm{input}}$; positive values indicate an improvement by the autoencoder.}", r"\label{tab:input-reconstruction-gain}", r"\scriptsize", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrr}", r"\toprule", r"Metric & Corruption & Improved configs. & Median relative gain \\", r"\midrule", *rows[:3], r"\midrule", *rows[3:], r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    (REPORT_TABLES / "input_reconstruction_gain_summary.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    cd_noise = gains[(gains.metric == "chamfer") & (gains.corruption == "gaussian_noise")].iloc[0]
    hd_rows = gains[gains.metric == "hausdorff"].set_index("corruption")
    hd_sentence = ", ".join(f"{hd_rows.loc[corruption, 'improved_percent']:.1f}\\% of cases for {CORRUPTIONS[corruption].lower()}" for corruption in ("gaussian_noise", "temporal_shuffle", "random_drop"))
    section = [r"\subsubsection{Does reconstruction reduce the input corruption?}", r"\label{sec:input-reconstruction-comparison}", "", r"The robustness trends above quantify how reconstruction quality changes with corruption severity relative to the clean-input baseline. This does not by itself establish whether the autoencoder corrects the corrupted input. We therefore directly compare, using the same clean target $\mathcal{P}$,", r"\[", r"E_{\mathrm{input}}^{(d)}(c) = d(\widetilde{\mathcal{P}}_c,\mathcal{P}), \qquad E_{\mathrm{rec}}^{(d)}(c) = d(f_\theta(\widetilde{\mathcal{P}}_c),\mathcal{P}).", r"\]", r"For a lower-is-better metric, the correction gain is defined as $G^{(d)}(c) = E_{\mathrm{input}}^{(d)}(c)-E_{\mathrm{rec}}^{(d)}(c)$. Thus, $G>0$ means that the reconstruction is closer to the clean target than the corrupted input itself, while $G<0$ means that the autoencoder amplifies that error. Table~\ref{tab:input-reconstruction-gain} summarizes this direct comparison at the strongest tested severity across the 30 configurations of the main protocol.", "", rf"The result is metric-dependent. According to Chamfer Distance, reconstruction rarely improves temporal shuffling and never improves random point dropping: a dropped cloud remains geometrically close to the target under nearest-neighbour matching, whereas the decoded cloud introduces an additional reconstruction error. Gaussian noise is corrected by {cd_noise.improved_percent:.1f}\% of configurations under CD. In contrast, HD95 improves in {hd_sentence}, with positive median gains. Hence the autoencoders often reduce the large tail errors captured by HD95, but they should not be interpreted as universal denoisers: their effect depends on both the corruption mechanism and the evaluation metric.", "", r"\input{report/tables/input_reconstruction_gain_summary}"]
    REPORT_SECTION.write_text("\n".join(section) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
