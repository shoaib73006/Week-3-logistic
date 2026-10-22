"""Week 3 logistics EDA and visualization pipeline.

All data is simulated for educational use. Run from the repository root:
    python src/analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

for folder in (DATA_DIR, FIG_DIR, OUT_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SHIPMENTS = 1200
sns.set_theme(style="whitegrid", context="notebook")


def generate_dataset(n=N_SHIPMENTS, seed=RANDOM_SEED):
    """Create a reproducible synthetic shipment-level dataset."""
    rng = np.random.default_rng(seed)
    regions = np.array(["North", "South", "East", "West", "Central"])
    modes = np.array(["Road", "Rail", "Air", "Sea"])
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 273, n), unit="D"
    )
    region = rng.choice(regions, n, p=[.24, .22, .17, .22, .15])
    mode = rng.choice(modes, n, p=[.52, .20, .13, .15])
    distance = np.clip(rng.gamma(shape=2.2, scale=360, size=n) + 80, 50, 3500)
    weight = np.clip(rng.lognormal(mean=5.2, sigma=.8, size=n), 5, 1800)
    volume = np.clip(weight / rng.uniform(120, 350, n), .02, 18)

    speed = pd.Series(mode).map({"Road": 420, "Rail": 600, "Air": 1500, "Sea": 300}).to_numpy()
    mode_rate = pd.Series(mode).map({"Road": 16, "Rail": 9, "Air": 48, "Sea": 6}).to_numpy()
    planned = np.maximum(1, np.ceil(distance / speed + rng.normal(1.0, .7, n))).astype(int)
    fuel_cost = distance * (pd.Series(mode).map(
        {"Road": 4.2, "Rail": 2.0, "Air": 8.5, "Sea": 1.4}
    ).to_numpy()) * (0.7 + weight / 1800)
    handling_cost = 180 + weight * .55 + volume * 32 + rng.normal(0, 80, n)
    transport_cost = distance * mode_rate + weight * .85 + handling_cost + fuel_cost
    transport_cost = np.maximum(transport_cost, 250)

    # Synthetic delay process: longer routes, road mode, and operational noise increase risk.
    mode_delay = pd.Series(mode).map({"Road": 1.0, "Rail": .2, "Air": -.6, "Sea": .8}).to_numpy()
    delay_signal = (distance / 1300) + mode_delay + rng.normal(0, 1.15, n)
    delay_days = np.maximum(0, np.floor(delay_signal)).astype(int)
    actual = planned + delay_days
    status = np.where(delay_days > 0, "Delayed", "On Time")
    satisfaction = np.clip(
        np.rint(4.7 - delay_days * .8 - np.log1p(transport_cost) / 18 + rng.normal(0, .55, n)),
        1, 5
    ).astype(int)

    df = pd.DataFrame({
        "shipment_id": [f"SH{idx:05d}" for idx in range(1, n + 1)],
        "shipment_date": pd.to_datetime(dates),
        "region": region,
        "transport_mode": mode,
        "distance_km": np.round(distance, 1),
        "weight_kg": np.round(weight, 1),
        "shipment_volume_m3": np.round(volume, 2),
        "transport_cost_inr": np.round(transport_cost, 2),
        "planned_delivery_days": planned,
        "actual_delivery_days": actual,
        "delay_days": delay_days,
        "delivery_status": status,
        "fuel_cost_inr": np.round(fuel_cost, 2),
        "warehouse_handling_cost_inr": np.round(np.maximum(handling_cost, 0), 2),
        "customer_satisfaction_score": satisfaction,
    })
    return df


def create_visualizations(df):
    """Save charts that explain distributions, trends, and relationships."""
    paths = {}

    plt.figure(figsize=(9, 5))
    plt.hist(df["actual_delivery_days"], bins=18, edgecolor="black", alpha=0.8)
    plt.title("Distribution of Actual Delivery Time")
    plt.xlabel("Actual delivery time (days)")
    plt.ylabel("Shipment count")
    plt.tight_layout()
    paths["delivery_distribution"] = FIG_DIR / "01_delivery_time_distribution.png"
    plt.savefig(paths["delivery_distribution"], dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    df["transport_mode"].value_counts().plot(kind="bar", edgecolor="black")
    plt.title("Shipment Volume by Transport Mode")
    plt.xlabel("Transport mode")
    plt.ylabel("Number of shipments")
    plt.tight_layout()
    paths["mode_volume"] = FIG_DIR / "02_shipment_volume_by_mode.png"
    plt.savefig(paths["mode_volume"], dpi=160)
    plt.close()

    monthly = df.assign(month=df["shipment_date"].dt.to_period("M").dt.to_timestamp())
    trend = monthly.groupby("month", as_index=False).agg(
        shipments=("shipment_id", "count"),
        average_cost=("transport_cost_inr", "mean"),
        delay_rate=("delivery_status", lambda x: (x == "Delayed").mean() * 100)
    )
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(trend["month"], trend["shipments"], marker="o", label="Shipments")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Shipment count")
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(trend["month"], trend["delay_rate"], marker="s", linestyle="--", label="Delay rate (%)")
    ax2.set_ylabel("Delayed shipments (%)")
    plt.title("Monthly Shipment Volume and Delay Rate")
    fig.tight_layout()
    paths["monthly_trend"] = FIG_DIR / "03_monthly_volume_and_delay_rate.png"
    fig.savefig(paths["monthly_trend"], dpi=160)
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    df.boxplot(column="transport_cost_inr", by="transport_mode", ax=plt.gca(), grid=False)
    plt.title("Transportation Cost by Mode")
    plt.xlabel("Transport mode")
    plt.ylabel("Cost (INR)")
    plt.tight_layout()
    paths["cost_by_mode"] = FIG_DIR / "04_transport_cost_by_mode.png"
    plt.savefig(paths["cost_by_mode"], dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    corr_cols = [
        "distance_km", "weight_kg", "shipment_volume_m3", "transport_cost_inr",
        "planned_delivery_days", "actual_delivery_days", "delay_days",
        "fuel_cost_inr", "warehouse_handling_cost_inr", "customer_satisfaction_score"
    ]
    corr_values = df[corr_cols].corr(numeric_only=True)
    image = plt.imshow(corr_values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=75, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr_values.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Correlation Matrix of Logistics Metrics")
    plt.tight_layout()
    paths["correlation"] = FIG_DIR / "05_correlation_heatmap.png"
    plt.savefig(paths["correlation"], dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for mode_name, subset in df.groupby("transport_mode"):
        plt.scatter(subset["distance_km"], subset["transport_cost_inr"], label=mode_name, alpha=.55, s=18)
    plt.legend(title="Transport mode")
    plt.title("Distance vs Transportation Cost")
    plt.xlabel("Distance (km)")
    plt.ylabel("Transportation cost (INR)")
    plt.tight_layout()
    paths["distance_cost"] = FIG_DIR / "06_distance_vs_cost.png"
    plt.savefig(paths["distance_cost"], dpi=160)
    plt.close()

    return paths, trend


def build_report(df, stats, corr, regional, status_summary, paths, trend):
    """Create a Word report with methods, findings, chart explanations, and code notes."""
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.65)
    sec.bottom_margin = Inches(.65)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(9.5)

    title = doc.add_heading("Week 3 Task: Advanced Data Analysis and Visualization in Logistics", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Python EDA report | Synthetic educational dataset | Generated automatically")

    doc.add_heading("1. Executive Summary", level=1)
    delay_rate = (df["delivery_status"] == "Delayed").mean() * 100
    doc.add_paragraph(
        f"This report analyzes {len(df):,} simulated shipments dated from "
        f"{df['shipment_date'].min():%d %b %Y} to {df['shipment_date'].max():%d %b %Y}. "
        f"The overall delayed-shipment rate is {delay_rate:.1f}%. Mean actual delivery time is "
        f"{df['actual_delivery_days'].mean():.2f} days and median transportation cost is "
        f"₹{df['transport_cost_inr'].median():,.0f}. These are results from generated data, "
        "not measurements from a real logistics company."
    )

    doc.add_heading("2. Dataset Structure and Simulation", level=1)
    doc.add_paragraph(
        "Each row represents one shipment. The dataset includes shipment date, region, transport "
        "mode, distance, weight, volume, transportation cost, planned and actual delivery time, "
        "delay days, delivery status, fuel cost, warehouse handling cost, and a customer "
        "satisfaction score. A fixed random seed makes the dataset reproducible."
    )
    doc.add_paragraph(
        "The simulation intentionally builds plausible relationships: distance and transport mode "
        "affect travel time and cost; weight and volume contribute to cost; delays lower the "
        "synthetic satisfaction score. Correlation therefore describes designed patterns and does "
        "not prove causation."
    )

    doc.add_heading("3. Methodology", level=1)
    for item in [
        "Generate a shipment-level dataset with 1,200 observations and a fixed random seed.",
        "Validate row count, column types, duplicate shipment IDs, and missing values.",
        "Calculate mean, median, standard deviation, minimum, quartiles, and maximum for numeric fields.",
        "Calculate Pearson correlations for numeric logistics metrics.",
        "Summarize delivery status and region-level performance.",
        "Create distribution, count, time-trend, box, heatmap, and scatter visualizations.",
        "Translate observed patterns into operational hypotheses and recommendations."
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4. Data Quality Checks", level=1)
    doc.add_paragraph(
        f"Rows: {len(df):,}; columns: {df.shape[1]}; duplicate shipment IDs: "
        f"{df['shipment_id'].duplicated().sum()}; total missing cells: {int(df.isna().sum().sum())}. "
        "The generated dataset is constructed without intentional missing values. For real data, "
        "missing values, duplicates, impossible distances/costs, inconsistent dates, and extreme "
        "values should be investigated before modeling."
    )

    doc.add_heading("5. Descriptive Statistics", level=1)
    numeric_summary = stats.reset_index().rename(columns={"index": "metric"})
    table = doc.add_table(rows=1, cols=6)
    table.style = "Light Shading Accent 1"
    for i, h in enumerate(["Metric", "Mean", "Median", "Std. dev.", "Min", "Max"]):
        table.rows[0].cells[i].text = h
    show_cols = ["distance_km", "weight_kg", "transport_cost_inr", "actual_delivery_days",
                 "delay_days", "fuel_cost_inr", "customer_satisfaction_score"]
    for metric in show_cols:
        row = table.add_row().cells
        row[0].text = metric
        s = stats.loc[metric]
        for i, col in enumerate(["mean", "50%", "std", "min", "max"], start=1):
            row[i].text = f"{s[col]:,.2f}"

    doc.add_heading("6. Visualizations and Interpretation", level=1)
    explanations = {
        "delivery_distribution": (
            "Histogram with a density curve",
            "Shows the typical delivery duration, spread, and whether a long-delivery tail exists. "
            "A long right tail would indicate a smaller group of shipments taking much longer."
        ),
        "mode_volume": (
            "Count plot",
            "Compares shipment counts by transport mode. High-volume modes are important for "
            "capacity planning, staffing, and prioritizing performance monitoring."
        ),
        "monthly_trend": (
            "Dual-axis time-series line chart",
            "Displays monthly shipment volume alongside delay rate. It helps identify periods when "
            "demand or delays rise together; investigate staffing, weather, capacity, and route mix."
        ),
        "cost_by_mode": (
            "Box plot",
            "Compares cost distributions by mode, including median, variability, and outliers. "
            "Compare cost together with speed, distance, and service requirements rather than "
            "assuming the lowest-cost mode is always best."
        ),
        "correlation": (
            "Correlation heatmap",
            "Shows the direction and strength of pairwise linear relationships. Strong positive "
            "distance-cost or delay-satisfaction relationships can guide further analysis, but "
            "correlation is not proof of cause."
        ),
        "distance_cost": (
            "Scatter plot",
            "Shows how cost changes with distance and distinguishes transport modes. The spread "
            "around the pattern may reflect weight, volume, mode, and other cost components."
        )
    }
    for key, (method, interpretation) in explanations.items():
        doc.add_heading(method, level=2)
        doc.add_picture(str(paths[key]), width=Inches(6.0))
        doc.add_paragraph(interpretation)

    doc.add_heading("7. Key Analytical Findings", level=1)
    mode_summary = df.groupby("transport_mode").agg(
        shipments=("shipment_id", "count"),
        avg_cost=("transport_cost_inr", "mean"),
        delay_rate=("delivery_status", lambda x: (x == "Delayed").mean() * 100),
        avg_days=("actual_delivery_days", "mean")
    ).sort_values("delay_rate", ascending=False)
    worst_mode = mode_summary.index[0]
    cheapest_mode = mode_summary["avg_cost"].idxmin()
    doc.add_paragraph(
        f"• {delay_rate:.1f}% of simulated shipments are delayed. The mode with the highest "
        f"delay rate in this generated sample is {worst_mode} "
        f"({mode_summary.loc[worst_mode, 'delay_rate']:.1f}%)."
    )
    doc.add_paragraph(
        f"• {cheapest_mode} has the lowest average cost per shipment in this sample "
        f"(₹{mode_summary.loc[cheapest_mode, 'avg_cost']:,.0f}). This comparison is not adjusted "
        "for shipment distance, weight, volume, or service requirements."
    )
    distance_corr = df["distance_km"].corr(df["transport_cost_inr"])
    delay_sat_corr = df["delay_days"].corr(df["customer_satisfaction_score"])
    doc.add_paragraph(
        f"• Pearson correlation between distance and transport cost: {distance_corr:.2f}. "
        "A positive relationship is plausible because distance is part of the simulated cost formula."
    )
    doc.add_paragraph(
        f"• Pearson correlation between delay days and satisfaction: {delay_sat_corr:.2f}. "
        "This is an intentionally simulated relationship; it should be validated using real customer data."
    )
    doc.add_paragraph(
        "• Monthly results should be reviewed for spikes rather than relying only on an overall "
        "average. Regional summary CSVs can help identify where to investigate first."
    )

    doc.add_heading("8. Recommendations", level=1)
    for item in [
        "Track on-time delivery rate, average delay, cost per shipment, and cost per tonne-kilometre by month, region, and mode.",
        "Investigate high-delay modes or regions using route-level data, carrier records, weather, dispatch time, and warehouse dwell time.",
        "Compare transport modes on a like-for-like basis after controlling for distance, weight, volume, urgency, and service level.",
        "Set alert thresholds for rising delay rates and review monthly trends with operations teams.",
        "Use real shipment records and validate data quality before making commercial decisions or training predictive models.",
        "For a next step, test a delay-risk model with a time-based holdout set and report precision, recall, F1, and calibration—not accuracy alone."
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("9. Python Tools and Code Excerpt", level=1)
    doc.add_paragraph(
        "pandas is used for data frames, grouping, descriptive statistics, and CSV exports; NumPy "
        "generates reproducible synthetic data; Seaborn and Matplotlib create charts; python-docx "
        "writes the report and embeds chart images."
    )
    code = (
        "stats = df.select_dtypes(include='number').describe().T\\n"
        "correlation = df.select_dtypes(include='number').corr()\\n"
        "regional = df.groupby('region').agg(\\n"
        "    shipments=('shipment_id', 'count'),\\n"
        "    average_cost=('transport_cost_inr', 'mean'),\\n"
        "    delay_rate=('delivery_status', lambda s: (s == 'Delayed').mean() * 100)\\n"
        ")\\n"
        "sns.scatterplot(data=df, x='distance_km', y='transport_cost_inr', hue='transport_mode')"
    )
    p = doc.add_paragraph()
    p.style = doc.styles["Normal"]
    p.add_run(code).font.name = "Consolas"

    doc.add_heading("10. Limitations", level=1)
    doc.add_paragraph(
        "The dataset is hypothetical and the relationships are simulated. It omits real-world "
        "factors such as weather, traffic, carrier contracts, route networks, customs, failed delivery "
        "attempts, and warehouse dwell time. Results should not be generalized to an actual business."
    )
    doc.add_heading("Appendix: Output Files", level=1)
    doc.add_paragraph(
        "The project exports the raw synthetic dataset, descriptive statistics, correlation matrix, "
        "regional summary, delivery-status summary, monthly trend summary, six PNG charts, and this report."
    )
    report_path = REPORT_DIR / "Week3_Logistics_Analysis_Report.docx"
    doc.save(report_path)
    return report_path


def main():
    df = generate_dataset()
    data_path = DATA_DIR / "logistics_shipments.csv"
    df.to_csv(data_path, index=False)

    # EDA and validation
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().T
    stats.to_csv(OUT_DIR / "descriptive_statistics.csv")
    corr = numeric.corr()
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")

    regional = df.groupby("region").agg(
        shipments=("shipment_id", "count"),
        average_cost_inr=("transport_cost_inr", "mean"),
        median_delivery_days=("actual_delivery_days", "median"),
        delay_rate_pct=("delivery_status", lambda s: (s == "Delayed").mean() * 100),
        average_satisfaction=("customer_satisfaction_score", "mean")
    ).reset_index()
    regional.to_csv(OUT_DIR / "regional_summary.csv", index=False)

    status_summary = df["delivery_status"].value_counts().rename_axis("delivery_status").reset_index(name="shipments")
    status_summary["percentage"] = status_summary["shipments"] / len(df) * 100
    status_summary.to_csv(OUT_DIR / "delivery_status_summary.csv", index=False)

    paths, trend = create_visualizations(df)
    trend.to_csv(OUT_DIR / "monthly_trend_summary.csv", index=False)
    report_path = build_report(df, stats, corr, regional, status_summary, paths, trend)

    print("Pipeline completed successfully.")
    print(f"Dataset: {data_path}")
    print(f"Report:  {report_path}")
    print(f"Charts:  {FIG_DIR}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Missing cells: {int(df.isna().sum().sum())}")
    print(f"Duplicate shipment IDs: {int(df['shipment_id'].duplicated().sum())}")


if __name__ == "__main__":
    main()
