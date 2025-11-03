
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

PERCENT_COLS = ["Weight (%)","Active Weight (%)",
                "%Contribution to Active Total Risk","%Contribution to Total Risk"]
NUMERIC_COLS = ["Mkt Value","Total Risk","Marginal Contribution to Active Total Risk",
                "Marginal Contribution to Total Risk","Beta (Bmk)","PRICE",
                "Contribution to Beta (Bmk)","Market Capitalization",
                "Overall ESG Score","Overall ESG Environmental Score",
                "Overall ESG Social Score","Overall ESG Governance Score"]

ninetyone_colors = [
    "#0E6F63",
    "#1D8A79",
    "#F4A78E",
    "#6F7C78",
    "#5A2D3C",
    "#44B0A1",
    "#9BD8CC",
    "#E17061",
    "#6E1E33",
    "#A9B7B1",
    "#E6ECEA",
]

def load_data(path: str, sheet_name=None) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(('.xls','.xlsx')):
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "refdate" in df.columns:
        df["refdate"] = pd.to_datetime(df["refdate"], dayfirst=True, errors="coerce")
    for col in PERCENT_COLS:
        if col in df.columns:
            s = (df[col].astype(str).str.replace('%','', regex=False)
                           .str.replace(',','', regex=False).str.strip())
            df[col] = pd.to_numeric(s, errors="coerce")/100.0
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Asset Name","GICS_sector","Country Of Exposure"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=["refdate","Asset Name"], how="any")
    return df

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "%Contribution to Active Total Risk" in out.columns and "Total Risk" in out.columns:
        out["Active Risk Share"] = out["%Contribution to Active Total Risk"] * out["Total Risk"]
    if "Weight (%)" in out.columns and "Total Risk" in out.columns:
        out["Weighted Risk"] = out["Weight (%)"] * out["Total Risk"]
    if "Overall ESG Score" in out.columns:
        out["ESG_norm"] = out["Overall ESG Score"] / 10.0
    return out

class PortfolioAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = add_derived_columns(df)

    def timeseries(self) -> pd.DataFrame:
        g = self.df.groupby("refdate", as_index=False)
        agg = g.agg({
            "Mkt Value":"sum",
            "Weight (%)":"sum",
            "Active Weight (%)":"sum",
            "Total Risk":"mean",
            "Weighted Risk":"sum",
            "Beta (Bmk)":"mean",
            "ESG_norm":"mean",
            "Overall ESG Score":"mean"
        }).rename(columns={
            "Mkt Value":"Portfolio MV",
            "Weight (%)":"Total Weight",
            "Active Weight (%)":"Total Active Weight",
            "Total Risk":"Avg Total Risk",
            "Weighted Risk":"Wgt Risk Sum",
            "Beta (Bmk)":"Avg Beta",
            "ESG_norm":"Avg ESG (norm)",
            "Overall ESG Score":"Avg ESG"
        })
        return agg.sort_values("refdate")
    
def _latest_date(df: pd.DataFrame) -> pd.Timestamp:
    if "refdate" not in df.columns:
        raise ValueError("refdate column missing")
    return pd.to_datetime(df["refdate"]).max()

def top10_pie_latest(df: pd.DataFrame, by: str = "Weight (%)"):
    if by not in df.columns: 
        raise ValueError(f"{by} not in dataframe")
    dt = pd.to_datetime(df["refdate"]).max()
    d = df[df["refdate"] == dt].sort_values(by, ascending=False).head(10).copy()
    names = d["Asset Name"].tolist()
    cmap = {n: ninetyone_colors[i % len(ninetyone_colors)] for i, n in enumerate(names)}
    fig = px.pie(d, names="Asset Name", values=by, color="Asset Name",
                 color_discrete_map=cmap,
                 title=f"Top 10 holdings by {by} — {dt.date()}")
    fig.update_traces(sort=False, textposition="inside", textinfo="percent+label")
    fig.update_layout(legend_traceorder="normal")
    return fig

def _robust_price_return(g, price_col="PRICE"):
    prev = g[price_col].shift(1)
    ret  = (g[price_col] - prev) / prev
    # guard: previous price <= 0 or missing -> NaN; remove infs; clip splits
    ret = ret.where(prev > 0)
    ret = ret.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
    return ret

def portfolio_vs_benchmarks(df):
    d = df.sort_values(["Asset Name", "refdate"]).copy()

    # 1) normalise portfolio weights per date
    d["w_norm"] = d["Weight (%)"]
    s = d.groupby("refdate")["w_norm"].transform("sum")
    d.loc[s > 0, "w_norm"] = d.loc[s > 0, "w_norm"] / s

    # 2) robust PRICE returns
    d["ret_i"] = d.groupby("Asset Name", group_keys=False).apply(_robust_price_return)

    # 3) lag weights and aggregate
    d["w_lag"] = d.groupby("Asset Name")["w_norm"].shift(1).fillna(0)
    port = (
        d.assign(contrib=d["w_lag"] * d["ret_i"])
         .groupby("refdate", as_index=False)["contrib"].sum()
         .rename(columns={"contrib": "port_ret"})
    )
    port["Portfolio TRI"] = 100 * (1 + port["port_ret"].fillna(0)).cumprod()
    port = port[["refdate", "Portfolio TRI"]]

    # ---- Benchmarks (Yahoo) ----
    tickers = {"MSCI World":"URTH", "MSCI EM":"EEM", "SA All Share":"^J203.JO"}
    start, end = port["refdate"].min().strftime("%Y-%m-%d"), port["refdate"].max().strftime("%Y-%m-%d")
    data = yf.download(list(tickers.values()), start=start, end=end, progress=False)

    # handle single/multi index; use Close
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data[["Close"]].rename(columns={"Close": list(tickers.values())[0]})
    data = data.rename(columns={v:k for k,v in tickers.items()}).reset_index().rename(columns={"Date":"refdate"})

    # TRI for each benchmark
    for name in tickers.keys():
        if name in data.columns:
            data[name] = 100 * (1 + data[name].pct_change().fillna(0)).cumprod()

    merged = port.merge(data, on="refdate", how="inner")
    fig = px.line(
        merged, x="refdate",
        y=["Portfolio TRI", "MSCI World", "MSCI EM", "SA All Share"],
        title="Portfolio vs Global Benchmarks — Total Return Index (base=100)",
        color_discrete_sequence=ninetyone_colors[:4]
    )
    fig.update_layout(yaxis_title="TRI (base=100)", legend_title_text="")
    return fig, merged

def compute_portfolio_benchmark_returns(df,
                                        price_col="PRICE",
                                        weight_col="Weight (%)",
                                        active_col="Active Weight (%)",
                                        id_col="Asset Name"):
    d = df.sort_values([id_col, "refdate"]).copy()

    # normalise portfolio weights
    d["w_norm"] = d[weight_col]
    s = d.groupby("refdate")["w_norm"].transform("sum")
    d.loc[s > 0, "w_norm"] = d.loc[s > 0, "w_norm"] / s

    # reconstruct & normalise benchmark weights
    d["b_norm"] = (d[weight_col] - d[active_col]).clip(lower=0)
    sb = d.groupby("refdate")["b_norm"].transform("sum")
    d.loc[sb > 0, "b_norm"] = d.loc[sb > 0, "b_norm"] / sb

    # robust returns from PRICE
    d["ret_i"] = d.groupby(id_col, group_keys=False).apply(
        lambda g: _robust_price_return(g, price_col=price_col)
    )

    # lag weights
    d["w_lag"] = d.groupby(id_col)["w_norm"].shift(1).fillna(0)
    d["b_lag"] = d.groupby(id_col)["b_norm"].shift(1).fillna(0)

    g = d.groupby("refdate", as_index=False)
    port  = g.apply(lambda x: np.nansum(x["w_lag"] * x["ret_i"])).reset_index(name="port_ret")
    bench = g.apply(lambda x: np.nansum(x["b_lag"] * x["ret_i"])).reset_index(name="bench_ret")

    out = port.merge(bench, on="refdate", how="inner")
    out["active_ret"]    = out["port_ret"] - out["bench_ret"]
    out["Portfolio TRI"] = 100 * (1 + out["port_ret"].fillna(0)).cumprod()
    out["Benchmark TRI"] = 100 * (1 + out["bench_ret"].fillna(0)).cumprod()
    out["Active TRI"]    = out["Portfolio TRI"] - out["Benchmark TRI"]
    return out