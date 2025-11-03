
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import plotly.io as pio
pio.renderers.default = "notebook"   # or: "notebook_connected"


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
    if path.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)

    # --- tidy columns & types ---
    df.columns = [c.strip() for c in df.columns]
    if "refdate" in df.columns:
        df["refdate"] = pd.to_datetime(df["refdate"], dayfirst=True, errors="coerce")

    for col in PERCENT_COLS:
        if col in df.columns:
            s = (df[col].astype(str).str.replace('%', '', regex=False)
                               .str.replace(',', '', regex=False).str.strip())
            df[col] = pd.to_numeric(s, errors="coerce") / 100.0

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Asset Name", "GICS_sector", "Country Of Exposure"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=["refdate", "Asset Name"], how="any")

    # --- NEW: equity-only normalised weights ---
    # treat empty strings as nulls for GICS
    if "GICS_sector" in df.columns and "Weight (%)" in df.columns:
        g = df["GICS_sector"].replace(["", " ", "nan", "None"], np.nan)
        eq_mask = g.notna()

        # 1) Net-normalised within equity sleeve (sums to 1 across equities each date)
        eq_sum = df.loc[eq_mask].groupby("refdate")["Weight (%)"].transform("sum")
        df["Weight (eq norm)"] = np.where(
            eq_mask, df["Weight (%)"] / eq_sum, np.nan
        )

        # 2) Gross-normalised (leverage-neutral) within equity sleeve
        eq_gross = df.loc[eq_mask].groupby("refdate")["Mkt Value"].transform(lambda x: x.abs().sum())
        # fall back to weights if Mkt Value missing
        if eq_gross.isna().all() and "Weight (%)" in df.columns:
            eq_gross = df.loc[eq_mask].groupby("refdate")["Weight (%)"].transform(lambda x: x.abs().sum())
            base = df["Weight (%)"].fillna(0)
        else:
            base = df["Mkt Value"].fillna(0)

        gross_w = np.sign(base) * np.abs(base) / eq_gross
        df["Weight (eq gross)"] = np.where(eq_mask, gross_w, np.nan)

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
    dt = pd.to_datetime(df["refdate"]).max()
    full = df[df["refdate"] == dt].copy()
    d = full.sort_values(by, ascending=False).head(10).copy()

    # Build colour map
    names = d["Asset Name"].tolist()
    cmap = {n: ninetyone_colors[i % len(ninetyone_colors)] for i, n in enumerate(names)}

    # customdata = true portfolio weight in %
    # If your 'by' column is in fractions (0.0301), multiply by 100 for display
    custom_weight_pct = (d[by] * 100).to_numpy()

    fig = px.pie(
        d, names="Asset Name", values=by, color="Asset Name",
        color_discrete_map=cmap, title=f"Most Recent Top 10 holdings by {by}"
    )

    # Show percent of top-10 AND true portfolio weight
    fig.update_traces(
        sort=False,
        textposition="inside",
        textinfo="label",  # show label only
        customdata=custom_weight_pct,
        texttemplate="%{label}<br>%{customdata:.2f}%",  # show portfolio weight on the slice
        hovertemplate="<b>%{label}</b><br>"
                      "Portfolio weight: %{customdata:.2f}%<br>"
    )
    fig.update_layout(legend_traceorder="normal")
    return fig

def sector_pie_latest(df, weight_col="Weight (%)", sector_col="GICS_sector", date_col="refdate"):
    dt = pd.to_datetime(df[date_col]).max()
    d = (df[df[date_col] == dt]
         .dropna(subset=[sector_col])
         .assign(w=lambda x: x[weight_col] * (100 if x[weight_col].max() <= 1 else 1))
         .groupby(sector_col, as_index=False)['w'].sum())
    d['pct'] = d['w'] / d['w'].sum() * 100
    fig = px.pie(d, names=sector_col, values='pct',
                 title=f"Most Recent GICS Sector Allocation — {dt:%Y-%m-%d}", color_discrete_sequence=ninetyone_colors)
    return fig

def country_pie_latest(df, weight_col="Weight (%)", country_col="Country Of Exposure",
                       date_col="refdate", min_weight=0.001):
    dt = pd.to_datetime(df[date_col]).max()
    day = df[df[date_col] == dt].dropna(subset=[country_col]).copy()

    g = (day.groupby(country_col, as_index=False)[weight_col]
             .sum()
             .rename(columns={weight_col: "w_raw"}))

    g = g[g["w_raw"] > min_weight]
    total = g["w_raw"].sum()
    g["pct"] = np.where(total > 0, g["w_raw"] / total * 100.0, 0.0)

    fig = px.pie(g, names=country_col, values="pct",
                 title=f"Most Recent Country Allocation — {dt:%Y-%m-%d}", color_discrete_sequence=ninetyone_colors)
    fig.update_traces(textinfo="label+percent")
    return fig

def risk_contrib_bar(df, value_col="%Contribution to Total Risk",
                     name_col="Asset Name", date_col="refdate", top_n=10):
    dt  = pd.to_datetime(df[date_col]).max()
    day = df[df[date_col] == dt].copy()
    day["ctr_pct"] = day[value_col] *100

    d = (day.nlargest(top_n, "ctr_pct")
            .sort_values("ctr_pct", ascending=True))

    fig = px.bar(
        d, x="ctr_pct", y=name_col, orientation="h",
        title=f"Top {top_n} Contributors to Total Risk — {dt:%Y-%m-%d}",
        color=name_col, color_discrete_sequence=ninetyone_colors
    )
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="% Contribution to Total Risk",
        yaxis_title=None,
        margin=dict(l=80, r=30, t=60, b=40)
    )
    return fig

def sector_risk_contrib(df, sector_col="GICS_sector",
                        contrib_col="%Contribution to Total Risk",
                        date_col="refdate"):
    dt  = pd.to_datetime(df[date_col]).max()
    day = df[df[date_col] == dt].dropna(subset=[sector_col]).copy()
    day["ctr_pct"] = day[contrib_col]*100

    d = (day.groupby(sector_col, as_index=False)["ctr_pct"]
            .sum()
            .sort_values("ctr_pct", ascending=True))

    fig = px.bar(
        d, x="ctr_pct", y=sector_col, orientation="h",
        title=f"Sector Contribution to Total Risk — {dt:%Y-%m-%d}",
        color=sector_col, color_discrete_sequence=ninetyone_colors
    )
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="% Contribution to Total Risk",
        yaxis_title=None,
        margin=dict(l=80, r=30, t=60, b=40)
    )
    return fig

def country_risk_contrib(df, country_col="Country Of Exposure",
                         contrib_col="%Contribution to Total Risk",
                         date_col="refdate"):
    dt  = pd.to_datetime(df[date_col]).max()
    day = df[df[date_col] == dt].dropna(subset=[country_col]).copy()
    day["ctr_pct"] = day[contrib_col] * 100

    d = (day.groupby(country_col, as_index=False)["ctr_pct"]
            .sum()
            .sort_values("ctr_pct", ascending=True))

    fig = px.bar(
        d, x="ctr_pct", y=country_col, orientation="h",
        title=f"Country Contribution to Total Risk — {dt:%Y-%m-%d}",
        color=country_col, color_discrete_sequence=ninetyone_colors
    )
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="% Contribution to Total Risk",
        yaxis_title=None,
        margin=dict(l=80, r=30, t=60, b=40)
    )
    return fig

def _robust_price_return(g, price_col="PRICE"):
    prev = g[price_col].shift(1)
    ret  = (g[price_col] - prev) / prev
    # guard: previous price <= 0 or missing -> NaN; remove infs; clip splits
    ret = ret.where(prev > 0)
    ret = ret.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
    return ret

def portfolio_vs_benchmarks(df, weight_col="Weight (eq norm)"):
    d = (df[df["GICS_sector"].notna()][["refdate","Asset Name","PRICE","Weight (eq norm)"]]
        .dropna(subset=["refdate","Asset Name","PRICE"])
        .drop_duplicates(subset=["refdate","Asset Name"], keep="last")
        .sort_values(["Asset Name","refdate"])
        .copy())

    # per-asset price returns (no cross-asset leakage)
    d["ret_i"] = d.groupby("Asset Name")["PRICE"].pct_change()
    d["ret_i"] = d["ret_i"].replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)

    # lag equity-normalised weights
    d["w_lag"] = d.groupby("Asset Name")["Weight (eq norm)"].shift(1)

    # daily portfolio return = Σ w_{t-1,i} * r_{t,i}
    port = ((d["w_lag"] * d["ret_i"])
            .groupby(d["refdate"])
            .sum(min_count=1)
            .rename("port_ret")
            .reset_index())

    # TRI (base = 100)
    port["Portfolio TRI"] = 100 * (1 + port["port_ret"].fillna(0)).cumprod()

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

def compute_portfolio_benchmark_returns(
    df,
    price_col="PRICE",
    weight_col="Weight (%)",          # or "Weight (eq norm)" if you prefer
    active_col="Active Weight (%)",
    id_col="Asset Name",
    base=100.0,
):
    d = df.copy()

    # 1) Equities only
    if "GICS_sector" in d.columns:
        d["GICS_sector"] = d["GICS_sector"].replace(["", " ", "nan", "None"], np.nan)
        d = d[d["GICS_sector"].notna()].copy()

    # 2) Keep only needed cols, dedupe, sort
    keep = ["refdate", id_col, price_col, weight_col, active_col]
    d = (d[keep]
           .dropna(subset=["refdate", id_col, price_col])
           .drop_duplicates(subset=["refdate", id_col], keep="last")
           .sort_values([id_col, "refdate"])
           .copy())

    # 3) Robust per-asset PRICE return
    prev = d.groupby(id_col)[price_col].shift(1)
    d["ret_i"] = ((d[price_col] - prev) / prev).where(prev > 0)
    d["ret_i"] = d["ret_i"].replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)

    # 4) Derived benchmark weights (normalised within the equity set)
    d["b_norm"] = (d[weight_col] - d[active_col]).clip(lower=0)
    sb = d.groupby("refdate")["b_norm"].transform("sum")
    d.loc[sb.gt(0), "b_norm"] = d.loc[sb.gt(0), "b_norm"] / sb

    # 5) Lag weights
    d["w_lag"] = d.groupby(id_col)[weight_col].shift(1).fillna(0)
    d["b_lag"] = d.groupby(id_col)["b_norm"].shift(1).fillna(0)

    # 6) Build contribution columns and aggregate (no .apply)
    d["port_contrib"]  = d["w_lag"] * d["ret_i"]
    d["bench_contrib"] = d["b_lag"] * d["ret_i"]

    port  = (d.groupby("refdate", as_index=False)["port_contrib"]
               .sum()
               .rename(columns={"port_contrib": "port_ret"}))
    bench = (d.groupby("refdate", as_index=False)["bench_contrib"]
               .sum()
               .rename(columns={"bench_contrib": "bench_ret"}))

    out = port.merge(bench, on="refdate", how="inner").sort_values("refdate")
    out["active_ret"]     = out["port_ret"] - out["bench_ret"]
    out["Portfolio TRI"]  = base * (1 + out["port_ret"].fillna(0)).cumprod()
    out["Benchmark TRI"]  = base * (1 + out["bench_ret"].fillna(0)).cumprod()
    out["Active TRI"]     = out["Portfolio TRI"] - out["Benchmark TRI"]
    return out

def monthly_bar_px_all(df, colors=[
    "#0E6F63",
    "#F4A78E"]):
    import pandas as pd, plotly.express as px
    out = compute_portfolio_benchmark_returns(df)
    d = out.assign(refdate=pd.to_datetime(out['refdate'])).set_index('refdate')

    # calendar-month returns over entire history
    m = ((1 + d[['port_ret','bench_ret']]).groupby(pd.Grouper(freq='M')).prod() - 1).reset_index()
    m['YearMonth'] = m['refdate'].dt.to_period('M').astype(str)

    long = m.melt(id_vars='YearMonth', value_vars=['port_ret','bench_ret'],
                  var_name='Series', value_name='Return')

    fig = px.bar(long, x='YearMonth', y='Return', color='Series',
                 barmode='group', labels={'YearMonth':'Month','Return':'Return'},
                 title='Monthly Returns – Portfolio vs Benchmark',
                 color_discrete_sequence=colors)
    fig.update_yaxes(tickformat='.1%')
    fig.update_traces(hovertemplate='%{y:.2%}')
    fig.update_layout(legend_title='')
    fig.update_xaxes(tickangle=-45)
    return fig

def plot_port_vs_bench(df):
    df_compare = compute_portfolio_benchmark_returns(df)
    fig = px.line(df_compare, x = 'refdate', y = ['Portfolio TRI', 'Benchmark TRI'], color_discrete_sequence=ninetyone_colors, title= 'Portfolio vs Benchmark price return index')
    fig.show()

def sector_exposure_area(df, weight_col="Weight (%)", sector_col="GICS_sector", date_col="refdate"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.dropna(subset=[sector_col])
    g = (d.groupby([date_col, sector_col], as_index=False)[weight_col]
           .sum()
           .sort_values(date_col))
    fig = px.area(
        g, x=date_col, y=weight_col, color=sector_col, groupnorm="fraction",
        title="Sector Exposure Over Time (stacked 100%)",
        color_discrete_sequence=ninetyone_colors
    )
    fig.update_yaxes(tickformat=".0%", title="Weight")
    fig.update_layout(legend_title="Sector", margin=dict(l=40, r=20, t=60, b=40))
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1%}<extra></extra>")
    return fig

def country_exposure_area(df, weight_col="Weight (%)", country_col="Country Of Exposure", date_col="refdate"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.dropna(subset=[country_col])
    g = (d.groupby([date_col, country_col], as_index=False)[weight_col]
           .sum()
           .sort_values(date_col))
    fig = px.area(
        g, x=date_col, y=weight_col, color=country_col, groupnorm="fraction",
        title="Country Exposure Over Time (stacked 100%)",
    color_discrete_sequence=ninetyone_colors)
    fig.update_yaxes(tickformat=".0%", title="Weight")
    fig.update_layout(legend_title="Country", margin=dict(l=40, r=20, t=60, b=40))
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1%}<extra></extra>")
    return fig

def _attrib_area(df, group_col, weight_col="Weight (eq norm)",
                 id_col="Asset Name", date_col="refdate", title="Return attribution"):
    # keep only what we need & clean
    d = (df[df[group_col].notna()][[date_col, id_col, group_col, "PRICE", weight_col]]
           .dropna(subset=[date_col, id_col, "PRICE"])
           .drop_duplicates(subset=[date_col, id_col], keep="last")
           .sort_values([id_col, date_col])
           .copy())

    # per-asset simple return
    d["ret_i"] = d.groupby(id_col)["PRICE"].pct_change()
    d["ret_i"] = d["ret_i"].replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)

    # beginning-of-period weights (already clean & in fraction for eq-norm)
    d["w_lag"] = d.groupby(id_col)[weight_col].shift(1)

    # period contribution = sum_i w_{t-1,i} * r_{t,i} per group
    g = ((d["w_lag"] * d["ret_i"])
         .groupby([d[date_col], d[group_col]])
         .sum(min_count=1)
         .rename("contrib")
         .reset_index()
         .sort_values(date_col))

    # cumulative contribution by group (for stacked area)
    g["cum_contrib"] = g.groupby(group_col)["contrib"].cumsum()

    fig = px.area(
        g, x=date_col, y="cum_contrib", color=group_col,
        title=title, color_discrete_sequence=ninetyone_colors
    )
    fig.update_yaxes(tickformat=".2%")
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2%}<extra></extra>")
    fig.update_layout(legend_title=group_col, margin=dict(l=40, r=20, t=60, b=40))
    return fig

# Wrappers
def sector_return_contrib_area(df, **kw):
    return _attrib_area(df, group_col="GICS_sector",
                        title="Cumulative Return Attribution by Sector", **kw)

def country_return_contrib_area(df, **kw):
    return _attrib_area(df, group_col="Country Of Exposure",
                        title="Cumulative Return Attribution by Country", **kw)

def _attrib_totals_bar(df, group_col, weight_col="Weight (eq norm)",
                       id_col="Asset Name", date_col="refdate",
                       title="Total Return Attribution"):
    # Clean + prep (same pattern you used)
    d = (df[df[group_col].notna()][[date_col, id_col, group_col, "PRICE", weight_col]]
           .dropna(subset=[date_col, id_col, "PRICE"])
           .drop_duplicates(subset=[date_col, id_col], keep="last")
           .sort_values([id_col, date_col])
           .copy())

    # Per-asset return and lagged weights
    d["ret_i"] = d.groupby(id_col)["PRICE"].pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
    d["w_lag"] = d.groupby(id_col)[weight_col].shift(1)

    # Daily group contribution
    g = ((d["w_lag"] * d["ret_i"])
         .groupby([d[date_col], d[group_col]])
         .sum(min_count=1)
         .rename("contrib")
         .reset_index())

    if g.empty:
        raise ValueError("No contribution rows to plot (check inputs).")

    # Cumulative to most recent date
    g = g.sort_values(date_col)
    g["cum_contrib"] = g.groupby(group_col)["contrib"].cumsum()
    last_dt = g[date_col].max()
    t = (g[g[date_col] == last_dt][[group_col, "cum_contrib"]]
         .sort_values("cum_contrib", ascending=True))
    t = t[t['cum_contrib'].abs() > 0.001]

    # Bar chart
    fig = px.bar(
        t, x="cum_contrib", y=group_col, orientation="h",
        title=f"{title} — through {pd.to_datetime(last_dt):%Y-%m-%d}",
        color=group_col, color_discrete_sequence=ninetyone_colors
    )
    fig.update_traces(texttemplate="%{x:.2%}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Cumulative Contribution",
        yaxis_title=None,
        margin=dict(l=100, r=30, t=60, b=40)
    )
    fig.update_xaxes(tickformat=".1%")
    return fig

# Wrappers
def sector_return_contrib_totals_bar(df, **kw):
    return _attrib_totals_bar(
        df, group_col="GICS_sector",
        title="Total Return Attribution by Sector", **kw
    )

def country_return_contrib_totals_bar(df, **kw):
    return _attrib_totals_bar(
        df, group_col="Country Of Exposure",
        title="Total Return Attribution by Country", **kw
    )

def _asset_cum_contrib(df, weight_col="Weight (eq norm)",
                       id_col="Asset Name", date_col="refdate"):
    # minimal prep: keep what's needed, dedupe per date-asset, sort
    d = (df[[date_col, id_col, "PRICE", weight_col]]
           .dropna(subset=[date_col, id_col, "PRICE"])
           .drop_duplicates(subset=[date_col, id_col], keep="last")
           .sort_values([id_col, date_col])
           .copy())

    # per-asset returns and lagged weights
    d["ret_i"] = d.groupby(id_col)["PRICE"].pct_change()
    d["ret_i"] = d["ret_i"].replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
    d["w_lag"] = d.groupby(id_col)[weight_col].shift(1)

    # daily contrib per asset, then cumulative to last date
    g = ((d["w_lag"] * d["ret_i"])
         .groupby([d[date_col], d[id_col]])
         .sum(min_count=1)
         .rename("contrib")
         .reset_index()
         .sort_values(date_col))

    g["cum_contrib"] = g.groupby(id_col)["contrib"].cumsum()
    last_dt = g[date_col].max()
    t = (g[g[date_col] == last_dt][[id_col, "cum_contrib"]]
           .dropna()
           .copy())
    return t, last_dt

def top_share_contributors_bar(df, top_n=10, weight_col="Weight (eq norm)",
                               id_col="Asset Name", date_col="refdate"):
    t, last_dt = _asset_cum_contrib(df, weight_col, id_col, date_col)
    top = (t.nlargest(top_n, "cum_contrib")
             .sort_values("cum_contrib", ascending=True))
    fig = px.bar(
        top, x="cum_contrib", y=id_col, orientation="h",
        title=f"Top {top_n} Share Contributors — through {pd.to_datetime(last_dt):%Y-%m-%d}",
        color=id_col, color_discrete_sequence=ninetyone_colors
    )
    fig.update_traces(texttemplate="%{x:.2%}", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Cumulative Contribution", yaxis_title=None,
                      margin=dict(l=120, r=30, t=60, b=40))
    fig.update_xaxes(tickformat=".1%")
    return fig

def top_share_detractors_bar(df, top_n=10, weight_col="Weight (eq norm)",
                             id_col="Asset Name", date_col="refdate"):
    t, last_dt = _asset_cum_contrib(df, weight_col, id_col, date_col)
    det = (t.nsmallest(top_n, "cum_contrib")
             .sort_values("cum_contrib", ascending=True))
    fig = px.bar(
        det, x="cum_contrib", y=id_col, orientation="h",
        title=f"Top {top_n} Share Detractors — through {pd.to_datetime(last_dt):%Y-%m-%d}",
        color=id_col, color_discrete_sequence=ninetyone_colors
    )
    fig.update_traces(texttemplate="%{x:.2%}", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Cumulative Contribution", yaxis_title=None,
                      margin=dict(l=120, r=30, t=60, b=40))
    fig.update_xaxes(tickformat=".1%")
    return fig

def portfolio_risk_stats(df):
    import numpy as np, pandas as pd
    out = compute_portfolio_benchmark_returns(df).dropna(subset=['port_ret','bench_ret']).copy()

    port, bench = out['port_ret'], out['bench_ret']
    active = port - bench
    ann = 252  # trading days per year

    def ann_ret(x): return (1 + x.mean())**ann - 1
    def ann_vol(x): return x.std(ddof=0) * np.sqrt(ann)

    # cumulative series for drawdowns
    port_tr = (1 + port).cumprod()
    bench_tr = (1 + bench).cumprod()
    dd = (port_tr / port_tr.cummax() - 1).min()

    stats = {
        'Ann. Return (Portfolio)': ann_ret(port),
        'Ann. Return (Benchmark)': ann_ret(bench),
        'Active Return (Ann.)'   : ann_ret(port) - ann_ret(bench),
        'Volatility (Ann.)'      : ann_vol(port),
        'Tracking Error (Ann.)'  : ann_vol(active),
        'Sharpe Ratio'           : port.mean()/port.std(ddof=0)*np.sqrt(ann),
        'Information Ratio'      : active.mean()/active.std(ddof=0)*np.sqrt(ann),
        'Max Drawdown'           : dd,
    }
    return pd.DataFrame(stats, index=['Value']).T

def drawdown_px(df):
    import pandas as pd, plotly.express as px
    out = compute_portfolio_benchmark_returns(df).sort_values('refdate')
    v = (1 + out['port_ret']).cumprod()
    dd = v / v.cummax() - 1
    m = pd.DataFrame({'refdate': pd.to_datetime(out['refdate']), 'drawdown': dd})
    fig = px.area(m, x='refdate', y='drawdown', title='Portfolio Drawdown', color_discrete_sequence=ninetyone_colors)
    fig.update_yaxes(tickformat='.0%', range=[dd.min()*1.05, 0])
    fig.update_traces(hovertemplate='%{y:.2%}')
    return fig

def beta_avg_series_px(df):
    import pandas as pd, numpy as np, plotly.express as px
    d = df.copy()
    d['refdate'] = pd.to_datetime(d['refdate'], dayfirst=True, errors='coerce')
    d['w'] = pd.to_numeric(d['Weight (%)'].astype(str).str.rstrip('%'), errors='coerce')/100.0
    d['beta'] = pd.to_numeric(d['Beta (Bmk)'], errors='coerce')
    s = d.dropna(subset=['refdate','w','beta']).groupby('refdate').apply(
        lambda x: (x['w']*x['beta']).sum() / x['w'].sum()
    ).reset_index(name='weighted_beta')
    fig = px.line(s, x='refdate', y='weighted_beta', title='Weighted Average Portfolio Beta', color_discrete_sequence=ninetyone_colors)
    fig.add_hline(y=1.0, line_dash='dash', annotation_text='Benchmark = 1.0', annotation_position='top left')
    fig.update_yaxes(title='Beta')
    fig.update_xaxes(title='')
    return fig

def sector_beta_bar_px(df, refdate=None):
    import pandas as pd, numpy as np, plotly.express as px
    d = df.copy()
    d['refdate'] = pd.to_datetime(d['refdate'], dayfirst=True, errors='coerce')
    d['w'] = pd.to_numeric(d['Weight (%)'].astype(str).str.rstrip('%'), errors='coerce')/100.0
    d['beta'] = pd.to_numeric(d['Beta (Bmk)'], errors='coerce')
    snap = d.loc[d['refdate'] == (pd.to_datetime(refdate) if refdate else d['refdate'].max())]
    g = (snap.dropna(subset=['GICS_sector','w','beta'])
              .groupby('GICS_sector')
              .apply(lambda x: (x['w']*x['beta']).sum()/x['w'].sum())
              .reset_index(name='sector_beta')
              .sort_values('sector_beta', ascending=False))
    fig = px.bar(g, x='GICS_sector', y='sector_beta', title='Sector Weighted-Average Beta (most recent)',
                 labels={'GICS_sector':'Sector','sector_beta':'Beta'}, color_discrete_sequence=ninetyone_colors)
    fig.add_hline(y=1.0, line_dash='dash', annotation_text='Benchmark = 1.0', annotation_position='top left')
    fig.update_layout(xaxis_tickangle=-30)
    return fig

def _to_pct(series):
    import pandas as pd
    return pd.to_numeric(series.astype(str).str.rstrip('%'), errors='coerce') / 100.0

def risk_contrib_area(df,
                      group_col="GICS_sector",
                      contrib_col="%Contribution to Total Risk",
                      date_col="refdate",
                      colors=None,
                      title_prefix="Total Risk Decomposition"):
    import pandas as pd, numpy as np, plotly.express as px

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")

    # Clean and convert
    d[group_col] = d[group_col].astype(str).str.strip()
    d.loc[d[group_col].isin(["", "nan", "None", "NaN"]), group_col] = np.nan
    d = d.dropna(subset=[date_col, group_col])
    d["pct"] = pd.to_numeric(d[contrib_col].astype(str).str.rstrip('%'), errors="coerce")

    g = (d.dropna(subset=["pct"])
           .groupby([date_col, group_col], as_index=False)["pct"].sum()
           .sort_values([date_col, "pct"], ascending=[True, False]))

    # Area chart
    fig = px.area(
        g, x=date_col, y="pct", color=group_col,
        title=f"{title_prefix} by {group_col}",
        labels={date_col:"", "pct":"% Contribution to Total Risk", group_col:group_col},
        color_discrete_sequence=colors
    )
    fig.update_yaxes(tickformat=".2%")
    fig.update_traces(hovertemplate="%{y:.2%}<br>%{fullData.name}")
    fig.update_layout(legend_title=group_col, margin=dict(l=40, r=20, t=60, b=40))
    return fig

# Convenience wrappers
def sector_risk_contrib_area(df, **kw):
    return risk_contrib_area(df, group_col="GICS_sector", **kw)

def country_risk_contrib_area(df, **kw):
    return risk_contrib_area(df, group_col="Country Of Exposure", **kw)


def risk_contrib_active_line(df,
                      group_col="GICS_sector",
                      contrib_col="%Contribution to Active Total Risk",
                      date_col="refdate",
                      colors=None,
                      title_prefix="Total Active Risk Decomposition"):
    import pandas as pd, numpy as np, plotly.express as px

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")

    # Clean and convert
    d[group_col] = d[group_col].astype(str).str.strip()
    d.loc[d[group_col].isin(["", "nan", "None", "NaN"]), group_col] = np.nan
    d = d.dropna(subset=[date_col, group_col])
    d["pct"] = pd.to_numeric(d[contrib_col].astype(str).str.rstrip('%'), errors="coerce")

    g = (d.dropna(subset=["pct"])
           .groupby([date_col, group_col], as_index=False)["pct"].sum()
           .sort_values([date_col, "pct"], ascending=[True, False]))

    # Area chart
    fig = px.area(
        g, x=date_col, y="pct", color=group_col,
        title=f"{title_prefix} by {group_col}",
        labels={date_col:"", "pct":"% Contribution to Active Total Risk", group_col:group_col},
        color_discrete_sequence=colors
    )
    fig.update_yaxes(tickformat=".2%")
    fig.update_traces(hovertemplate="%{y:.2%}<br>%{fullData.name}")
    fig.update_layout(legend_title=group_col, margin=dict(l=40, r=20, t=60, b=40))
    return fig

# Convenience wrappers
def sector_risk_act_contrib_area(df, **kw):
    return risk_contrib_active_line(df, group_col="GICS_sector", **kw)

def country_risk_act_contrib_area(df, **kw):
    return risk_contrib_active_line(df, group_col="Country Of Exposure", **kw)

def _prep(df):
    df = df.copy()
    df["refdate"] = pd.to_datetime(df["refdate"], dayfirst=True)
    w = pd.to_numeric(df["Weight (%)"].astype(str).str.replace("%","", regex=False), errors="coerce")
    df["w"] = np.where(w > 1, w/100.0, w)
    return df

def _wavg(g, col):
    g = g.dropna(subset=[col, "w"])
    if g.empty:
        return np.nan
    w = g["w"].to_numpy()
    a = g[col].to_numpy()
    wsum = np.nansum(w)
    if not np.isfinite(wsum) or wsum == 0:
        return np.nan               # <- key fix: skip groups with zero total weight
    return np.average(a, weights=w)

# 1) Portfolio total ESG through time
def fig_portfolio_esg(df, col="Overall ESG Score"):
    df = _prep(df)
    d = (df.groupby("refdate")
           .apply(lambda g: _wavg(g, col))
           .reset_index(name="ESG"))
    return px.line(d, x="refdate", y="ESG",
                   title="Portfolio ESG (Weighted Avg)", color_discrete_sequence=ninetyone_colors)

# 2) Portfolio E/S/G through time
def fig_portfolio_esg_pillars(df,
    cols=("Overall ESG Environmental Score","Overall ESG Social Score","Overall ESG Governance Score")):
    df = _prep(df)
    out = []
    for c in cols:
        out.append(df.groupby("refdate").apply(lambda g: _wavg(g, c)).rename(c))
    d = pd.concat(out, axis=1).reset_index()
    d = d.melt("refdate", var_name="Pillar", value_name="Score")
    return px.line(d, x="refdate", y="Score", color="Pillar",
                   title="Portfolio E / S / G (Weighted Avgs)", color_discrete_sequence=ninetyone_colors)

# 3) Sector average ESG through time
def fig_sector_esg(df, esg_col="Overall ESG Score", sector_col="GICS_sector"):
    df = _prep(df).dropna(subset=[sector_col])
    d = (df.groupby(["refdate", sector_col])
           .apply(lambda g: _wavg(g, esg_col))
           .reset_index(name="ESG"))
    return px.line(d, x="refdate", y="ESG", color=sector_col,
                   title="Sector ESG (Weighted Avg) Over Time", color_discrete_sequence=ninetyone_colors)