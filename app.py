import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import os

# ==================== 1. 加载数据（支持本地文件 + 网页上传） ====================
@st.cache_data
def load_raw_data_from_bytes(file_bytes):
    """从上传的文件字节流加载数据"""
    df = pd.read_csv(file_bytes)
    return _build_raw_data(df)

@st.cache_data
def load_raw_data_from_local():
    """从本地 data.csv 加载数据"""
    df = pd.read_csv('data.csv')
    return _build_raw_data(df)

def _build_raw_data(df):
    """将 DataFrame 转换为 raw_data 字典结构"""
    raw_data = {}
    for day in df['day'].unique():
        raw_data[day] = {}
        for plate in df['plate'].unique():
            raw_data[day][plate] = {}
            sub = df[(df['day'] == day) & (df['plate'] == plate)]
            rows = sub['row'].unique()
            cols = sub['col'].unique()
            matrix = np.zeros((len(rows), len(cols)))
            for _, row in sub.iterrows():
                matrix[int(row['row']), int(row['col'])] = row['value']
            raw_data[day][plate] = matrix.tolist()
    return raw_data

# --- 数据加载逻辑 ---
if os.path.exists('data.csv'):
    # 本地有文件，直接加载（方便本地测试）
    raw_data = load_raw_data_from_local()
    st.sidebar.success("✅ 已从本地 data.csv 加载数据")
else:
    # 本地没有文件，显示上传按钮（适合云端部署）
    uploaded_file = st.sidebar.file_uploader(
        "📤 请上传 data.csv 文件",
        type=['csv'],
        help="上传包含实验数据的 CSV 文件"
    )
    if uploaded_file is not None:
        raw_data = load_raw_data_from_bytes(uploaded_file)
        st.sidebar.success("✅ 数据上传成功！")
    else:
        st.sidebar.warning("⚠️ 请上传 data.csv 文件以继续")
        st.stop()

# ==================== 2. 静态定义 ====================
blank_values = {1: 0.107, 3: 0.102, 5: 0.103, 7: 0.104}

sample_positions = {
    'W1': ('plate1', 0, [0,1,2]), 'W2': ('plate1', 1, [0,1,2]), 'W3': ('plate1', 2, [0,1,2]), 'W4': ('plate1', 3, [0,1,2]),
    'W5': ('plate1', 0, [3,4,5]), 'W6': ('plate1', 1, [3,4,5]), 'W7': ('plate1', 2, [3,4,5]), 'W8': ('plate1', 3, [3,4,5]),
    'W9': ('plate2', 0, [0,1,2]), 'W10': ('plate2', 1, [0,1,2]), 'W11': ('plate2', 2, [0,1,2]),
    'W12': ('plate2', 0, [3,4,5]), 'W13': ('plate2', 1, [3,4,5]), 'W14': ('plate2', 2, [3,4,5]),
    'Control': ('plate2', 3, [3,4,5])
}

days = sorted(raw_data.keys())

# ==================== 3. 自定义排序 ====================
def natural_sort_key(sample):
    if sample == "Control":
        return (0, 0)
    num = int(sample[1:])
    return (1, num)

samples = sorted(sample_positions.keys(), key=natural_sort_key)

# ==================== 4. 计算所有数据 ====================
@st.cache_data
def compute_all_data():
    all_data = {}
    growth_rates = {}
    for sample in samples:
        plate, row, cols = sample_positions[sample]
        sample_data = {}
        for day in days:
            vals = [raw_data[day][plate][row][c] - blank_values[day] for c in cols]
            sample_data[day] = {'mean': np.mean(vals), 'sd': np.std(vals, ddof=1)}
        all_data[sample] = sample_data
        
        rates = {}
        rates['1-3'] = (sample_data[3]['mean'] - sample_data[1]['mean']) / 2
        rates['3-5'] = (sample_data[5]['mean'] - sample_data[3]['mean']) / 2
        rates['5-7'] = (sample_data[7]['mean'] - sample_data[5]['mean']) / 2
        rates['1-7'] = (sample_data[7]['mean'] - sample_data[1]['mean']) / 6
        growth_rates[sample] = rates
        
    return all_data, growth_rates

all_data, growth_rates = compute_all_data()
CONTROL_SAMPLE = "Control"

# ==================== 5. UI ====================
st.set_page_config(layout="wide")
st.title("🧫 Bacterial Growth Curves – Interactive Dashboard")

with st.expander("📖 How to read these curves? (Click to expand)"):
    st.markdown("""
    - **Y‑axis (OD)** : Optical density – higher values indicate more bacterial cells (turbidity).
    - **Curve trend** : Upward slope means bacterial growth; flattening indicates nutrient depletion or stationary phase.
    - **Slope (growth rate)** : The table below shows slopes for 4 intervals. Larger values = faster growth during that period.
        - **D1→D3** : Early exponential (burst phase)  
        - **D3→D5** : Late exponential  
        - **D5→D7** : Stationary / decline phase  
        - **D1→D7** : Overall average over the whole experiment  
    - **Error bars (vertical lines)** : Standard deviation of 3 replicate wells. Shorter bars = better reproducibility.
    - **Black bold line = Control (positive control)** : Your experimental reference (untreated or standard strain).
        - Lines **above the black line** = grew better / faster than control.
        - Lines **below the black line** = grew worse (possibly inhibited).
    """)

st.markdown("Recalculated from **raw absorbance data** and **position mapping**. Values are **blank‑subtracted**, shown as **Mean ± SD (n=3)**.")

# --- Sidebar ---
st.sidebar.header("🎛️ Control Panel")

st.sidebar.subheader("1. Select Samples to Display")
default_selection = ["Control", "W1", "W4", "W10"]
selected_samples = st.sidebar.multiselect(
    "Choose samples (multiple allowed)",
    options=samples,
    default=[s for s in default_selection if s in samples]
)

st.sidebar.subheader(f"2. Smart Filter (Reference = {CONTROL_SAMPLE})")
filter_type = st.sidebar.radio(
    "Filter condition",
    ["Show all", f"Only above control (final OD > {CONTROL_SAMPLE})", f"Only below control (final OD < {CONTROL_SAMPLE})"]
)

st.sidebar.subheader("3. Growth Rate Filter (based on Day1→Day3)")
growth_rate_threshold = st.sidebar.number_input(
    "Minimum growth rate (OD/day) — range: 0.000 – 0.150",
    min_value=0.0,
    max_value=0.15,
    value=0.0,
    step=0.005,
    format="%.3f"
)

st.sidebar.subheader("4. Chart Elements")
show_errorbars = st.sidebar.checkbox("Show error bars (SD)", value=True)
show_smooth = st.sidebar.checkbox("Show smooth trend lines", value=True)
show_markers = st.sidebar.checkbox("Show raw data markers", value=True)

# ==================== 6. 应用筛选 ====================
control_final_od = all_data[CONTROL_SAMPLE][7]['mean']

if filter_type == "Show all":
    filtered = selected_samples.copy()
elif f"Only above control (final OD > {CONTROL_SAMPLE})" in filter_type:
    filtered = [s for s in selected_samples if all_data[s][7]['mean'] > control_final_od]
    if CONTROL_SAMPLE in selected_samples and CONTROL_SAMPLE not in filtered:
        filtered.append(CONTROL_SAMPLE)
else:
    filtered = [s for s in selected_samples if all_data[s][7]['mean'] < control_final_od]
    if CONTROL_SAMPLE in selected_samples and CONTROL_SAMPLE not in filtered:
        filtered.append(CONTROL_SAMPLE)

final_samples = [s for s in filtered if growth_rates[s]['1-3'] >= growth_rate_threshold]

if not final_samples:
    if CONTROL_SAMPLE in selected_samples:
        final_samples = [CONTROL_SAMPLE]
    else:
        st.warning("⚠️ No samples match the current filters. Please adjust your criteria.")
        final_samples = [CONTROL_SAMPLE]

final_samples = list(dict.fromkeys(final_samples))
final_samples.sort(key=natural_sort_key)

# ==================== 7. 绘图 ====================
st.subheader("📈 Growth Curves (Interactive Plotly)")
fig = go.Figure()

for sample in final_samples:
    means = [all_data[sample][d]['mean'] for d in days]
    sds = [all_data[sample][d]['sd'] for d in days]
    
    is_control = (sample == CONTROL_SAMPLE)
    line_color = 'black' if is_control else None
    line_width = 4 if is_control else 2
    marker_size = 12 if is_control else 8
    display_name = f"{sample} (Control)" if is_control else sample
    
    if show_smooth and len(days) >= 3:
        try:
            x_smooth = np.linspace(min(days), max(days), 100)
            spline = make_interp_spline(days, means, k=3)
            y_smooth = spline(x_smooth)
            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth,
                mode='lines',
                name=f'{display_name} (trend)',
                line=dict(width=line_width-1 if line_width>1 else 2, dash='dash', color=line_color),
                opacity=0.6,
                showlegend=True
            ))
        except:
            pass
    
    fig.add_trace(go.Scatter(
        x=days,
        y=means,
        mode='markers+lines' if show_markers else 'lines',
        name=display_name,
        marker=dict(size=marker_size if show_markers else 0, symbol='circle', color=line_color),
        line=dict(width=line_width, color=line_color),
        error_y=dict(
            type='data',
            array=sds,
            visible=show_errorbars,
            color='rgba(0,0,0,0.3)',
            thickness=1.5,
            width=4
        )
    ))

fig.update_layout(
    xaxis_title="Time (days)",
    yaxis_title="Absorbance (OD) – Blank",
    hovermode="x unified",
    legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
    template="plotly_white",
    height=600
)
fig.update_xaxes(tickvals=days, ticktext=[f"Day {d}" for d in days])

st.plotly_chart(fig, use_container_width=True)

# ==================== 8. 数据表格 ====================
st.subheader("📋 Filtered Data Details (Mean ± SD & Growth Rates)")

if final_samples:
    table_data = []
    for sample in final_samples:
        row = {
            "Sample": sample,
            "Day1 Mean": round(all_data[sample][1]['mean'], 4),
            "Day1 SD": round(all_data[sample][1]['sd'], 4),
            "Day3 Mean": round(all_data[sample][3]['mean'], 4),
            "Day3 SD": round(all_data[sample][3]['sd'], 4),
            "Day5 Mean": round(all_data[sample][5]['mean'], 4),
            "Day5 SD": round(all_data[sample][5]['sd'], 4),
            "Day7 Mean": round(all_data[sample][7]['mean'], 4),
            "Day7 SD": round(all_data[sample][7]['sd'], 4),
            "Rate D1→D3": round(growth_rates[sample]['1-3'], 4),
            "Rate D3→D5": round(growth_rates[sample]['3-5'], 4),
            "Rate D5→D7": round(growth_rates[sample]['5-7'], 4),
            "Rate D1→D7": round(growth_rates[sample]['1-7'], 4)
        }
        table_data.append(row)
    df_display = pd.DataFrame(table_data)
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("No data to display")

st.caption("💡 **Tip**: Click on a sample name in the legend to show/hide that line. Combine with sidebar filters to fully resolve overlapping.")