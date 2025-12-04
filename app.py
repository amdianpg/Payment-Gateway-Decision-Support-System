# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO

st.set_page_config(page_title="SAW & TOPSIS (Crisp + Fuzzy) — Stepwise", layout="wide")
st.title("Analisis Perbandingan Metode SAW dan TOPSIS dalam Pemilihan Payment Gateway untuk UMKM Berbasis E-Commerce")
st.caption("Aplikasi ini digunakan untuk melakukan perhitungan dan perbandingan metode SAW (Simple Additive Weighting) dan TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) berdasarkan beberapa kriteria penilaian payment gateway.")


# --------------------------
# Config: alternatives & criteria
# --------------------------
default_alternatives = ["Layanan 1","Layanan 2","Layanan 3","Layanan 4","Layanan 5"]
default_criteria = [
    ("Cost", "Cost"),
    ("Payout Fee", "Cost"),
    ("Settlement", "Benefit"),
    ("Security", "Benefit"),
    ("APIease", "Benefit"),
    ("Features", "Benefit"),
]
criteria_names = [c[0] for c in default_criteria]
criteria_types = [c[1] for c in default_criteria]

# --------------------------
# Sidebar: upload / input mode
# --------------------------
with st.sidebar:
    st.header("Data Input")
    mode = st.radio("Pilih mode input", ["Manual (editable table)", "Upload dataset (CSV/Excel)"])
    st.subheader("Panduan Penggunaan")
    st.markdown("""
**1️⃣ Bobot Kriteria**  
  Isi bobot tiap kriteria. Total tidak harus 1, sistem akan normalisasi.

**2️⃣ Decision Matrix**  
  Pilih input manual atau upload file CSV/XLSX. Isi nilai untuk tiap alternatif.

**3️⃣ SAW**  
  Sistem menghitung normalisasi.

**4️⃣ TOPSIS**  
  Sistem menghitung normalisasi vector.

**5️⃣ Visualisasi**  
  Grafik SAW & TOPSIS tersedia otomatis.
  """)
    
# --------------------------
# Weights input (manual)
# --------------------------
st.subheader("1. Bobot Kriteria")
cols = st.columns(3)
weights_input = []
for i, (name, ctype) in enumerate(default_criteria):
    with cols[i % 3]:
        v = st.number_input(f"{name} ({ctype})", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        weights_input.append(v)
weights = np.array(weights_input, dtype=float)
if weights.sum() == 0:
    st.warning("Total bobot = 0. Silakan isi bobot (mis. 0.2, 0.15, ...).")
else:
    if abs(weights.sum() - 1.0) > 1e-9:
        st.info(f"Bobot dinormalisasi (sebelum: {weights.sum():.3f})")
        weights = weights / weights.sum()
w_df = pd.DataFrame({"Criteria": criteria_names, "Type": criteria_types, "Weight": weights})
st.dataframe(w_df.set_index("Criteria"), use_container_width=True)

# --------------------------
# Decision matrix input / upload
# --------------------------
st.subheader("2. Decision Matrix")
if mode == "Manual (editable table)":
    st.write("Isi nilai untuk tiap alternatif atau upload dataset")
    initial = pd.DataFrame({
        criteria_names[i]: [0 for _ in default_alternatives]
        for i in range(len(criteria_names))
    }, index=default_alternatives)
    df_values = st.data_editor(initial, use_container_width=True)
else:
    uploaded = st.file_uploader("Upload CSV atau Excel", type=["csv","xlsx","xls"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_values = pd.read_csv(uploaded, index_col=0)
            else:
                df_values = pd.read_excel(uploaded, index_col=0)
            st.success("Dataset berhasil diunggah.")
        except Exception as e:
            st.error("Gagal membaca file: " + str(e))
            df_values = pd.DataFrame({
                c: [0 for _ in default_alternatives]
                for c in criteria_names
            }, index=default_alternatives)

    else:
        st.info("Belum ada file diupload. Gunakan mode manual atau upload file.")
        df_values = pd.DataFrame({
            criteria_names[i]: [0 for _ in default_alternatives]
            for i in range(len(criteria_names))
        }, index=default_alternatives)


# convert crisp or parse fuzzy
def parse_cell(val):
        # crisp: ensure numeric
        try:
            return float(val)
        except:
            return 0.0

# build data structures
alternatives = list(df_values.index)
# crisp matrix as DataFrame of floats

# ------------------------------------------------------
# CRISP SAW & TOPSIS step-by-step (if not fuzzy)
# ------------------------------------------------------
st.markdown("---")
st.header("A. METODE SAW")

    
st.subheader("1) Normalisasi")
# Gunakan df_values sebagai crisp input
df_crisp = df_values.copy().astype(float)
norm_saw = df_crisp.copy().astype(float)
for i,c in enumerate(criteria_names):
        if criteria_types[i].lower().startswith("cost"):
            norm_saw[c] = df_crisp[c].min() / df_crisp[c].replace(0, np.nan)
        else:
            denom = df_crisp[c].max()
            norm_saw[c] = df_crisp[c] / (denom if denom!=0 else 1)
st.dataframe(norm_saw.style.format("{:.4f}"), use_container_width=True)

st.subheader("2) Matriks Ternormalisasi Terbobot & Skor")
st.latex(r"r_{ij} = \begin{cases}\dfrac{x_{ij}}{\max_j x_{ij}} & \text{(benefit)} \\[6pt] \dfrac{\min_j x_{ij}}{x_{ij}} & \text{(cost)} \end{cases}")
weighted_saw = norm_saw * weights
st.dataframe(weighted_saw.style.format("{:.4f}"), use_container_width=True)

saw_score = weighted_saw.sum(axis=1)
df_saw_rank = pd.DataFrame({"SAW Score": saw_score})
df_saw_rank["Rank"] = df_saw_rank["SAW Score"].rank(ascending=False, method="dense").astype(int)
df_saw_rank = df_saw_rank.sort_values("Rank")
st.subheader("3) Nilai Preferensi SAW & Ranking")
st.latex(r"V_i = \sum_{j=1}^n w_j \cdot r_{ij}")
st.dataframe(df_saw_rank.style.format({"SAW Score":"{:.4f}"}), use_container_width=True)

    # manual example expander
with st.expander("Contoh perhitungan manual SAW untuk satu alternatif"):
        ex = alternatives[0]
        st.write("Nilai asli:")
        st.write(df_crisp.loc[ex].to_frame().T)
        st.write("Nilai normalisasi:")
        st.write(norm_saw.loc[ex].to_frame().T.style.format("{:.4f}"))
        st.write("Pembobotan:")
        st.write((norm_saw.loc[ex] * weights).to_frame().T.style.format("{:.4f}"))
        st.write(f"Skor akhir V_i = {saw_score.loc[ex]:.4f}")

    # TOPSIS
st.markdown("---")
st.header("B. METODE TOPSIS")

st.subheader("1) Normalisasi")
st.latex(r"r_{ij} = \dfrac{x_{ij}}{\sqrt{\sum_{i=1}^m x_{ij}^2}}")
denom = np.sqrt((df_crisp**2).sum(axis=0))
denom = denom.replace(0, np.nan)
norm_topsis = df_crisp / denom
st.dataframe(norm_topsis.style.format("{:.4f}"), use_container_width=True)

st.subheader("2) Matriks Ternormalisasi Terbobot")
st.latex(r"y_{ij} = w_j \cdot r_{ij}")
weighted_topsis = norm_topsis * weights
st.dataframe(weighted_topsis.style.format("{:.4f}"), use_container_width=True)

st.subheader("3) Solusi ideal & jarak")
st.latex(r"A^+,\ A^- \rightarrow S_i^+,\ S_i^- \rightarrow C_i = \dfrac{S_i^-}{S_i^- + S_i^+}")
ideal_pos = {}
ideal_neg = {}
for i,c in enumerate(criteria_names):
        if criteria_types[i].lower().startswith("benefit"):
            ideal_pos[c] = weighted_topsis[c].max()
            ideal_neg[c] = weighted_topsis[c].min()
        else:
            ideal_pos[c] = weighted_topsis[c].min()
            ideal_neg[c] = weighted_topsis[c].max()
st.dataframe(pd.DataFrame([ideal_pos, ideal_neg], index=["A+","A-"]).style.format("{:.4f}"), use_container_width=True)
dist_pos = np.sqrt(((weighted_topsis - pd.Series(ideal_pos))**2).sum(axis=1))
dist_neg = np.sqrt(((weighted_topsis - pd.Series(ideal_neg))**2).sum(axis=1))
st.dataframe(pd.DataFrame({"D+":dist_pos, "D-":dist_neg}).style.format("{:.4f}"), use_container_width=True)

st.subheader("4) Skor TOPSIS & perankingan")
ci = dist_neg / (dist_pos + dist_neg)
df_topsis_rank = pd.DataFrame({"TOPSIS Score": ci})
df_topsis_rank["TOPSIS Score"] = (
df_topsis_rank["TOPSIS Score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
df_topsis_rank["Rank"] = df_topsis_rank["TOPSIS Score"].rank(ascending=False, method="dense").astype(int)
df_topsis_rank = df_topsis_rank.sort_values("Rank")
st.dataframe(df_topsis_rank.style.format({"TOPSIS Score":"{:.4f}"}), use_container_width=True)

with st.expander("Contoh perhitungan manual TOPSIS untuk satu alternatif"):
        ex = alternatives[0]
        st.write("Nilai asli:")
        st.write(df_crisp.loc[ex].to_frame().T)
        st.write("Normalisasi:")
        st.write(norm_topsis.loc[ex].to_frame().T.style.format("{:.4f}"))
        st.write("Pembobotan:")
        st.write(weighted_topsis.loc[ex].to_frame().T.style.format("{:.4f}"))
        st.write(f"D+ = {dist_pos.loc[ex]:.4f}, D- = {dist_neg.loc[ex]:.4f}")
        st.write(f"C_i = {ci.loc[ex]:.4f}")

    # comparison visuals
st.markdown("---")
st.header("Perbandingan")
comp = pd.concat([df_saw_rank["SAW Score"], df_topsis_rank["TOPSIS Score"]], axis=1).fillna(0)
comp = comp.reset_index(names="Alternative")
comp = comp.melt(id_vars="Alternative", var_name="Method", value_name="Score")
bar = alt.Chart(comp).mark_bar().encode(x='Alternative:N', y='Score:Q', color='Method:N', tooltip=['Alternative','Method','Score']).properties(height=350)
st.altair_chart(bar, use_container_width=True)

# --------------------------
# Perbandingan
# --------------------------
st.markdown("---")
st.header("Perbandingan SAW & TOPSIS")
comp = pd.concat([df_saw_rank["SAW Score"], df_topsis_rank["TOPSIS Score"]], axis=1).fillna(0)
comp = comp.reset_index(names="Alternative")
comp = comp.melt(id_vars="Alternative", var_name="Method", value_name="Score")
bar = alt.Chart(comp).mark_bar().encode(
    x='Alternative:N', 
    y='Score:Q', 
    color='Method:N', 
    tooltip=['Alternative','Method','Score']
).properties(height=350)
st.altair_chart(bar, use_container_width=True)

# --------------------------
# Analisis Kecocokan
# --------------------------
st.markdown("---")
st.header("Analisis Kecocokan SAW vs TOPSIS (tanpa scipy)")

# Ranking
saw_rank = df_saw_rank["Rank"]
topsis_rank = df_topsis_rank["Rank"]

# Selisih ranking kuadrat
d_squared = (saw_rank - topsis_rank)**2
n = len(saw_rank)

# Spearman Rank Correlation manual
spearman_corr = 1 - (6 * d_squared.sum()) / (n * (n**2 - 1))
st.write(f"Spearman Rank Correlation (manual): {spearman_corr:.4f}")

# Selisih ranking per alternatif
df_diff = pd.DataFrame({
    "SAW Rank": saw_rank,
    "TOPSIS Rank": topsis_rank,
    "Selisih Rank": (saw_rank - topsis_rank).abs()
})
st.subheader("Selisih Ranking per Alternatif")
st.dataframe(df_diff.style.format({"Selisih Rank":"{:.0f}"}), use_container_width=True)

# Visualisasi perbandingan ranking
df_diff_vis = df_diff.reset_index().melt(id_vars="index", value_vars=["SAW Rank","TOPSIS Rank"], 
                                        var_name="Method", value_name="Rank")
bar_corr = alt.Chart(df_diff_vis).mark_bar().encode(
    x='index:N',
    y='Rank:Q',
    color='Method:N',
    tooltip=['index','Method','Rank']
).properties(height=350)
st.subheader("Visualisasi Perbandingan Ranking")
st.altair_chart(bar_corr, use_container_width=True)
