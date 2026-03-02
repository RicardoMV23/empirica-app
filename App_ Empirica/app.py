"""
================================================================================
Empirica App: Plataforma Base + 3 Pasos + QA/QC + Paso 2 Control Manual (Corregido)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import norm
from scipy.spatial import cKDTree
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib
matplotlib.use('Agg') 
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURACIÓN Y ESTILOS
# ============================================================================

st.set_page_config(
    page_title="Empirica App",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'df_principal' not in st.session_state: st.session_state.df_principal = None
if 'df_filtrado' not in st.session_state: st.session_state.df_filtrado = None
if 'cols' not in st.session_state: st.session_state.cols = None
if 'var_actual' not in st.session_state: st.session_state.var_actual = None
if 'cat_actual' not in st.session_state: st.session_state.cat_actual = None

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1c24; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .main-header { font-family: 'Helvetica Neue', sans-serif; font-size: 2.2rem; font-weight: 700; color: #1a1c24; border-bottom: 3px solid #FF8C00; padding-bottom: 10px; margin-bottom: 20px; }
    .metric-card { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-val { font-size: 1.8rem; font-weight: bold; color: #007BFF; }
    .metric-lbl { font-size: 0.9rem; color: #6c757d; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. FUNCIONES DE PROCESAMIENTO BASE Y PLOTEO
# ============================================================================

def reiniciar_app():
    st.cache_data.clear()
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

def limpiar_formato_numerico(df):
    df.columns = df.columns.str.strip()
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                cleaned = df_clean[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                df_clean[col] = pd.to_numeric(cleaned)
            except: pass
    return df_clean

def detectar_columnas(df):
    cols = {'coords': {}, 'vars': [], 'cats': []}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ['holeid', 'sondaje', 'id', 'compid'] and 'id' not in cols: cols['id'] = c
        elif any(x in cl for x in ['midx', 'este', 'east', 'x']) and 'x' not in cols['coords']: cols['coords']['x'] = c
        elif any(y in cl for y in ['midy', 'norte', 'north', 'y']) and 'y' not in cols['coords']: cols['coords']['y'] = c
        elif any(z in cl for z in ['midz', 'cota', 'elev', 'z']) and 'z' not in cols['coords']: cols['coords']['z'] = c
        elif any(k in cl for k in ['lito', 'alt', 'rock', 'unit', 'geo', 'zone', 'dom']): cols['cats'].append(c)
        elif df[c].dtype in ['float64', 'int64']: cols['vars'].append(c)
    if not cols['vars']: cols['vars'] = list(df.select_dtypes(include=[np.number]).columns)
    return cols

@st.cache_data
def cargar_archivo(file):
    if file.name.endswith('.csv'):
        try: 
            file.seek(0)
            df = pd.read_csv(file, sep=';')
        except: 
            file.seek(0)
            df = pd.read_csv(file, sep=',')
    else:
        df = pd.read_excel(file)
    df = df.loc[:, ~df.columns.duplicated()]
    return limpiar_formato_numerico(df)

def tarjeta_metrica(label, value, col):
    col.markdown(f"""<div class="metric-card"><div class="metric-val">{value}</div><div class="metric-lbl">{label}</div></div>""", unsafe_allow_html=True)

def obtener_df_activo():
    if st.session_state.df_filtrado is not None: return st.session_state.df_filtrado, "Filtrado"
    elif st.session_state.df_principal is not None: return st.session_state.df_principal, "Original"
    else: return None, None

def generar_boxplot_reporte(df, col_ug, col_var):
    valid_df = df.dropna(subset=[col_ug, col_var]).copy()
    valid_df = valid_df[~valid_df[col_ug].astype(str).str.lower().str.contains('nan', na=False)]
    valid_df = valid_df[~valid_df[col_ug].astype(str).str.lower().isin(['none', ''])]
    
    ugs = sorted(valid_df[col_ug].unique())
    datos = [valid_df[valid_df[col_ug] == ug][col_var].values for ug in ugs]
    if not datos:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No hay datos clasificados válidos.", ha='center')
        return fig

    fig, ax = plt.subplots(figsize=(14, 8))
    box = ax.boxplot(datos, patch_artist=True, showmeans=True, widths=0.5,
                     meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6},
                     medianprops={"color":"#FF8C00", "linewidth":2},
                     flierprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"#555", "markersize":5})
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F1C40F', '#E67E22', '#1ABC9C', '#34495E'] * (len(ugs) // 8 + 1)
    for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, ug in enumerate(ugs):
        series = valid_df[valid_df[col_ug] == ug][col_var]
        if len(series) > 0:
            q1, med, q3 = series.quantile(0.25), series.median(), series.quantile(0.75)
            props = dict(boxstyle='round', alpha=0.9, edgecolor='none')
            ax.text(i + 1.32, q3, f'Q3:{q3:.1f}', fontsize=8, va='center', bbox=dict(facecolor='#D6EAF8', **props))
            ax.text(i + 1.32, med, f'Med:{med:.1f}', fontsize=8, va='center', bbox=dict(facecolor='#D5F5E3', **props))
            ax.text(i + 1.32, q1, f'Q1:{q1:.1f}', fontsize=8, va='center', bbox=dict(facecolor='#FCF3CF', **props))
    ax.set_xticklabels([str(u) for u in ugs], fontsize=10, fontweight='bold', rotation=45)
    ax.set_ylabel(f'{col_var}', fontweight='bold')
    ax.set_title(f'Distribución por {col_ug} - {col_var}', fontweight='bold', fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    return fig

PROB_TICKS=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]

def logprob_points(dfv, domcol, title, var_name):
    fig, ax = plt.subplots(figsize=(12, 8))
    for d in sorted(dfv[domcol].dropna().unique()):
        vals = pd.to_numeric(dfv.loc[dfv[domcol]==d, var_name], errors="coerce").dropna().values
        vals = vals[vals > 0]
        if len(vals) < 5: continue
        x = np.sort(vals)
        p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
        y = norm.ppf(p)
        ax.scatter(x, y, s=18, alpha=0.85, label=f"{domcol} {int(d)} (n={len(x)})")
    ax.set_xscale("log")
    ax.set_xlabel(f"{var_name} (escala log)")
    ax.set_ylabel("Quantiles Normal (probabilidad)")
    ax.set_title(title, fontweight='bold')
    zticks = norm.ppf(PROB_TICKS)
    labels = [f"{p*100:.1f}%" if p < 0.01 else f"{p*100:.0f}%" for p in PROB_TICKS]
    ax.set_yticks(zticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(norm.ppf(0.001), norm.ppf(0.999))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    return fig

def effect_prop(stats, domcol, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(stats["mean"], stats["std"], s=90, color='#3498DB', edgecolor='black')
    for _, r in stats.iterrows():
        ax.text(r["mean"], r["std"], f"{domcol} {int(r[domcol])}\n(n={int(r['count'])})", fontsize=9)
    ax.set_xlabel("Media")
    ax.set_ylabel("Desv. estándar")
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def scatter_dom_fig(dfv, x_col, y_col, domcol, title):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for d in sorted(dfv[domcol].dropna().unique()):
        m = dfv[domcol] == d
        ax.scatter(dfv.loc[m, x_col], dfv.loc[m, y_col], s=7, alpha=0.7, label=f"{domcol} {int(d)}")
    ax.set_xlabel(x_col.upper(), fontweight='bold')
    ax.set_ylabel(y_col.upper(), fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def to_excel(df_stats):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_stats.to_excel(writer, index=False, sheet_name='Stats_por_DOM')
    return output.getvalue()

# ============================================================================
# 3. BARRA LATERAL
# ============================================================================

try: st.sidebar.image("Logo.png", use_container_width=True)
except: st.sidebar.markdown("### ⚒️ EMPIRICA")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navegación")

menu = st.sidebar.radio("Ir a:", [
    "📂 Cargar Datos", 
    "🔍 QA/QC & Filtros", 
    "📊 EDA & Análisis", 
    "🌳 Modelamiento (Paso 1: DOM1 Espacial)", 
    "🌳 Modelamiento (Paso 2: DOM2 Geológico)",
    "🌳 Modelamiento (Paso 3: DOM3 Definitivo)",
    "🔷 Secciones"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Variable de Estudio Global")

if st.session_state.cols is not None:
    cols_data = st.session_state.cols
    var_options = cols_data['vars']
    default_idx = 0
    for i, v in enumerate(var_options):
        if 'rec' in v.lower() or 'cut' in v.lower() or 'cu' in v.lower():
            default_idx = i; break
            
    sel_var = st.sidebar.selectbox("Selecciona Variable para EDA", var_options, index=default_idx)
    st.session_state.var_actual = sel_var
else:
    st.sidebar.info("Carga datos para ver selectores.")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Resetear", type="secondary"): reiniciar_app()

# ============================================================================
# 4. PÁGINAS PRINCIPALES
# ============================================================================

if menu == "📂 Cargar Datos":
    st.markdown("<div class='main-header'>📂 Cargar Base de Datos</div>", unsafe_allow_html=True)
    if st.session_state.df_principal is not None:
        st.success(f"✅ Archivo activo: {len(st.session_state.df_principal):,} registros.")
    else:
        upload = st.file_uploader("Sube DataPRM.csv o Excel", type=['csv', 'xlsx'])
        if upload:
            df = cargar_archivo(upload)
            st.session_state.df_principal = df
            st.session_state.cols = detectar_columnas(df)
            st.success("Carga exitosa. Columnas detectadas.")
            st.rerun()
    if st.session_state.df_principal is not None:
        st.dataframe(st.session_state.df_principal.head())

elif menu == "🔍 QA/QC & Filtros":
    st.markdown("<div class='main-header'>🔍 QA/QC y Depuración</div>", unsafe_allow_html=True)
    
    if st.session_state.df_principal is not None:
        df = st.session_state.df_principal
        cols = st.session_state.cols
        var = st.session_state.var_actual 
        
        st.markdown(f"#### Validación Variable: **{var}**")
        c1, c2, c3, c4 = st.columns(4)
        tarjeta_metrica("Total Registros", len(df), c1)
        tarjeta_metrica("Nulos (NaN)", df[var].isnull().sum(), c2)
        tarjeta_metrica("Ceros/Negativos (<= 0)", (df[var]<=0).sum(), c3)
        tarjeta_metrica("Media Original", f"{df[var].mean():.2f}", c4)
        
        c_ctrl, c_rep = st.columns([1, 2])
        with c_ctrl:
            st.markdown("##### Reglas de Limpieza")
            st.info("Estas reglas se aplicarán directamente a la variable objetivo.")
            
            drop_na = st.checkbox("Eliminar valores Nulos (NaN)", value=True)
            drop_zeros = st.checkbox("Eliminar valores Menores o Iguales a 0", value=True)
            
            st.markdown("##### Rango Manual (Opcional)")
            min_v = st.number_input("Mínimo Absoluto", value=float(df[var].min()))
            max_v = st.number_input("Máximo Absoluto", value=float(df[var].max()))
            
            if st.button("🚀 APLICAR FILTROS", type="primary"):
                df_t = df.copy()
                if drop_na: df_t = df_t.dropna(subset=[var])
                if drop_zeros: df_t = df_t[df_t[var] > 0]
                df_t = df_t[(df_t[var] >= min_v) & (df_t[var] <= max_v)]
                
                st.session_state.df_filtrado = df_t
                if 'df_ug' in st.session_state: del st.session_state.df_ug
                if 'modelo_paso1' in st.session_state: del st.session_state.modelo_paso1
                if 'modelo_paso2' in st.session_state: del st.session_state.modelo_paso2
                st.success(f"✅ Filtro aplicado con éxito. Datos listos: {len(df_t)} registros.")
                
        with c_rep:
            if st.session_state.df_filtrado is not None:
                fig_qc, ax_qc = plt.subplots(figsize=(8, 4))
                sns.histplot(st.session_state.df_filtrado[var], kde=True, color='#2ECC71', edgecolor='black', ax=ax_qc)
                ax_qc.set_title(f"Distribución Post-Filtro: {var}", fontweight='bold')
                st.pyplot(fig_qc)
            else:
                st.warning("👈 Haz clic en 'APLICAR FILTROS' para limpiar tu base de datos.")
    else:
        st.error("⚠️ Sube un archivo primero en la pestaña 'Cargar Datos'.")

elif menu == "📊 EDA & Análisis":
    st.markdown("<div class='main-header'>📊 EDA Avanzado</div>", unsafe_allow_html=True)
    df_act, _ = obtener_df_activo()
    if df_act is not None:
        cols = st.session_state.cols
        var = st.session_state.var_actual
        data_series = df_act[var].dropna()
        tab_uni, tab_3d = st.tabs(["📉 Univariado", "📍 Mapa 3D"])
        
        with tab_uni:
            c1, c2 = st.columns([1, 1.5]) 
            mean_val, median_val = data_series.mean(), data_series.median()
            q1_val, q3_val = data_series.quantile(0.25), data_series.quantile(0.75)
            
            with c1:
                fig_box, ax_box = plt.subplots(figsize=(6, 8))
                sns.boxplot(y=data_series, color='#5DADE2', width=0.4, ax=ax_box, showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6}, flierprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"#555", "markersize":5})
                props = dict(boxstyle='round', alpha=0.9, edgecolor='none')
                ax_box.text(0.35, q3_val, f'Q3: {q3_val:.2f}', fontsize=10, va='center', bbox=dict(facecolor='#D6EAF8', **props))
                ax_box.text(0.35, median_val, f'Med: {median_val:.2f}', fontsize=10, va='center', bbox=dict(facecolor='#D5F5E3', **props))
                ax_box.text(0.35, mean_val, f'Mean: {mean_val:.2f}', fontsize=10, va='center', bbox=dict(facecolor='#FADBD8', **props))
                ax_box.text(0.35, q1_val, f'Q1: {q1_val:.2f}', fontsize=10, va='center', bbox=dict(facecolor='#FCF3CF', **props))
                ax_box.set_title(f"Boxplot: {var}", fontweight='bold', fontsize=12)
                ax_box.grid(True, linestyle='--', alpha=0.6, axis='y')
                st.pyplot(fig_box)
                st.dataframe(data_series.describe().to_frame().T.style.format("{:.2f}"))
            
            with c2:
                fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                sns.histplot(data_series, kde=True, color='#5DADE2', edgecolor='black', ax=ax_hist)
                ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
                ax_hist.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Mediana: {median_val:.2f}')
                ax_hist.axvline(q1_val, color='orange', linestyle='-.', linewidth=2, label=f'Q1: {q1_val:.2f}')
                ax_hist.axvline(q3_val, color='purple', linestyle='-.', linewidth=2, label=f'Q3: {q3_val:.2f}')
                ax_hist.set_title(f"Distribución: {var}", fontweight='bold', fontsize=12)
                ax_hist.legend(); st.pyplot(fig_hist)
                
        with tab_3d:
            fig3d = px.scatter_3d(df_act.dropna(subset=[cols['coords']['x'], var]), x=cols['coords']['x'], y=cols['coords']['y'], z=cols['coords']['z'], color=var, opacity=0.7, color_continuous_scale='Jet')
            fig3d.update_traces(marker=dict(size=3)); st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------------------------------------------------------------
# PASO 1: DOM1 ESPACIAL 
# ---------------------------------------------------------------------------------
elif menu == "🌳 Model
