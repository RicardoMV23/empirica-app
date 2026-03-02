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
elif menu == "🌳 Modelamiento (Paso 1: DOM1 Espacial)":
    st.markdown("<div class='main-header'>🌳 Modelamiento (PASO 1: DOM1 Espacial)</div>", unsafe_allow_html=True)
    df_act, _ = obtener_df_activo()
    if df_act is not None:
        cols = st.session_state.cols
        var_global = st.session_state.var_actual
        
        st.markdown("### 1. Selección de Variables")
        c_var1, c_var2 = st.columns(2)
        with c_var1:
            idx_obj = cols['vars'].index(var_global) if var_global in cols['vars'] else 0
            target_var = st.selectbox("🎯 Variable Objetivo (Y):", options=cols['vars'], index=idx_obj)
        with c_var2:
            default_coords = [c for c in [cols['coords'].get('x'), cols['coords'].get('y'), cols['coords'].get('z')] if c]
            predictor_vars = st.multiselect("📏 Variables Espaciales (X):", options=cols['vars'] + default_coords, default=default_coords)

        st.markdown("### 2. Parámetros del Árbol")
        c1, c2, c3 = st.columns(3)
        with c1: d1_depth = st.slider("Profundidad (max_depth)", 1, 10, 3) 
        with c2: d1_leaves = st.slider("Límite de Nodos (max_leaf_nodes)", 2, 20, 4) 
        with c3: st.info("ℹ️ min_samples_leaf calculado internamente: max(ceil(0.02*N), 5)")

        if st.button("🚀 EJECUTAR PASO 1", type="primary"):
            if not predictor_vars:
                st.error("Debes seleccionar al menos una Variable Espacial (X).")
            else:
                col_x, col_y, col_z = cols['coords']['x'], cols['coords']['y'], cols['coords']['z']
                df_f = df_act.loc[df_act[target_var].notna() & (df_act[target_var] >= 0) & 
                                  df_act[col_x].notna() & df_act[col_y].notna() & df_act[col_z].notna()].copy()
                
                X_data = df_f[predictor_vars].to_numpy(float)
                Y_target = df_f[target_var].to_numpy(float)
                
                MIN_LEAF_FRAC = 0.02
                MIN_LEAF_ABS = 5
                min_leaf = max(int(np.ceil(MIN_LEAF_FRAC * len(df_f))), MIN_LEAF_ABS)
                
                reg = DecisionTreeRegressor(max_depth=d1_depth, max_leaf_nodes=d1_leaves, random_state=42, min_samples_leaf=min_leaf)
                reg.fit(X_data, Y_target)
                leaf = reg.apply(X_data)
                
                counts = pd.Series(leaf).value_counts().sort_values(ascending=False)
                leaf_map = {leaf_id: i+1 for i, leaf_id in enumerate(counts.index.tolist())}
                dom_arr = np.array([leaf_map[l] for l in leaf], dtype=int)
                
                df_act.loc[df_f.index, 'DOM1_num'] = dom_arr
                df_act.loc[df_f.index, 'UG_Activa'] = [f"DOM1 {x}" for x in dom_arr]
                
                stats = df_f.assign(DOM=dom_arr).groupby("DOM")[target_var].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index()
                
                st.session_state.df_ug = df_act
                st.session_state.modelo_paso1 = reg
                st.session_state.stats_paso1 = stats
                st.session_state.paso1_target = target_var
                st.session_state.paso1_preds = predictor_vars
                st.success(f"✅ Paso 1 ejecutado. Se definieron {len(set(dom_arr))} dominios espaciales.")

        if 'df_ug' in st.session_state and 'modelo_paso1' in st.session_state:
            df_ug = st.session_state.df_ug
            stats = st.session_state.stats_paso1
            t_target = st.session_state.paso1_target
            
            t1, t2, t3, t4, t5, t6 = st.tabs(["📄 Exportar", "🌳 Árbol", "📈 Log-Prob", "📊 Efecto Prop", "📍 Vistas 2D", "📦 Boxplots"])
            with t1:
                c_out1, c_out2 = st.columns(2)
                with c_out1:
                    st.dataframe(stats)
                    st.download_button("📥 Descargar Excel", to_excel(stats), "Resumen_DOM1.xlsx")
                with c_out2:
                    st.download_button("📥 Descargar CSV", df_ug.to_csv(index=False, sep=";").encode('utf-8'), "DataPRM_DOM1.csv", "text/csv")
            with t2:
                st.markdown("### Árbol de Regresión (DOM1)")
                fig_t = plt.figure(figsize=(25, 12)) 
                plot_tree(st.session_state.modelo_paso1, filled=True, feature_names=st.session_state.paso1_preds, max_depth=st.session_state.modelo_paso1.max_depth, fontsize=12) 
                st.pyplot(fig_t)
                
                buf_p1 = io.BytesIO()
                fig_t.savefig(buf_p1, format="png", dpi=300, bbox_inches="tight")
                st.download_button("📥 Descargar Imagen del Árbol (Alta Resolución)", buf_p1.getvalue(), "Arbol_Paso1_HD.png", "image/png")
            
            with t3:
                df_valid = df_ug.dropna(subset=[t_target, 'DOM1_num']).copy()
                fig_log = logprob_points(df_valid, 'DOM1_num', f"Log-Probabilístico {t_target} — DOM1", t_target)
                st.pyplot(fig_log)
            with t4:
                fig_ef = effect_prop(stats, "DOM", f"Efecto proporcional — DOM1")
                st.pyplot(fig_ef)
            with t5:
                c_2d1, c_2d2, c_2d3 = st.columns(3)
                df_valid = df_ug.dropna(subset=['DOM1_num', cols['coords']['x'], cols['coords']['y'], cols['coords']['z']]).copy()
                with c_2d1: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['x'], cols['coords']['y'], 'DOM1_num', "DOM1 — Vista XY"))
                with c_2d2: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['x'], cols['coords']['z'], 'DOM1_num', "DOM1 — Vista XZ"))
                with c_2d3: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['y'], cols['coords']['z'], 'DOM1_num', "DOM1 — Vista YZ"))
            with t6:
                fig_box = generar_boxplot_reporte(df_ug, 'UG_Activa', t_target)
                st.pyplot(fig_box)

# ---------------------------------------------------------------------------------
# PASO 2: DOM2 GEOLÓGICO (AHORA CON SLIDERS MANUALES Y TODOS LOS GRÁFICOS CORREGIDO)
# ---------------------------------------------------------------------------------
elif menu == "🌳 Modelamiento (Paso 2: DOM2 Geológico)":
    st.markdown("<div class='main-header'>🌳 Modelamiento (PASO 2: DOM2 Geológico)</div>", unsafe_allow_html=True)
    df_act, _ = obtener_df_activo()
    
    if df_act is not None:
        if 'DOM1_num' not in df_act.columns:
            st.warning("⚠️ Debes ejecutar el Paso 1 primero. DOM2 requiere la validación de filas filtradas para cruzar con DOM1 en el futuro.")
            
        cols = st.session_state.cols
        var_global = st.session_state.var_actual
        
        # Mantenemos la estructura de variables del script Python
        VARS17 = ['alt', 'bn', 'cpy', 'cs', 'cv', 'cus', 'cut', 'dens', 'enar', 'esf', 'fet', 'lito', 'mo', 'pb', 'py', 'sb', 'zn']
        CAT_VARS_SENIOR = ["lito", "alt"]
        
        colmap = {c.lower(): c for c in df_act.columns}
        present = [v for v in VARS17 if v in colmap]
        avail_vars = [colmap[v] for v in present]
        
        st.markdown("### 1. Selección de Variables")
        c1, c2 = st.columns(2)
        with c1: 
            idx_obj = cols['vars'].index(var_global) if var_global in cols['vars'] else 0
            target_var = st.selectbox("🎯 Variable a discretizar en cuartiles (Y):", options=cols['vars'], index=idx_obj)
        with c2: 
            predictor_vars = st.multiselect("⚙️ Variables Predictoras (X):", options=avail_vars, default=avail_vars)

        st.markdown("### 2. Parámetros del Árbol (Control Manual)")
        st.info("💡 A diferencia del script automático, aquí tú defines exactamente hasta dónde quieres que crezca el árbol geológico.")
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1: 
            d2_depth = st.slider("Profundidad (max_depth)", 1, 20, 5) 
        with c_p2: 
            d2_leaves = st.slider("Límite de Nodos (max_leaf_nodes)", 2, 50, 15) 
        with c_p3: 
            st.info("ℹ️ Límite de muestras (min_samples_leaf) fijado internamente en 1.")

        if st.button("🚀 EJECUTAR PASO 2", type="primary"):
            if not predictor_vars:
                st.error("Debes seleccionar al menos una variable predictora.")
            else:
                predictor_vars = [c for c in VARS17 if c in predictor_vars]
                
                col_x, col_y, col_z = cols['coords']['x'], cols['coords']['y'], cols['coords']['z']
                valid_mask = df_act[target_var].notna() & (df_act[target_var] >= 0) & df_act[col_x].notna() & df_act[col_y].notna() & df_act[col_z].notna()
                w = df_act.loc[valid_mask].copy()
                
                num_vars_lower = [v for v in present if v not in CAT_VARS_SENIOR]
                cat_vars_lower = [v for v in present if v in CAT_VARS_SENIOR]
                
                num_vars = [colmap[v] for v in num_vars_lower if colmap[v] in predictor_vars]
                cat_vars = [colmap[v] for v in cat_vars_lower if colmap[v] in predictor_vars]
                
                # Preprocesamiento: Imputación + One Hot Encoding
                Xdf = w[num_vars].copy()
                for c in num_vars:
                    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
                    Xdf[c] = Xdf[c].fillna(Xdf[c].median()) 
                
                if len(cat_vars) > 0:
                    for c in cat_vars:
                        w[c] = w[c].fillna("MISSING").astype(str)
                    Xcat = pd.get_dummies(w[cat_vars], prefix=cat_vars, dummy_na=False)
                    Xdf = pd.concat([Xdf, Xcat], axis=1)
                
                feat_names = Xdf.columns.tolist()
                X_data = Xdf.values.astype(float)
                
                # Cuartiles
                y_raw = w[target_var].values.astype(float)
                try: y_class = pd.qcut(pd.Series(y_raw), q=4, labels=[0,1,2,3]).values.astype(int)
                except ValueError: y_class = pd.qcut(pd.Series(y_raw).rank(method="average"), q=4, labels=[0,1,2,3]).values.astype(int)
                
                # Entrenamiento manual respetando los sliders
                clf = DecisionTreeClassifier(max_depth=d2_depth, max_leaf_nodes=d2_leaves, min_samples_leaf=1, random_state=42)
                clf.fit(X_data, y_class)
                
                # Asignación de hojas (DOM2)
                leaf = clf.apply(X_data)
                counts = pd.Series(leaf).value_counts().sort_values(ascending=False)
                leaf_map = {leaf_id: i+1 for i, leaf_id in enumerate(counts.index.tolist())}
                dom2_arr = np.array([leaf_map[l] for l in leaf], dtype=int)
                
                df_act.loc[w.index, 'DOM2_num'] = dom2_arr
                df_act.loc[w.index, 'UG_Activa'] = [f"DOM2 {x}" for x in dom2_arr]
                
                # Generamos las estadísticas para el DOM2 necesarias para el Efecto Proporcional
                stats_p2 = df_act.loc[w.index].groupby('DOM2_num')[target_var].agg(['count', 'mean', 'std', 'min', 'median', 'max']).reset_index().rename(columns={'DOM2_num': 'DOM'})
                
                st.session_state.df_ug = df_act
                st.session_state.modelo_paso2 = clf
                st.session_state.paso2_target = target_var
                st.session_state.paso2_feats = feat_names
                st.session_state.paso2_stats = stats_p2
                st.session_state.paso2_metrics = {
                    "max_leaf": d2_leaves, 
                    "min_leaf": 1, 
                    "hojas": len(set(dom2_arr)) # CORRECCIÓN VITAL PARA EVITAR EL KEYERROR
                }
                
                st.success(f"✅ Paso 2 ejecutado manualmente. Se definieron {st.session_state.paso2_metrics['hojas']} dominios geológicos.")

        if 'df_ug' in st.session_state and 'modelo_paso2' in st.session_state:
            df_ug = st.session_state.df_ug
            t_target = st.session_state.paso2_target
            metrics = st.session_state.paso2_metrics
            stats_p2 = st.session_state.paso2_stats
            
            # Se añadieron todas las pestañas de validación
            t1, t2, t3, t4, t5, t6 = st.tabs(["📄 Exportar", "🌳 Árbol", "📈 Log-Prob", "📊 Efecto Prop", "📍 Vistas 2D", "📦 Boxplots"])
            
            with t1:
                c_out1, c_out2 = st.columns(2)
                with c_out1:
                    st.dataframe(stats_p2)
                    st.download_button("📥 Descargar Excel DOM2", to_excel(stats_p2), "Resumen_DOM2.xlsx")
                with c_out2:
                    st.download_button("📥 Descargar CSV (Data + DOM2)", df_ug.to_csv(index=False, sep=";").encode('utf-8'), "DataPRM_Paso2.csv", "text/csv")
            with t2:
                st.markdown("### Árbol de Clasificación Óptimo (DOM2)")
                fig_t2, ax_t2 = plt.subplots(figsize=(40, 18))
                plot_tree(st.session_state.modelo_paso2, feature_names=st.session_state.paso2_feats, class_names=["Q1", "Q2", "Q3", "Q4"], filled=True, rounded=True, proportion=True, fontsize=12, max_depth=3, impurity=False, ax=ax_t2)
                ax_t2.set_title(f"Árbol clasificación {t_target} (cuartiles) | max_leaf={metrics['max_leaf']} min_leaf={metrics['min_leaf']} | hojas={metrics['hojas']}", fontsize=18, fontweight='bold')
                st.pyplot(fig_t2)
                
                buf_p2 = io.BytesIO()
                fig_t2.savefig(buf_p2, format="png", dpi=300, bbox_inches="tight")
                st.download_button("📥 Descargar Imagen del Árbol (Alta Resolución)", buf_p2.getvalue(), "Arbol_Paso2_HD.png", "image/png")

            with t3:
                df_valid = df_ug.dropna(subset=[t_target, 'DOM2_num']).copy()
                fig_log = logprob_points(df_valid, 'DOM2_num', f"Log-Probabilístico {t_target} — DOM2", t_target)
                st.pyplot(fig_log)
                
            with t4:
                fig_ef = effect_prop(stats_p2, "DOM", f"Efecto proporcional — DOM2")
                st.pyplot(fig_ef)
                
            with t5:
                c_2d1, c_2d2, c_2d3 = st.columns(3)
                df_valid = df_ug.dropna(subset=['DOM2_num', cols['coords']['x'], cols['coords']['y'], cols['coords']['z']]).copy()
                with c_2d1: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['x'], cols['coords']['y'], 'DOM2_num', "DOM2 — Vista XY (Planta)"))
                with c_2d2: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['x'], cols['coords']['z'], 'DOM2_num', "DOM2 — Vista XZ"))
                with c_2d3: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['y'], cols['coords']['z'], 'DOM2_num', "DOM2 — Vista YZ"))

            with t6:
                fig_box = generar_boxplot_reporte(df_ug, 'UG_Activa', t_target)
                st.pyplot(fig_box)

# ---------------------------------------------------------------------------------
# PASO 3: DOM3 DEFINITIVO (BUFFER OP2)
# ---------------------------------------------------------------------------------
elif menu == "🌳 Modelamiento (Paso 3: DOM3 Definitivo)":
    st.markdown("<div class='main-header'>🌳 Modelamiento (PASO 3: DOM3 Definitivo OP2)</div>", unsafe_allow_html=True)
    df_act, _ = obtener_df_activo()
    
    if df_act is not None:
        if 'DOM1_num' not in df_act.columns or 'DOM2_num' not in df_act.columns:
            st.error("⚠️ Para ejecutar el Paso 3, debes correr primero el Paso 1 (Derivas) y el Paso 2 (Clasificación).")
        else:
            cols = st.session_state.cols
            target_var = st.session_state.get('paso2_target', st.session_state.var_actual)
            
            st.info("💡 **Objetivo OP2:** Mantener los dominios base DOM1 (derivas rectas), pero permitir que la variable geometalúrgica (DOM2) module los contactos **SOLO** cerca de los límites (Buffer P20).")
            
            if st.button("🚀 EJECUTAR PASO 3 Y APLICAR BUFFER", type="primary"):
                col_x, col_y, col_z = cols['coords']['x'], cols['coords']['y'], cols['coords']['z']
                
                valid_mask = df_act[target_var].notna() & df_act[col_x].notna() & df_act[col_y].notna() & df_act[col_z].notna() & df_act['DOM1_num'].notna() & df_act['DOM2_num'].notna()
                w = df_act.loc[valid_mask].copy()
                
                coords = w[[col_x, col_y, col_z]].to_numpy(float)
                dom = w['DOM1_num'].to_numpy(int)
                dom2 = w['DOM2_num'].to_numpy(int)
                
                with st.spinner("Calculando fronteras KDTree y aplicando Buffer..."):
                    dom2_to_dom = {}
                    for d2 in np.unique(dom2):
                        m = (dom2 == d2)
                        dom2_to_dom[d2] = int(pd.Series(dom[m]).mode().iloc[0])
                    target_dom = np.array([dom2_to_dom[d2] for d2 in dom2], dtype=int)
                    
                    trees = {}
                    for d in np.unique(dom):
                        trees[d] = cKDTree(coords[dom == d])
                    
                    dist_other = np.full(len(coords), np.inf)
                    for i in range(len(coords)):
                        d = dom[i]
                        best = np.inf
                        for od, tr in trees.items():
                            if od == d: continue
                            dd, _ = tr.query(coords[i], k=1)
                            if dd < best: best = dd
                        dist_other[i] = best
                    
                    buffer = float(np.nanpercentile(dist_other[np.isfinite(dist_other)], 20))
                    
                    dom3 = dom.copy()
                    change = (target_dom != dom) & (dist_other <= buffer)
                    dom3[change] = target_dom[change]
                    
                    df_act.loc[w.index, 'DOM3_num'] = dom3
                    df_act.loc[w.index, 'UG_Activa'] = [f"DOM3 {x}" for x in dom3]
                    
                    st.session_state.df_ug = df_act
                    st.session_state.paso3_target = target_var
                    st.session_state.paso3_stats = df_act.groupby('DOM3_num')[target_var].agg(['count', 'mean', 'std']).reset_index().rename(columns={'DOM3_num': 'DOM'})
                    
                    st.success(f"✅ Buffer (p20) calculado en: {buffer:.2f} metros.")
                    st.info(f"📊 Bloques reasignados en los bordes: {int(change.sum())}")

            if 'df_ug' in st.session_state and 'DOM3_num' in st.session_state.df_ug.columns:
                df_ug = st.session_state.df_ug
                t_target = st.session_state.paso3_target
                stats_p3 = st.session_state.paso3_stats
                
                t1, t2, t3, t4, t5 = st.tabs(["📄 Exportar Final", "📈 Log-Probabilístico", "📊 Efecto Proporcional", "📍 Vistas 2D", "📦 Boxplots"])
                
                with t1:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.dataframe(stats_p3)
                        st.download_button("📥 Descargar Excel DOM3", to_excel(stats_p3), "Resumen_DOM3.xlsx")
                    with c2:
                        st.download_button("📥 Descargar CSV Final (Data + DOM1, DOM2, DOM3)", df_ug.to_csv(index=False, sep=";").encode('utf-8'), "DataPRM_DOM3_FINAL.csv", "text/csv")
                
                with t2:
                    df_valid = df_ug.dropna(subset=[t_target, 'DOM3_num']).copy()
                    fig_log = logprob_points(df_valid, 'DOM3_num', f"Log-Probabilístico {t_target} — DOM3", t_target)
                    st.pyplot(fig_log)
                    
                with t3:
                    fig_ef = effect_prop(stats_p3, "DOM", f"Efecto proporcional — DOM3")
                    st.pyplot(fig_ef)
                    
                with t4:
                    c_2d1, c_2d2, c_2d3 = st.columns(3)
                    df_valid = df_ug.dropna(subset=['DOM3_num', cols['coords']['x'], cols['coords']['y'], cols['coords']['z']]).copy()
                    with c_2d1: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['x'], cols['coords']['y'], 'DOM3_num', "DOM3 — Vista XY (Planta)"))
                    with c_2d2: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['x'], cols['coords']['z'], 'DOM3_num', "DOM3 — Vista XZ"))
                    with c_2d3: st.pyplot(scatter_dom_fig(df_valid, cols['coords']['y'], cols['coords']['z'], 'DOM3_num', "DOM3 — Vista YZ"))
                
                with t5:
                    fig_box = generar_boxplot_reporte(df_ug, 'UG_Activa', t_target)
                    st.pyplot(fig_box)

# ---------------------------------------------------------------------------------
# SECCIONES 3D (MEJORADAS PARA VISUALIZAR Y DIFERENCIAR DOMINIOS)
# ---------------------------------------------------------------------------------
elif menu == "🔷 Secciones":
    st.markdown("<div class='main-header'>🔷 Secciones 3D (Visor de Dominios)</div>", unsafe_allow_html=True)
    
    df_act = st.session_state.df_ug if 'df_ug' in st.session_state and st.session_state.df_ug is not None else obtener_df_activo()[0]
    
    if df_act is not None:
        cols = st.session_state.cols
        
        dom_options = []
        if 'DOM1_num' in df_act.columns: dom_options.append('DOM1 (Espacial)')
        if 'DOM2_num' in df_act.columns: dom_options.append('DOM2 (Geológico)')
        if 'DOM3_num' in df_act.columns: dom_options.append('DOM3 (Definitivo)')
        
        if not dom_options:
            st.warning("⚠️ Aún no se han generado dominios. Ve a los Pasos 1, 2 o 3 para ejecutar el modelo y luego vuelve aquí para visualizarlo en 3D.")
        else:
            c_conf, c_plot = st.columns([1, 3])
            
            with c_conf:
                st.markdown("### ⚙️ Configuración")
                capa_seleccionada = st.radio("Selecciona el modelo a visualizar:", dom_options)
                
                st.markdown("---")
                tamaño_punto = st.slider("Tamaño de los bloques", 1, 15, 4)
                opacidad = st.slider("Opacidad", 0.1, 1.0, 0.8)
                
                if capa_seleccionada == 'DOM1 (Espacial)': col_target = 'DOM1_num'
                elif capa_seleccionada == 'DOM2 (Geológico)': col_target = 'DOM2_num'
                else: col_target = 'DOM3_num'
                
            with c_plot:
                df_plot = df_act.dropna(subset=[cols['coords']['x'], cols['coords']['y'], cols['coords']['z'], col_target]).copy()
                df_plot[col_target] = df_plot[col_target].astype(int)
                df_plot = df_plot.sort_values(by=col_target)
                
                dominios_unicos_numericos = sorted(df_plot[col_target].unique())
                orden_categorias_leyenda = [f"DOM {d}" for d in dominios_unicos_numericos]
                
                df_plot['Dominio'] = "DOM " + df_plot[col_target].astype(str)
                
                fig = px.scatter_3d(
                    df_plot, 
                    x=cols['coords']['x'], 
                    y=cols['coords']['y'], 
                    z=cols['coords']['z'], 
                    color='Dominio',
                    category_orders={"Dominio": orden_categorias_leyenda},
                    color_discrete_sequence=px.colors.qualitative.Alphabet,
                    title=f"Visor Interactivo - {capa_seleccionada}"
                )
                
                fig.update_traces(marker=dict(size=tamaño_punto, opacity=opacidad, line=dict(width=0)))
                fig.update_layout(
                    height=800, 
                    margin=dict(l=0, r=0, b=0, t=40),
                    legend=dict(title=dict(text="Dominios"), itemsizing='constant')
                )
                
                st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("<div style='text-align: center; color: #666;'>Empirica v14.1 (Bug Paso 2 Resuelto)</div>", unsafe_allow_html=True)
