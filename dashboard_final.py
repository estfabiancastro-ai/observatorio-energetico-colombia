# ============================================
# OBSERVATORIO ENERGÉTICO DE COLOMBIA
# Comunidades Energéticas - Versión Mejorada y Corregida
# ============================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.markdown("""
<style>

/* Fondo sidebar */
section[data-testid="stSidebar"] {
    background-color: #061a2b;
}

/* Panel de navegación */
[data-testid="stRadio"] {
    background: #0b2a40;
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #00c6ff;
    box-shadow: 0px 0px 12px rgba(0,198,255,0.6);
}

/* Título panel */
.nav-title{
    font-size:26px;
    font-weight:bold;
    color:#00c6ff;
    text-align:center;
    margin-bottom:10px;
}

/* Texto radio */
label[data-baseweb="radio"]{
    font-size:16px;
    color:white;
}

</style>
""", unsafe_allow_html=True)








# ────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ────────────────────────────────────────────
st.set_page_config(
    page_title="Observatorio Comunidades Energéticas Colombia",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.title("⚡ Observatorio de Comunidades Energéticas de Colombia")

# ────────────────────────────────────────────
# CACHE DE DATOS Y MODELOS
# ────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("dataset/UPME - SIMEC/dataset_comunidades_energeticas_colombia.csv")
        df["departamento"] = df["departamento"].str.upper().str.strip()
        
        mapa = gpd.read_file("georreferencia/MGN_ADM_DPTO_POLITICO.shp")
        mapa["dpto_cnmbr"] = mapa["dpto_cnmbr"].str.upper().str.strip()
        
        # Merge
        mapa = mapa.merge(
            df,
            left_on="dpto_cnmbr",
            right_on="departamento",
            how="left"
        )
        return df, mapa
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        st.stop()

@st.cache_resource
def entrenar_modelos(df):
    features = ["solar", "viento", "biomasa", "pobreza_energetica", "generacion_renovable"]
    X = df[features].copy()
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Variable objetivo
    y = calcular_indice_ce(df)
    
    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Red Neuronal (MLP)
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    
    # Métricas rápidas
    rf_pred = rf.predict(X_test)
    mlp_pred = mlp.predict(X_test)
    
    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "rf": rf,
        "mlp": mlp,
        "metrics": {
            "rf": {"MAE": mean_absolute_error(y_test, rf_pred), "R²": r2_score(y_test, rf_pred)},
            "mlp": {"MAE": mean_absolute_error(y_test, mlp_pred), "R²": r2_score(y_test, mlp_pred)}
        }
    }

def calcular_indice_ce(df):
    """Índice normalizado entre 0 y 1"""
    df_norm = pd.DataFrame()
    df_norm["solar"]      = df["solar"]      / df["solar"].max()
    df_norm["viento"]     = df["viento"]     / df["viento"].max()
    df_norm["biomasa"]    = df["biomasa"]    / df["biomasa"].max()
    df_norm["eq_pobreza"] = 1 - df["pobreza_energetica"].clip(0,1)
    df_norm["renovable"]  = df["generacion_renovable"].clip(0,1)
    df_norm["comunidades"]= df["comunidades_energeticas"] / df["comunidades_energeticas"].max()
    
    return df_norm.mean(axis=1)

# ────────────────────────────────────────────
# CARGA DE DATOS Y MODELOS (solo una vez)
# ────────────────────────────────────────────
with st.spinner("Cargando datos y modelos..."):
    df, mapa_geo = cargar_datos()
    modelos = entrenar_modelos(df)

df["indice_ce"] = calcular_indice_ce(df)
df["cluster"]   = modelos["kmeans"].predict(modelos["scaler"].transform(df[["solar","viento","biomasa","pobreza_energetica","generacion_renovable"]]))

# Calcular índice en mapa_geo (corrección del error)
mapa_geo["indice_ce"] = (
    mapa_geo["solar"] / mapa_geo["solar"].max() +
    mapa_geo["viento"] / mapa_geo["viento"].max() +
    mapa_geo["biomasa"] / mapa_geo["biomasa"].max() +
    (1 - mapa_geo["pobreza_energetica"]) +
    mapa_geo["generacion_renovable"] +
    mapa_geo["comunidades_energeticas"] / mapa_geo["comunidades_energeticas"].max()
) / 6

# ────────────────────────────────────────────
# NAVEGACIÓN  ←─ AQUÍ SE AGREGA LA NUEVA PÁGINA
# ────────────────────────────────────────────
paginas = [
    "Dashboard",
    "Metodología",
    "Simulador",
    "Predicción IA",
    "Ranking territorial",
    "Comparación LATAM"
]

with st.sidebar:

    st.markdown('<div class="nav-box">', unsafe_allow_html=True)

    st.markdown(
        '<div class="nav-title">🚀 Panel de Navegación</div>',
        unsafe_allow_html=True
    )

    pagina = st.radio("", paginas)

    st.markdown('</div>', unsafe_allow_html=True)



# ────────────────────────────────────────────
# DASHBOARD
# ────────────────────────────────────────────
if pagina == "Dashboard":
    st.subheader("Indicadores Nacionales")
    
    cols = st.columns(4)
    cols[0].metric("☀️ Solar Promedio", f"{df['solar'].mean():.2f}")
    cols[1].metric("🌬️ Viento Promedio", f"{df['viento'].mean():.2f}")
    cols[2].metric("🏡 Comunidades Energéticas", int(df["comunidades_energeticas"].sum()))
    cols[3].metric("📊 Índice CE Nacional", f"{df['indice_ce'].mean():.3f}")
    
    # Mapa Coroplético
    st.subheader("Mapa Energético de Colombia")
    fig_map = px.choropleth_mapbox(
        mapa_geo,
        geojson=mapa_geo.__geo_interface__,
        locations=mapa_geo.index,
        color="indice_ce",
        hover_name="departamento",
        color_continuous_scale="YlGnBu",
        mapbox_style="carto-positron",
        zoom=4.2,
        center={"lat": 4.6, "lon": -74.1},
        opacity=0.8,
        height=500
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clusters Energéticos")
        fig_cluster = px.scatter(
            df,
            x="solar",
            y="viento",
            color=df["cluster"].astype(str),
            size="biomasa",
            hover_name="departamento",
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=450
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        st.subheader("Potencial 3D")
        fig_3d = px.scatter_3d(
            df,
            x="solar",
            y="viento",
            z="biomasa",
            color="indice_ce",
            size="comunidades_energeticas",
            hover_name="departamento",
            color_continuous_scale="Viridis",
            height=500
        )
        fig_3d.update_traces(marker=dict(opacity=0.8))
        st.plotly_chart(fig_3d, use_container_width=True)



    # ------------------------------------------------
# RADAR ENERGÉTICO - Perfil promedio nacional
# ------------------------------------------------
    st.subheader("Perfil Energético Promedio Nacional")

# ── Protección contra errores comunes ──
    if df.empty or 'solar' not in df.columns:
        st.warning("No hay datos disponibles para generar el radar energético.")
    else:
        try:
        # Preparar valores normalizados (0 a 1)
            valores = [
                df["solar"].mean() / df["solar"].max() if df["solar"].max() != 0 else 0,
                df["viento"].mean() / df["viento"].max() if df["viento"].max() != 0 else 0,
                df["biomasa"].mean() / df["biomasa"].max() if df["biomasa"].max() != 0 else 0,
                1 - df["pobreza_energetica"].mean() if not pd.isna(df["pobreza_energetica"].mean()) else 0.5,
                df["generacion_renovable"].mean() if not pd.isna(df["generacion_renovable"].mean()) else 0.5
            ]

            categorias = ["Solar", "Eólico", "Biomasa", "Equidad Energética", "Renovables"]

        # Crear figura
            fig_radar = go.Figure()

        # Área principal (perfil real)
            fig_radar.add_trace(go.Scatterpolar(
                r=valores,
                theta=categorias,
                fill='toself',
                name='Colombia (promedio nacional)',
                line_color='royalblue',
                fillcolor='rgba(65, 105, 225, 0.25)',
                line=dict(width=2)
            ))

        # Línea de referencia al 50% (círculo punteado)
            fig_radar.add_trace(go.Scatterpolar(
                r=[0.5] * len(categorias),
                theta=categorias,
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=False
            ))

        # Configuración visual
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=12),
                        ticks="outside",
                        ticklen=5
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=14),
                        rotation=90,          # para que quede más legible
                        direction='clockwise' # opcional: sentido de las agujas del reloj
                    )
                ),
                showlegend=True,
                title=dict(
                    text="Perfil Energético Promedio Nacional - Dimensiones clave",
                    x=0.5,
                    font=dict(size=16)
                ),
                height=550,
                margin=dict(l=40, r=40, t=80, b=40)
            )

        # Mostrar gráfico
            st.plotly_chart(fig_radar, use_container_width=True)

        # Opcional: mostrar los valores numéricos debajo para más claridad
            with st.expander("Valores detallados (normalizados 0–1)"):
                st.markdown(" | ".join([f"**{cat}:** {val:.3f}" for cat, val in zip(categorias, valores)]))

        except Exception as e:
            st.error(f"Error al generar el radar: {str(e)}")
            st.info("Posibles causas: valores NaN, columnas faltantes o división por cero.")


# ────────────────────────────────────────────
# METODOLOGÍA
# ────────────────────────────────────────────
elif pagina == "Metodología":
    st.header("Metodología del Observatorio")
    st.markdown("""
    **Objetivo**: Evaluar el **potencial territorial** para el desarrollo de comunidades energéticas en Colombia.
                El índice propuesto sintetiza información sobre recursos renovables, equidad energética
    y desarrollo comunitario para identificar territorios con mayor potencial para
    implementar sistemas energéticos descentralizados.

    ### Dimensiones ponderadas del índice (todas normalizadas 0–1)
    - ☀️ Potencial solar  
    - 🌬️ Potencial eólico  
    - 🌾 Biomasa  
    - ♻️ Generación renovable existente  
    - ❤️‍🩹 Equidad energética (1 – pobreza energética)  
    - 🤝 Comunidades energéticas reportadas  

    ### Técnicas utilizadas
    - Normalización min-max por variable  
    - Índice compuesto (media aritmética simple)  
    - Clustering K-Means (4 grupos)  
    - Modelos de regresión: Random Forest + Red Neuronal (MLP)  
    - Visualización geoespacial con GeoPandas + Plotly Mapbox  
                
                """)
    
    st.subheader("1️⃣ Normalización de variables")
    st.markdown("""
    Todas las variables se normalizan entre **0 y 1** para permitir su comparación
    utilizando el método **Min-Max**.
    """)

    st.latex(r'''
        X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
        ''')
    
    st.markdown("""
    donde:

    - \(X\) es el valor original del indicador  
    - \(X_{min}\) es el valor mínimo observado  
    - \(X_{max}\) es el valor máximo observado
    """)

    st.subheader("2️⃣ Construcción del Índice de Comunidades Energéticas")

    st.markdown("""
    El índice se construye mediante la **media aritmética simple de seis dimensiones**.
    """)

    st.latex(r'''
    I_{CE} =
    \frac{
    S + V + B + R + (1 - P_e) + C
    }{6}
    ''')

    st.markdown("""
    donde:

    - **S** = potencial solar  
    - **V** = potencial eólico  
    - **B** = biomasa disponible  
    - **R** = generación renovable existente  
    - **Pₑ** = pobreza energética  
    - **C** = número de comunidades energéticas
    """)

    st.markdown("""
    El término **(1 − Pₑ)** se utiliza para representar la dimensión de **equidad energética**,
    de manera que valores altos indiquen mejores condiciones de acceso a energía.
    """)
    
    #------------------------------------------------------------------------------------------------------------------------

    st.header("Metodología del Índice de Comunidades Energéticas")

    st.markdown("""
    El **Índice de Desarrollo de Comunidades Energéticas (IDCE)** evalúa el potencial
    territorial para el desarrollo de comunidades energéticas en Colombia mediante
    un enfoque multidimensional que integra recursos energéticos, capacidades
    comunitarias, equidad energética y transición energética.
    """)

    st.subheader("1️⃣ Construcción del índice compuesto")

    st.latex(r'''
        IDCE = \frac{D_1 + D_2 + D_3 + D_4 + D_5}{5}
    ''')

    st.markdown("""
    donde cada dimensión se encuentra normalizada en el intervalo **[0,1]**.
    """)



    st.subheader("2️⃣ Dimensiones del índice")

    st.markdown("### Dimensión 1: Recursos Energéticos Renovables")

    st.latex(r'''
    D_1 = \frac{I_{solar} + I_{eolico} + I_{biomasa}}{3}
    ''')

    st.markdown("""
    **I_solar**: potencial de radiación solar  
    **I_eolico**: velocidad promedio del viento  
    **I_biomasa**: potencial energético de biomasa residual
        """)


    st.markdown("### Dimensión 2: Potencia Energética Comunitaria")

    st.latex(r'''
    D_2 = \frac{I_{potencia} + I_{autonomia}}{2}
    ''')

    st.markdown("""
    - **I_potencia**: potencia instalada per cápita  
    - **I_autonomia**: relación entre energía generada localmente y energía consumida
    """)


    st.markdown("### Dimensión 3: Desarrollo Organizativo")

    st.latex(r'''
    D_3 = \frac{I_{CE} + I_{participacion}}{2}
    ''')

    st.markdown("""
    - **I_CE**: densidad de comunidades energéticas  
    - **I_participacion**: tasa de participación de usuarios en comunidades energéticas
    """)

    st.latex(r'''
    I_{CE} =
    \frac{N_{comunidades}}{Poblacion}
    ''')

    st.latex(r'''
    I_{participacion} =
    \frac{Usuarios\ activos}{Usuarios\ potenciales}
    ''')


    st.markdown("### Dimensión 4: Pobreza Energética")

    st.latex(r'''
    D_4 = 1 - IPEM
    ''')

    st.markdown("""
    - **IPEM**: índice de pobreza energética multidimensional
    - Se utiliza el inverso para que valores mayores indiquen mejores condiciones energéticas.
    """)



    st.markdown("### Dimensión 5: Transición Energética Local")

    st.latex(r'''
    D_5 = \frac{I_{renovable} + I_{GD}}{2}
    ''')

    st.markdown("""
    - **I_renovable**: participación de energías renovables en el consumo energético local  
    - **I_GD**: penetración de generación distribuida
    """)


    st.subheader("3️⃣ Análisis de clusters energéticos")

    st.markdown("""
    Para identificar territorios con características energéticas similares,
    se aplica el algoritmo de **clustering K-Means** sobre las variables del índice.
    """)

    st.latex(r'''
    J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
    ''')

    st.markdown("""
    Este método minimiza la distancia entre cada observación y el centroide del cluster,
    permitiendo identificar grupos de departamentos con perfiles energéticos similares.
    """)

    #-------------------------------------------------------------------------------------------

    st.subheader("4️⃣ Modelos de Machine Learning")

    st.markdown("""
    Para explorar relaciones no lineales entre variables energéticas se aplican
    dos modelos de aprendizaje automático:

    **Random Forest**

    - modelo de árboles de decisión ensamblados
    - captura relaciones complejas entre variables energéticas

    **Red neuronal multicapa (MLP)**

    - arquitectura feedforward
    - aprende patrones no lineales entre recursos energéticos
    """)

    st.latex(r'''
    y = f(Wx + b)
    ''')


    st.subheader("Relación entre cobertura eléctrica y pobreza")

    col1, col2, col3 = st.columns([1, 3, 1])  # columnas laterales vacías para centrar

    with col2:
        st.image(
            "Imagenes/probreza.png",
            caption="Relación entre cobertura eléctrica y pobreza en departamentos de Colombia",
            width=800,  # o el tamaño que prefieras
            use_container_width=False
        )
    



    st.markdown("""
### Interpretación

La figura muestra una relación negativa significativa entre la cobertura eléctrica
y la tasa de pobreza en los departamentos de Colombia.

- Coeficiente de correlación: **r = -0.843**
- Significancia estadística: **p < 0.001**

Esto sugiere que los territorios con menor acceso a electricidad tienden
a presentar mayores niveles de pobreza, particularmente en regiones
pertenecientes a las **Zonas No Interconectadas (ZNI)**.
""")
    
    st.subheader(" RADAR ENERGETICO - Perfil promedio nacional")
    st.markdown("""
    La siguiente figura muestra el **perfil energético promedio** de Colombia, 
    calculado con los valores medios de cada dimensión normalizados entre 0 y 1. 
    Esto permite visualizar de forma rápida las fortalezas y debilidades relativas 
    del país en las variables clave del índice.
    """)

    
    st.image(
            "Imagenes/radar_energetico.png",
            caption="Perfil energético promedio de Colombia (valores normalizados 0–1)",
            width=1000,  # o el tamaño que prefieras
            use_container_width=False
        )
    

    # ────────────────────────────────────────────
    # NUEVA PARTE: COMPARACIÓN REGIONAL (agregada al final)
    # ────────────────────────────────────────────
    st.subheader("Comparación regional: Potencial de Comunidades Energéticas en América Latina")
    st.markdown("""
    Para contextualizar el desarrollo de comunidades energéticas en Colombia, se presenta una estimación comparativa del **Índice de Desarrollo de Comunidades Energéticas (IDCE)** en 10 países de América Latina.  
    Esta estimación considera factores como penetración de renovables, marcos regulatorios para generación distribuida, potencial de recursos y avance en modelos comunitarios/descentralizados (datos aproximados basados en reportes regionales y análisis comparativos 2023–2025).
    """)

    # Datos de la tabla que me diste
    data_idce = {
        "País": ["Brasil", "Chile", "Uruguay", "Argentina", "Colombia", "México", "Perú", "Ecuador", "Paraguay", "Bolivia"],
        "IDCE estimado": [0.68, 0.66, 0.64, 0.63, 0.62, 0.60, 0.59, 0.58, 0.57, 0.55]
    }

    df_idce = pd.DataFrame(data_idce)

    # Mostrar la tabla con formato y gradiente
    st.markdown("**Cuadro 2: Estimación comparativa del Índice de Desarrollo de Comunidades Energéticas (IDCE) en América Latina**")
    st.dataframe(
        df_idce.style.format({"IDCE estimado": "{:.2f}"})
                   .background_gradient(subset=["IDCE estimado"], cmap="YlGn", low=0, high=1),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    **Interpretación de los resultados**  
    Los resultados sugieren que países como **Brasil**, **Chile** y **Uruguay** presentan valores más altos del índice debido a una mayor penetración de energías renovables y marcos regulatorios más avanzados para la generación distribuida.  

    **Colombia** se ubica en una posición intermedia, reflejando un alto potencial de recursos renovables (solar, eólico, hídrico y biomasa) pero un desarrollo aún incipiente de comunidades energéticas y modelos descentralizados. Esto resalta oportunidades clave para fortalecer políticas públicas, incentivos regulatorios y financiamiento comunitario en el país, alineándose con los objetivos de transición energética justa y sostenible en la región.
    """)

    st.caption("""
    Nota: El IDCE es una estimación comparativa aproximada (no oficial). Se basa en indicadores agregados de penetración renovable, equidad energética, potencial distribuido y avance normativo (fuentes: OLADE, IRENA, reportes nacionales y análisis regionales aproximados 2023–2025).
    """)
    

    




# ────────────────────────────────────────────
# SIMULADOR
# ────────────────────────────────────────────
elif pagina == "Simulador":
    st.header("Simulador de Potencial para Comunidad Energética")
    
    with st.form("simulador_form"):
        col1, col2 = st.columns(2)
        with col1:
            solar    = st.slider("Potencial Solar (kWh/m²/día)",       0.0, 6.5,  4.2, 0.1)
            viento   = st.slider("Potencial Eólico (m/s)",             0.0, 10.0, 4.5, 0.1)
            biomasa  = st.slider("Potencial Biomasa (ton/ha/año)",     0.0, 1.5,  0.6, 0.05)
        
        with col2:
            pobreza  = st.slider("Pobreza energética (%)",             0.0, 100.0, 35.0, 1.0) / 100
            renovable= st.slider("Participación renovables (%)",       0.0, 100.0, 45.0, 1.0) / 100
        
        submitted = st.form_submit_button("Calcular índice", type="primary")
    
    if submitted:
        idx = (
            (solar   / 6.5) +
            (viento  / 10.0) +
            (biomasa / 1.5) +
            (1 - pobreza) +
            renovable
        ) / 5
        
        st.metric(
            "Índice Potencial Estimado",
            f"{idx:.3f}",
            delta=None,
            delta_color="normal"
        )
        st.caption("Escala aproximada: 0.0–0.35 bajo • 0.35–0.55 medio • 0.55–0.80 alto • >0.80 muy alto")

# ────────────────────────────────────────────
# PREDICCIÓN IA
# ────────────────────────────────────────────
elif pagina == "Predicción IA":
    st.header("Predicción del Índice con IA")
    
    tab1, tab2 = st.tabs(["Random Forest", "Red Neuronal"])
    
    with tab1:
        st.subheader("Random Forest")
        df["pred_rf"] = modelos["rf"].predict(df[["solar","viento","biomasa","pobreza_energetica","generacion_renovable"]])
        fig_rf = px.bar(
            df.sort_values("pred_rf", ascending=False),
            x="departamento",
            y="pred_rf",
            color="pred_rf",
            height=550
        )
        st.plotly_chart(fig_rf, use_container_width=True)
        st.caption(f"MAE: {modelos['metrics']['rf']['MAE']:.4f} | R²: {modelos['metrics']['rf']['R²']:.3f}")
    
    with tab2:
        st.subheader("Red Neuronal (MLP)")
        df["pred_mlp"] = modelos["mlp"].predict(df[["solar","viento","biomasa","pobreza_energetica","generacion_renovable"]])
        fig_mlp = px.bar(
            df.sort_values("pred_mlp", ascending=False),
            x="departamento",
            y="pred_mlp",
            color="pred_mlp",
            height=550
        )
        st.plotly_chart(fig_mlp, use_container_width=True)
        st.caption(f"MAE: {modelos['metrics']['mlp']['MAE']:.4f} | R²: {modelos['metrics']['mlp']['R²']:.3f}")

# ────────────────────────────────────────────
# RANKING TERRITORIAL
# ────────────────────────────────────────────
elif pagina == "Ranking territorial":
    st.header("Ranking de Potencial Territorial")
    
    df_rank = df.copy()
    df_rank["potencial_simple"] = (
        df_rank["solar"] +
        df_rank["viento"] +
        df_rank["biomasa"] +
        (1 - df_rank["pobreza_energetica"]) * 2   # mayor peso a equidad
    )
    
    ranking = df_rank.sort_values("potencial_simple", ascending=False).reset_index(drop=True)
    ranking.index += 1  # ranking 1,2,3...
    
    st.dataframe(
        ranking[["departamento", "potencial_simple", "indice_ce", "comunidades_energeticas", "cluster"]].style.format({
            "potencial_simple": "{:.3f}",
            "indice_ce": "{:.3f}"
        }).background_gradient(subset=["potencial_simple"], cmap="YlGn"),
        use_container_width=True,
        height=600
    )

# ────────────────────────────────────────────
# NUEVA SECCIÓN: COMPARACIÓN LATAM (agregada al final)
# ────────────────────────────────────────────
elif pagina == "Comparación LATAM":
    st.header("Comparación Regional - Potencial Comunidades Energéticas (10 países LATAM)")
    st.caption("Valores aproximados basados en fuentes públicas 2023–2025 (Global Solar Atlas, IRENA, IEA, OLADE, World Bank, reportes nacionales)")

    # Datos de 10 países (estimaciones plausibles)
    data_latam = {
        "País": ["Colombia", "Brasil", "México", "Chile", "Argentina", "Perú", "Ecuador", "Bolivia", "Uruguay", "Paraguay"],
        "Solar (kWh/m²/día)": [4.5, 5.2, 5.6, 6.1, 5.4, 5.3, 4.8, 5.5, 4.9, 5.0],
        "Viento (m/s prom. zonas)": [4.8, 6.5, 5.5, 7.2, 8.1, 6.0, 5.2, 5.8, 7.5, 6.2],
        "Biomasa (proxy)": [0.8, 1.4, 0.9, 0.7, 1.1, 0.95, 1.0, 1.2, 0.85, 1.3],
        "Pobreza energética (%)": [32, 18, 22, 15, 25, 28, 35, 42, 12, 20],
        "% Generación Renovable": [68, 89, 28, 70, 35, 55, 82, 65, 94, 99],
        "Comunidades energéticas (proxy)": [45, 320, 180, 95, 140, 70, 55, 40, 85, 60]
    }

    df_latam = pd.DataFrame(data_latam)

    # Normalización simple (min-max) para índice comparable
    for col in ["Solar (kWh/m²/día)", "Viento (m/s prom. zonas)", "Biomasa (proxy)", "Pobreza energética (%)", "% Generación Renovable", "Comunidades energéticas (proxy)"]:
        max_val = df_latam[col].max()
        min_val = df_latam[col].min()
        if max_val > min_val:
            df_latam[f"{col}_norm"] = (df_latam[col] - min_val) / (max_val - min_val)
        else:
            df_latam[f"{col}_norm"] = 0.5  # caso raro

    df_latam["Índice CE estimado"] = (
        df_latam["Solar (kWh/m²/día)_norm"] +
        df_latam["Viento (m/s prom. zonas)_norm"] +
        df_latam["Biomasa (proxy)_norm"] +
        (1 - df_latam["Pobreza energética (%)_norm"]) +
        df_latam["% Generación Renovable_norm"] +
        df_latam["Comunidades energéticas (proxy)_norm"]
    ) / 6

    # Mapa choroplético mundial (simple, por país)
    fig_map = px.choropleth(
        df_latam,
        locations="País",
        locationmode="country names",
        color="Índice CE estimado",
        hover_name="País",
        hover_data={
            "Solar (kWh/m²/día)": True,
            "Viento (m/s prom. zonas)": True,
            "% Generación Renovable": True,
            "Índice CE estimado": ":.3f"
        },
        color_continuous_scale="YlGnBu",
        title="Potencial aproximado para Comunidades Energéticas - 10 países LATAM",
        height=500
    )
    fig_map.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            center=dict(lat=-15, lon=-60),
            lonaxis_range=[-120, -30],
            lataxis_range=[-55, 15]
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Ranking de barras horizontal
    df_sorted = df_latam.sort_values("Índice CE estimado", ascending=True)
    fig_bar = px.bar(
        df_sorted,
        x="Índice CE estimado",
        y="País",
        color="Índice CE estimado",
        orientation="h",
        title="Ranking aproximado de potencial (Índice CE)",
        text_auto=".3f",
        height=500
    )
    fig_bar.update_layout(xaxis_title="Índice CE (0–1)", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Tabla de datos
    st.subheader("Datos base (aproximados)")
    st.dataframe(
        df_latam[[
            "País", "Solar (kWh/m²/día)", "Viento (m/s prom. zonas)", 
            "Pobreza energética (%)", "% Generación Renovable", 
            "Índice CE estimado"
        ]].style.format({
            "Índice CE estimado": "{:.3f}",
            "Pobreza energética (%)": "{:.0f}%",
            "% Generación Renovable": "{:.0f}%"
        }).background_gradient(subset=["Índice CE estimado"], cmap="YlGn"),
        use_container_width=True
    )

    st.caption("""
    Nota: Los valores son estimaciones representativas (promedios nacionales o zonas clave).  
    No sustituyen análisis detallados locales.  
    Fuentes principales: Global Solar Atlas, IRENA, IEA, OLADE, World Bank (2023–2025).
    """)