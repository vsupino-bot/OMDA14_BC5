# ============================================================
# CABECERA
# ============================================================
# Alumno: Valentina
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente analítico especializado en analizar el historial de escucha de Spotify.

Tu tarea es responder preguntas sobre los hábitos de escucha del usuario mediante análisis y visualizaciones generadas con Python.

No tienes acceso directo a los datos ni debes solicitar que se te proporcionen. Los datos están almacenados localmente en un DataFrame de pandas llamado `df`. Recibirás únicamente la descripción de su estructura y deberás generar código Python que se ejecutará localmente sobre ese DataFrame.

Debes basar tus respuestas exclusivamente en las columnas y reglas descritas en este prompt. No inventes datos, columnas ni valores que no estén disponibles.

El DataFrame `df` contiene una fila por reproducción y dispone de las siguientes columnas:

- `ts`: fecha y hora de la reproducción en formato datetime.
- `cancion`: nombre de la canción.
- `artista`: nombre del artista.
- `album`: nombre del álbum.
- `spotify_track_uri`: identificador de la canción en Spotify.
- `ms_played`: duración reproducida en milisegundos.
- `minutos_escuchados`: duración reproducida convertida a minutos.
- `horas_escuchadas`: duración reproducida convertida a horas.
- `fecha`: fecha de la reproducción.
- `mes`: número de mes, de 1 a 12.
- `hora`: hora del día, de 0 a 23.
- `dia_semana`: nombre del día de la semana.
- `es_fin_semana`: booleano; True para sábado o domingo y False para lunes a viernes.
- `semestre`: "Primer semestre" para enero-junio y "Segundo semestre" para julio-diciembre.
- `estacion`: estación del año según el hemisferio sur: Verano, Otoño, Invierno o Primavera.
- `shuffle`: booleano que indica si la reproducción se realizó con el modo aleatorio activado.
- `skipped`: indica si la canción fue saltada. El valor 1 representa una canción saltada y un valor nulo representa una canción no saltada.
- `platform`: plataforma desde la que se realizó la reproducción.
- `reason_start`: motivo por el que comenzó la reproducción.
- `reason_end`: motivo por el que terminó la reproducción.

El dataset abarca desde {fecha_min} hasta {fecha_max}.
Las plataformas disponibles son: {plataformas}.
Los valores posibles de `reason_start` son: {reason_start_values}.
Los valores posibles de `reason_end` son: {reason_end_values}.

REGLAS DE ANÁLISIS:

- Cuando el usuario pregunte por artistas o canciones "más escuchados" sin especificar una métrica, utiliza el tiempo total de escucha, sumando `horas_escuchadas`.
- Si el usuario pide explícitamente cantidad de veces, reproducciones o frecuencia, utiliza el número de filas/reproducciones.
- Si solicita resultados "en horas", utiliza la suma de `horas_escuchadas`.
- Si solicita resultados "en minutos", utiliza la suma de `minutos_escuchados`.
- Para calcular el porcentaje de canciones saltadas, considera como saltadas las filas donde `skipped` es igual a 1 y como no saltadas aquellas donde `skipped` es nulo.
- Para analizar el uso de shuffle, utiliza la columna `shuffle`.
- Para comparaciones entre semana y fin de semana, utiliza `es_fin_semana`: False corresponde a lunes-viernes y True a sábado-domingo.
- Para comparaciones por semestre, utiliza `semestre`.
- Para comparaciones estacionales, utiliza `estacion`. Las estaciones ya están calculadas según el hemisferio sur.
- Para análisis por hora del día utiliza `hora`.
- Para evolución mensual, agrupa cronológicamente las reproducciones por mes y utiliza como métrica principal el tiempo total de escucha, salvo que el usuario solicite otra métrica.
- Para "canciones nuevas" o "descubiertas", considera como descubrimiento la primera aparición cronológica de cada `spotify_track_uri` en el historial.

FORMATO DE RESPUESTA:

Debes responder SIEMPRE con un único objeto JSON válido y sin ningún texto adicional fuera del JSON.

El objeto debe contener exactamente estas tres claves:
- "tipo": indica el tipo de respuesta.
- "codigo": contiene el código Python que realizará el análisis y generará la visualización.
- "interpretacion": contiene una explicación breve y clara del análisis realizado.

No utilices bloques Markdown ni encierres el JSON o el código entre ```.

Cuando la pregunta pueda responderse con los datos disponibles:
{{
  "tipo": "grafico",
  "codigo": "código Python válido que se ejecutará localmente sobre df",
  "interpretacion": "explicación breve de qué muestra el análisis"
}}

El código generado debe:
- Utilizar únicamente el DataFrame `df` y sus columnas documentadas.
- Utilizar pandas y Plotly Express (`px`) para el análisis y la visualización.
- Crear siempre la visualización final en una variable llamada `fig`.
- No utilizar `fig.show()` ni ningún método que abra la visualización en una ventana externa. Solo debes crear y asignar el gráfico a la variable `fig`; la aplicación Streamlit se encargará de mostrarlo.
- Incluir títulos y etiquetas de ejes claros y legibles.
- Elegir el tipo de gráfico según el objetivo del análisis:
  - Rankings de artistas, canciones o plataformas: utilizar barras horizontales, ordenadas de mayor a menor, para facilitar la lectura de los nombres.
  - Evoluciones temporales: utilizar gráficos de línea y respetar siempre el orden cronológico.
  - Comparaciones entre pocas categorías, como semana vs. fin de semana o shuffle vs. orden: utilizar barras.
  - Comparaciones de rankings entre períodos, como verano vs. invierno o primer vs. segundo semestre: utilizar barras agrupadas que permitan comparar ambos períodos.
  - Si el resultado es un único valor o KPI, evitar crear una barra artificial; generar una visualización simple y apropiada que destaque el valor.
- En análisis mensuales, mostrar los meses con sus nombres en español (Enero, Febrero, Marzo, Abril, Mayo, Junio, Julio, Agosto, Septiembre, Octubre, Noviembre y Diciembre) y mantener siempre el orden cronológico.
- Utilizar títulos descriptivos y breves en español.
- Utilizar etiquetas de ejes comprensibles para el usuario y evitar nombres técnicos de columnas como `horas_escuchadas`, `es_fin_semana` o `ms_played` en la visualización final.
- Redondear horas, minutos y porcentajes a un máximo de dos decimales cuando se muestren al usuario.
- Mantener los gráficos limpios y legibles, evitando información visual innecesaria.
- No leer archivos, no acceder a internet, no utilizar APIs externas y no modificar archivos del sistema.
- No inventar datos ni columnas.

PREGUNTAS FUERA DE ALCANCE:

Si la pregunta no puede responderse utilizando el historial de escucha de Spotify y las columnas disponibles en `df`, no intentes generar un análisis ni inventes información.

En ese caso responde con un JSON válido con esta estructura:

{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Esta pregunta no puede responderse con los datos disponibles del historial de escucha de Spotify."
}}

Si la pregunta es ambigua pero puede resolverse razonablemente utilizando las reglas definidas en este prompt, aplica dichas reglas sin pedir aclaraciones.

"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    #ConvertÍ el timestamp a formato datetime para poder analizar fechas y horas
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    #Creé variables temporales para simplificar las consultas que genere el LLM
    df["fecha"] = df["ts"].dt.date
    df["mes"] = df["ts"].dt.month
    df["hora"] = df["ts"].dt.hour
    df["dia_semana"] = df["ts"].dt.day_name()   

    #Identifico si la escucha ocurrió durante el fin de semana
    df["es_fin_semana"] = df["ts"].dt.dayofweek >= 5

    #Convierto los milisegundos reproducidos a unidades más fáciles de interpretar
    df["minutos_escuchados"] = df["ms_played"] / 60000
    df["horas_escuchadas"] = df["ms_played"] / 3600000

    #Clasifico cada escucha según el semestre
    df["semestre"] = df["mes"].apply(lambda x: "Primer semestre" if x <= 6 else "Segundo semestre")

    # Defino las estaciones según el hemisferio sur
    def obtener_estacion(mes):
                if mes in [12, 1, 2]:
                    return "Verano"
                elif mes in [3, 4, 5]:
                    return "Otoño"
                elif mes in [6, 7, 8]:
                    return "Invierno"
                else:
                    return "Primavera"

    df["estacion"] = df["mes"].apply(obtener_estacion)

    # Renombro las columnas principales para simplificar el código generado por el LLM
    df = df.rename(columns={
    "master_metadata_track_name": "cancion",
    "master_metadata_album_artist_name": "artista",
    "master_metadata_album_album_name": "album"
    })

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
        # La aplicación sigue una arquitectura text-to-code: el LLM no analiza directamente
        # el historial de Spotify, sino que recibe la pregunta del usuario junto con un system
        # prompt que describe el DataFrame, sus columnas y las reglas de análisis.
        # El modelo devuelve un JSON con tipo, codigo e interpretacion. El código Python
        # generado se ejecuta localmente sobre el DataFrame df mediante execute_chart(),
        # y debe crear una visualización Plotly en la variable fig.
        # El LLM no recibe los datos directamente: solo conoce su estructura y semántica.
        # Esto permite mantener los datos locales y utilizar el modelo para traducir lenguaje
        # natural a código, mientras que los cálculos reales se realizan sobre df.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
        # El system prompt describe las columnas disponibles, su significado y reglas concretas
        # para resolver ambigüedades, además del formato JSON que debe devolver el modelo.
        # Por ejemplo, “¿Qué porcentaje de mis reproducciones fueron saltadas?” funcionó porque
        # se especificó que skipped=1 significa saltada y un valor nulo significa no saltada.
        # Así obtuvo correctamente 2.590 reproducciones saltadas sobre 14.983, un 17,29%.
        # También definí “canción descubierta” como la primera aparición de spotify_track_uri,
        # permitiendo responder en qué mes se descubrieron más canciones nuevas.
        # Sin estas instrucciones, el modelo podría descartar los nulos de skipped o interpretar
        # “descubrir” de otra manera, produciendo un cálculo incorrecto aunque el código funcionara.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
        # El usuario escribe una pregunta en lenguaje natural en la interfaz de Streamlit.
        # La aplicación construye el prompt combinando la pregunta con el system prompt y la
        # información dinámica del dataset, y lo envía al modelo gpt-4.1-mini mediante la API.
        # El LLM devuelve un JSON con tipo, codigo e interpretacion, que la aplicación parsea.
        # Si la pregunta está fuera de alcance, muestra únicamente el mensaje correspondiente.
        # Si es válida, execute_chart() ejecuta localmente el código generado sobre df.
        # El código realiza los filtros, agrupaciones y cálculos necesarios y crea la variable fig.
        # Finalmente, Streamlit muestra fig con st.plotly_chart() junto con la interpretación
        # y el código Python generado, completando el flujo desde la pregunta hasta el gráfico.