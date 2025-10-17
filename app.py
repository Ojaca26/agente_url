import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os

# --- Configuración de la página ---
st.set_page_config(page_title="Agente URL → Prompt Bot Asesor", page_icon="🤖")
st.title("🤖 Agente IA – Generador de Prompt para Bot Asesor")
st.caption("Escanea una URL, estructura su contenido y genera un prompt profesional para un chatbot de atención al cliente.")

# --- Inicializar API de Gemini ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

    st.success("✅ Conectado correctamente al modelo Gemini.")
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        google_api_key=api_key
    )
except Exception as e:
    st.error(f"❌ Error al configurar la API de Google: {e}")
    st.stop()


# --- Funciones auxiliares ---
def es_valido(url):
    parsed = urlparse(url)
    esquema_valido = bool(parsed.netloc) and bool(parsed.scheme)
    extensiones_excluidas = ['.pdf', '.jpg', '.png', '.zip', '.docx', '.gif', '.mp3', '.mp4']
    no_es_archivo = not any(parsed.path.lower().endswith(ext) for ext in extensiones_excluidas)
    return esquema_valido and no_es_archivo


def obtener_enlaces_pagina(url):
    urls = set()
    try:
        r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        sopa = BeautifulSoup(r.content, 'html.parser')
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        for link in sopa.find_all('a', href=True):
            href = link['href']
            absoluta = urljoin(base, href)
            if urlparse(absoluta).netloc == urlparse(base).netloc and es_valido(absoluta):
                urls.add(absoluta)
    except Exception as e:
        st.warning(f"No se pudo acceder a {url}: {e}")
    return urls


def extraer_texto(url):
    try:
        r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        sopa = BeautifulSoup(r.content, 'html.parser')
        for s in sopa(["script", "style"]):
            s.decompose()
        texto = " ".join(t.strip() for t in sopa.stripped_strings)
        return texto
    except Exception as e:
        st.warning(f"No se pudo extraer texto de {url}: {e}")
        return ""


# --- Paso 1: estructurar contenido ---
def analizar_y_estructurar_contenido(texto_pagina, url):
    plantilla = """
    Eres un asistente de IA experto en analizar y documentar contenido web.
    Procesa el texto de la URL '{url}' y preséntalo completo, organizado por secciones.
    Usa Markdown (## para secciones). No resumas ni omitas detalles.

    Contenido:
    ---
    {texto}
    ---
    """
    prompt = ChatPromptTemplate.from_template(plantilla)
    cadena = prompt | llm | StrOutputParser()
    texto_limitado = texto_pagina[:30000]
    try:
        return cadena.invoke({"texto": texto_limitado, "url": url})
    except Exception as e:
        st.warning(f"⚠️ Error procesando {url}: {e}")
        return ""


# --- Paso 2: generar prompt final ---
def generar_prompt_bot(nombre_empresa, link_empresa, contenido_total, guia_base):
    plantilla_bot = f"""
Eres un analista de IA especializado en crear prompts de asistentes virtuales.

Usa toda la información proporcionada del sitio web de la empresa **{nombre_empresa}** para generar un prompt completo y profesional para un **bot asesor y de servicio al cliente**. 

Debes inspirarte en la siguiente guía estructural y adaptarla al contexto real de la empresa:

---
### Guía estructural del prompt del bot:
{guia_base}
---

Ahora, con base en el contenido analizado y la estructura guía anterior, genera el **prompt final** personalizado para {nombre_empresa}.
Incluye:
- Descripción de la empresa y su propósito.
- Reglas del chatbot.
- Introducción y consentimiento.
- Menú principal y subopciones.
- Flujo conversacional con respuestas naturales y humanas.
- Frases y tono guía.
- Cierre amable y proactivo.

Información base del sitio ({link_empresa}):
---
{contenido_total[:25000]}
---
"""
    prompt = ChatPromptTemplate.from_template(plantilla_bot)
    cadena = prompt | llm | StrOutputParser()
    try:
        return cadena.invoke({})
    except Exception as e:
        st.error(f"⚠️ Error generando el prompt del bot: {e}")
        return "Error al generar el prompt del bot."


# --- Entrada de usuario ---
url_usuario = st.text_input("🌐 Introduce la URL del sitio web:", placeholder="https://ejemplo.com")
nombre_empresa = st.text_input("🏢 Nombre de la empresa:", placeholder="Kravata")
archivo_prompt = "prompt_kravata.txt"

# --- Cargar guía base desde archivo ---
try:
    with open(archivo_prompt, "r", encoding="utf-8") as f:
        guia_base = f.read()
        st.success(f"📄 Guía base cargada desde '{archivo_prompt}' correctamente.")
except Exception as e:
    st.error(f"⚠️ No se pudo cargar la guía base '{archivo_prompt}': {e}")
    st.stop()

st.markdown("---")

# --- Ejecución principal ---
if st.button("🚀 Iniciar Escaneo y Generar Prompt"):
    if url_usuario and es_valido(url_usuario):
        with st.spinner('🔍 Buscando enlaces en la página principal...'):
            enlaces = list(obtener_enlaces_pagina(url_usuario))
            enlaces.insert(0, url_usuario)
            visitados, contenido_final = set(), f"# Reporte: {urlparse(url_usuario).netloc}\n\n"

        max_paginas = st.slider("Número máximo de páginas a analizar:", 1, 15, 5)
        barra = st.progress(0)

        for i, enlace in enumerate(enlaces[:max_paginas]):
            if enlace not in visitados:
                visitados.add(enlace)
                with st.spinner(f"Analizando página {i+1}/{len(enlaces[:max_paginas])}: {enlace}"):
                    texto = extraer_texto(enlace)
                    if texto:
                        resultado = analizar_y_estructurar_contenido(texto, enlace)
                        contenido_final += f"## Página: {enlace}\n\n{resultado}\n\n---\n\n"
                    barra.progress((i + 1) / max_paginas)

        st.success("✅ Contenido estructurado correctamente")
        st.markdown("---")
        st.header("📄 Contenido Analizado del Sitio")
        st.markdown(contenido_final[:30000])

        # Generar prompt final
        st.header("🧠 Generando Prompt Personalizado del Bot...")
        prompt_final = generar_prompt_bot(nombre_empresa, url_usuario, contenido_final, guia_base)

        st.success("🎯 Prompt generado exitosamente")
        st.text_area("📘 Prompt Final para el Bot", prompt_final, height=500)

        # Guardar como archivo .txt
        archivo_salida = f"Prompt_{nombre_empresa.replace(' ', '_')}.txt"
        with open(archivo_salida, "w", encoding="utf-8") as f:
            f.write(prompt_final)

        with open(archivo_salida, "rb") as file:
            st.download_button(
                label="📥 Descargar Prompt (.txt)",
                data=file,
                file_name=archivo_salida,
                mime="text/plain"
            )
    else:
        st.error("Por favor, introduce una URL válida.")
