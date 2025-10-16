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
st.set_page_config(page_title="Agente Web Scraper IA", page_icon="🤖")
st.title("🤖 Agente IA Web Scraper")
st.caption("Introduce la URL de un sitio web para extraer y ESTRUCTURAR su contenido.")

# --- Inicializar API de Gemini ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

    # --- Diagnóstico visual ---
    st.success("✅ Conectado al modelo Gemini correctamente (API v1 estable).")
    st.write("🔐 Clave cargada correctamente.")

    # --- Mostrar modelos disponibles (opcional para depuración) ---
    try:
        modelos = genai.list_models()
        disponibles = [m.name for m in modelos]
        st.write("📦 Modelos detectados por tu API key:", disponibles)
    except Exception as e:
        st.warning(f"No se pudieron listar los modelos: {e}")

    # --- Inicializa el modelo principal ---
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",  # 👈 usa multimodal si tu clave lo soporta
        google_api_key=api_key
    )

except Exception as e:
    st.error(f"❌ Error al configurar el modelo de IA: {e}")
    st.stop()

# --- Funciones de Web Scraping ---
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


# --- Función IA para estructurar contenido ---
def analizar_y_estructurar_contenido(texto_pagina, url):
    if not texto_pagina:
        return "No se encontró contenido textual para analizar."

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
        return "⚠️ No se pudo estructurar el contenido de esta página."


# --- Interfaz principal ---
url_usuario = st.text_input("🔗 Introduce la URL del sitio web a escanear:", placeholder="https://ejemplo.com")

if st.button("🚀 Iniciar Escaneo"):
    if url_usuario and es_valido(url_usuario):
        with st.spinner('🔍 Buscando enlaces en la página principal...'):
            enlaces = list(obtener_enlaces_pagina(url_usuario))
            enlaces.insert(0, url_usuario)
            visitados, contenido_final = set(), f"# Reporte: {urlparse(url_usuario).netloc}\n\n"

        max_paginas = st.slider("Número máximo de páginas a analizar:", 1, 20, 5)
        barra = st.progress(0)

        for i, enlace in enumerate(enlaces[:max_paginas]):
            if enlace not in visitados:
                visitados.add(enlace)
                with st.spinner(f"Analizando página {i+1}/{len(enlaces[:max_paginas])}: {enlace}"):
                    texto = extraer_texto(enlace)
                    if texto:
                        resultado = analizar_y_estructurar_contenido(texto, enlace)
                        contenido_final += f"## Página: {enlace}\n\n{resultado}\n\n---\n\n"
                        st.write(f"✅ Contenido estructurado para: {enlace}")
                    else:
                        st.write(f"⚠️ No se pudo extraer texto de: {enlace}")
                barra.progress((i + 1) / max_paginas)

        st.success("🎯 Escaneo completado")
        st.markdown("---")
        st.header("📄 Contenido Estructurado")
        st.markdown(contenido_final)
    else:
        st.error("Por favor, introduce una URL válida.")

