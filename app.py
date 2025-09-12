import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- Configuración de la Página de Streamlit ---
st.set_page_config(page_title="Agente Web Scraper IA", page_icon="🤖")
st.title("🤖 Agente IA Web Scraper")
st.caption("Introduce la URL de un sitio web para extraer y ESTRUCTURAR su contenido.")

# --- Configuración del Modelo de Lenguaje (LLM) ---
# Para despliegue en Streamlit Cloud, usa st.secrets
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("Error al configurar el modelo de IA. Asegúrate de que tu GOOGLE_API_KEY esté configurada en los secretos de Streamlit.")
    st.stop()


# --- Funciones de Web Scraping ---

def es_valido(url):
    """Verifica si una URL es válida."""
    parsed = urlparse(url)
    esquema_valido = bool(parsed.netloc) and bool(parsed.scheme)
    extensiones_excluidas = ['.pdf', '.jpg', '.png', '.zip', '.docx', '.gif', '.mp3', '.mp4']
    no_es_archivo = not any(parsed.path.lower().endswith(ext) for ext in extensiones_excluidas)
    return esquema_valido and no_es_archivo

def obtener_enlaces_pagina(url):
    """Obtiene todos los enlaces válidos de una página web."""
    urls = set()
    try:
        respuesta = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        respuesta.raise_for_status()
        sopa = BeautifulSoup(respuesta.content, 'html.parser')
        dominio_base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        for link in sopa.find_all('a', href=True):
            href = link['href']
            url_absoluta = urljoin(dominio_base, href)
            if urlparse(url_absoluta).netloc == urlparse(dominio_base).netloc and es_valido(url_absoluta):
                urls.add(url_absoluta)
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        st.warning(f"No se pudo acceder a {url}: {e}")
    return urls

def extraer_texto(url):
    """Extrae el texto principal de una página web."""
    try:
        respuesta = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        respuesta.raise_for_status()
        sopa = BeautifulSoup(respuesta.content, 'html.parser')
        for script_o_estilo in sopa(["script", "style"]):
            script_o_estilo.decompose()
        texto = " ".join(t.strip() for t in sopa.stripped_strings)
        return texto
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        st.warning(f"No se pudo extraer texto de {url}: {e}")
        return ""

# --- Lógica del Agente con LangChain (Versión que no resume) ---

def analizar_y_estructurar_contenido(texto_pagina, url):
    """
    Utiliza el LLM para ESTRUCTURAR todo el contenido de una página, SIN RESUMIR.
    """
    if not texto_pagina:
        return "No se encontró contenido textual para analizar."

    plantilla_nueva = """
    Eres un asistente de IA experto en analizar y documentar contenido web de forma exhaustiva.
    Tu tarea es procesar el texto extraído de la URL '{url}' y presentarlo de forma completa y bien organizada. **No debes resumir, acortar ni omitir información relevante.**

    Realiza las siguientes acciones con el texto proporcionado:
    1.  Genera un título descriptivo que refleje el contenido principal de la página.
    2.  Identifica las diferentes secciones, temas o apartados principales del texto.
    3.  Para cada sección identificada, extrae y presenta TODA la información correspondiente. Usa subtítulos en formato Markdown (ej: ## Título de la Sección) para separar cada tema.
    4.  El resultado final debe ser una transcripción fiel y estructurada del contenido original, ideal para ser guardada en un documento.

    Contenido de la página:
    ---
    {texto}
    ---

    Presenta el resultado completo y estructurado.
    """
    prompt = ChatPromptTemplate.from_template(plantilla_nueva)
    cadena = prompt | llm | StrOutputParser()

    # Se limita la cantidad de texto por si la página es extremadamente larga
    texto_limitado = texto_pagina[:30000]

    return cadena.invoke({"texto": texto_limitado, "url": url})


# --- Interfaz de Usuario de Streamlit ---

url_usuario = st.text_input("🔗 Introduce la URL del sitio web a escanear:", placeholder="https://ejemplo.com")

if st.button("🚀 Iniciar Escaneo"):
    if url_usuario and es_valido(url_usuario):
        with st.spinner('🔍 Buscando enlaces en la página principal...'):
            enlaces_a_visitar = list(obtener_enlaces_pagina(url_usuario))
            enlaces_a_visitar.insert(0, url_usuario)
            visitados = set()
            contenido_final = f"# Reporte de Contenido Web: {urlparse(url_usuario).netloc}\n\n"

        max_paginas = st.slider("Número máximo de páginas a analizar:", 1, 20, 5)
        barra_progreso = st.progress(0)

        for i, enlace in enumerate(enlaces_a_visitar[:max_paginas]):
            if enlace not in visitados:
                visitados.add(enlace)
                with st.spinner(f"Analizando página {i+1}/{len(enlaces_a_visitar[:max_paginas])}: {enlace}"):
                    texto = extraer_texto(enlace)
                    if texto:
                        # Llamada a la nueva función que ESTRUCTURA en vez de resumir
                        contenido_estructurado = analizar_y_estructurar_contenido(texto, enlace)
                        contenido_final += f"## Página Analizada: {enlace}\n\n{contenido_estructurado}\n\n---\n\n"
                        st.write(f"✅ Contenido estructurado para: {enlace}")
                    else:
                        st.write(f"⚠️ No se pudo extraer texto de: {enlace}")

                barra_progreso.progress((i + 1) / max_paginas)

        st.success("¡Escaneo completado!")
        st.markdown("---")
        st.header("📄 Contenido Estructurado del Sitio Web")
        st.markdown(contenido_final)
    else:
        st.error("Por favor, introduce una URL válida.")

