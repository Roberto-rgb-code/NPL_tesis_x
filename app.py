import streamlit as st
import pandas as pd
from google.cloud import language_v1
from google.api_core.exceptions import InvalidArgument, BadRequest
import os

# ---- Configuración de la aplicación (must be the first Streamlit command) ----
st.set_page_config(page_title="Análisis NLP con Google Cloud", layout="wide")

# ---- Título y subtítulo ----
st.title("Análisis de sentimientos en X para la comprensión multidimensional del mundo socio digital")
st.subheader("Procesamiento de lenguaje natural")

st.sidebar.header("Autenticación a Google Cloud")

# Modo de autenticación: JSON o API Key
auth_method = st.sidebar.radio(
    "Selecciona método de autenticación:",
    options=["Service Account JSON", "API Key"]
)
service_account_path = None
api_key = None
if auth_method == "Service Account JSON":
    service_account_path = st.sidebar.text_input(
        "Ruta a tu clave JSON de servicio", value="credentials/service_account.json"
    )
    if service_account_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
elif auth_method == "API Key":
    api_key = st.sidebar.text_input(
        "Tu API Key de Cloud Natural Language",
        value="AIzaSyAang0-iMQFSMngx1WlSzzkoFpcHQ26zLQ"
    )

# ---- Carga de datos ----
st.sidebar.header("Carga de datos")
file_type = st.sidebar.selectbox("Selecciona el tipo de archivo", ["CSV", "Excel", "JSON"])
data_file = st.sidebar.file_uploader(f"Sube tu archivo {file_type}", type=["csv", "xlsx", "xls", "json"])

if data_file:
    try:
        if file_type == "CSV":
            df = pd.read_csv(data_file)
        elif file_type == "Excel":
            df = pd.read_excel(data_file)
        elif file_type == "JSON":
            df = pd.read_json(data_file)
        else:
            st.error("Tipo de archivo no soportado. Por favor, selecciona CSV, Excel o JSON.")
            st.stop()
        
        # Verificar columna 'Text'
        if 'Text' not in df.columns:
            st.error("El archivo no contiene la columna 'Text'.")
            st.stop()
        st.success(f"Datos cargados: {len(df)} registros")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}. Asegúrate de que tenga la estructura correcta.")
        st.stop()
    
    # Selección de la fila a analizar
    st.header("Selecciona la fila a analizar")
    row_index = st.number_input(
        "Ingresa el índice de la fila a analizar:",
        min_value=0,
        max_value=len(df)-1,
        step=1
    )
    
    if row_index not in df.index:
        st.error(f"El índice {row_index} no existe en el DataFrame.")
        st.stop()
    
    texto_seleccionado = df.loc[row_index, 'Text']
    st.write(f"Texto de la fila {row_index}:")
    st.write(texto_seleccionado)
    
    # Selección de funciones de NLP
    analisis = st.multiselect(
        "Selecciona funciones de NLP a aplicar:",
        options=[
            'Sentiment Analysis',
            'Entity Recognition',
            'Entity Sentiment',
            'Content Classification',
            'Moderate Text'
        ],
        default=['Sentiment Analysis', 'Entity Recognition']
    )
    
    # Ejecutar análisis
    if st.button("Ejecutar análisis"):
        # Crear cliente según método de autenticación
        client_args = {}
        if auth_method == "API Key" and api_key:
            client_args = {"client_options": {"api_key": api_key}}
        client = language_v1.LanguageServiceClient(**client_args)
        
        doc = language_v1.Document(
            content=texto_seleccionado, type_=language_v1.Document.Type.PLAIN_TEXT
        )
        res = {'text': texto_seleccionado}
        
        try:
            with st.spinner("Analizando..."):
                if 'Sentiment Analysis' in analisis:
                    try:
                        sent = client.analyze_sentiment(request={'document': doc})
                        res['document_sentiment'] = {
                            'score': sent.document_sentiment.score,
                            'magnitude': sent.document_sentiment.magnitude
                        }
                        res['sentences'] = [{
                            'text': s.text.content,
                            'score': s.sentiment.score,
                            'magnitude': s.sentiment.magnitude
                        } for s in sent.sentences]
                    except BadRequest as e:
                        st.warning(f"Error en analyze_sentiment: {e}")
                
                if 'Entity Recognition' in analisis:
                    try:
                        ents = client.analyze_entities(request={'document': doc}).entities
                        res['entities'] = [{'name': e.name, 'type': e.type_.name} for e in ents]
                    except BadRequest as e:
                        st.warning(f"Error en analyze_entities: {e}")
                
                if 'Entity Sentiment' in analisis:
                    try:
                        ents_sent = client.analyze_entity_sentiment(request={'document': doc}).entities
                        if 'entities' in res:
                            for e in ents_sent:
                                for ent in res['entities']:
                                    if ent['name'] == e.name:
                                        ent['sentiment'] = e.sentiment.score
                        else:
                            res['entities'] = [{
                                'name': e.name,
                                'type': e.type_.name,
                                'sentiment': e.sentiment.score
                            } for e in ents_sent]
                    except BadRequest as e:
                        st.warning(f"Error en analyze_entity_sentiment: {e}")
                
                if 'Content Classification' in analisis:
                    try:
                        cats = client.classify_text(request={'document': doc}).categories
                        res['categories'] = [{
                            'name': c.name,
                            'confidence': c.confidence
                        } for c in cats]
                    except BadRequest as e:
                        st.warning(f"Error en classify_text: {e}")
                
                if 'Moderate Text' in analisis:
                    try:
                        mod = client.moderate_text(request={'document': doc}).moderation_categories
                        res['moderation'] = [{
                            'category': m.name,
                            'confidence': m.confidence
                        } for m in mod]
                    except BadRequest as e:
                        st.warning(f"Error en moderate_text: {e}")
                
                # Define entity colors based on documentation examples
                entity_colors = {
                    "PERSON": "blue",
                    "LOCATION": "green",
                    "ORGANIZATION": "purple",
                    "EVENT": "orange",
                    "WORK_OF_ART": "red",
                    "CONSUMER_GOOD": "brown",
                    "NUMBER": "blue",
                    "ADDRESS": "pink",
                    "PRICE": "red",
                    "OTHER": "gray"
                }
                
                # CSS styles to mimic documentation UI
                st.markdown("""
                <style>
                .entity-card, .moderation-card, .category-card {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .entity-name, .category-name {
                    font-weight: bold;
                    margin-right: 10px;
                }
                .entity-type {
                    color: white;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-size: 12px;
                    display: inline-block;
                }
                .confidence, .sentiment-score {
                    float: right;
                    color: #555;
                }
                .sentiment-box {
                    padding: 5px 10px;
                    border-radius: 3px;
                    color: white;
                    display: inline-block;
                    margin-right: 10px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Create tabs
                st.subheader("Resultados del análisis")
                tab1, tab2, tab3, tab4 = st.tabs(["Entities", "Sentiment", "Moderation", "Categories"])
                
                # Entities Tab
                with tab1:
                    if 'entities' in res:
                        for entity in res['entities']:
                            color = entity_colors.get(entity['type'], 'gray')
                            sentiment_str = f"<span class='sentiment-score'>Sentiment: {entity['sentiment']:.2f}</span>" if 'sentiment' in entity else ""
                            st.markdown(f"""
                            <div class="entity-card">
                                <span class="entity-name">{entity['name']}</span>
                                <span class="entity-type" style="background-color: {color};">{entity['type']}</span>
                                {sentiment_str}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No entity recognition performed.")
                
                # Sentiment Tab
                with tab2:
                    if 'document_sentiment' in res:
                        st.subheader("Document and Sentence Level Sentiment")
                        st.write("**Entire Document:**")
                        st.write(res['text'])
                        st.markdown(f"""
                        <div>
                            <span class="sentiment-box" style="background-color: green;">Score: {res['document_sentiment']['score']:.3f}</span>
                            <span class="sentiment-box" style="background-color: blue;">Magnitude: {res['document_sentiment']['magnitude']:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if 'sentences' in res and res['sentences']:
                            st.write("**Sentences:**")
                            for sentence in res['sentences']:
                                st.write(sentence['text'])
                                st.markdown(f"""
                                <div>
                                    <span class="sentiment-box" style="background-color: green;">Score: {sentence['score']:.3f}</span>
                                    <span class="sentiment-box" style="background-color: blue;">Magnitude: {sentence['magnitude']:.3f}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        st.subheader("Score Range")
                        st.markdown("""
                        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                            <div style="background-color: green; width: 33%; height: 20px;"></div>
                            <div style="background-color: yellow; width: 33%; height: 20px;"></div>
                            <div style="background-color: red; width: 33%; height: 20px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Positive (0.25 - 1.0)</span>
                            <span>Neutral (-0.25 - 0.25)</span>
                            <span>Negative (-1.0 - -0.25)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write("No sentiment analysis performed.")
                
                # Moderation Tab
                with tab3:
                    if 'moderation' in res:
                        for mod in res['moderation']:
                            st.markdown(f"""
                            <div class="moderation-card">
                                <span class="category-name">{mod['category']}</span>
                                <span class="confidence">Confidence: {mod['confidence']:.6f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No moderation analysis performed.")
                
                # Categories Tab
                with tab4:
                    if 'categories' in res:
                        for cat in res['categories']:
                            st.markdown(f"""
                            <div class="category-card">
                                <span class="category-name">{cat['name']}</span>
                                <span class="confidence">Confidence: {cat['confidence']:.6f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("[See complete list of categories](https://cloud.google.com/natural-language/docs/categories)", unsafe_allow_html=True)
                    else:
                        st.write("No content classification performed.")
        
        except InvalidArgument as e:
            st.error(
                "Error de API: clave inválida o expirada. "
                "Por favor, renueva tu API Key o usa un Service Account JSON válido."
            )
            st.stop()
else:
    st.info("Por favor, sube un archivo CSV, Excel o JSON para comenzar.")
