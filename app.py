# ============================================================
# CLASS 2.0 — NEUROLANDSCAPE
# COMPLETE STREAMLIT APPLICATION
# Single-cell / single-file version
# ============================================================

# Dependencies:
# pip install --upgrade streamlit torch torchvision
# pip install transformers huggingface_hub pillow pandas numpy requests
#
# Run:
# streamlit run app.py

import io
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from huggingface_hub import hf_hub_download


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

HF_REPO = "DOReilly2/swin_regressor"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Replace with the actual raw GitHub URL of your logo.
LOGO_URL = (
    "https://raw.githubusercontent.com/"
    "davidoreilly12/CLASS-2.0/main/neurolandscape_logo.png"
)

APP_TITLE = "CLASS 2.0"

# Exact model labels used by the original checkpoint.
# Do not remove the "_embedding" suffix.
MODEL_LABELS = [
    "Layers of the Landscape_embedding",
    "Landform_embedding",
    "Biodiversity_embedding",
    "Color and Light_embedding",
    "Compatibility_embedding",
    "Archetypal Elements_embedding",
    "Character of Peace and Silence_embedding",
]

# Display labels.
DIMENSION_LABELS = [
    "Layers of the Landscape",
    "Landform",
    "Biodiversity",
    "Color and Light",
    "Compatibility",
    "Archetypal Elements",
    "Character of Peace and Silence",
]


# ============================================================
# 2. LOGO
# ============================================================

@st.cache_data(show_spinner=False)
def load_logo(url):

    if not url or "YOUR_USERNAME" in url:
        return None

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        return Image.open(
            io.BytesIO(response.content)
        ).convert("RGBA")

    except Exception:
        return None


logo_image = load_logo(LOGO_URL)


# ============================================================
# 3. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=logo_image if logo_image is not None else "🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 4. LANGUAGES
# ============================================================

LANGUAGES = {
    "🇬🇧 English": "en",
    "🇫🇷 Français": "fr",
    "🇮🇹 Italiano": "it",
    "🇩🇪 Deutsch": "de",
    "🇸🇪 Svenska": "sv",
    "🇨🇳 中文": "zh",
    "🇰🇷 한국어": "ko",
    "🇪🇸 Español": "es",
    "🇵🇹 Português": "pt",
    "🇩🇰 Dansk": "da",
    "🇳🇱 Nederlands": "nl",
    "🇷🇺 Русский": "ru",
}

# Compatibility with language values stored by previous versions.
LANGUAGE_ALIASES = {
    "English": "🇬🇧 English",
    "French": "🇫🇷 Français",
    "Italian": "🇮🇹 Italiano",
    "German": "🇩🇪 Deutsch",
    "Swedish": "🇸🇪 Svenska",
    "Chinese": "🇨🇳 中文",
    "Korean": "🇰🇷 한국어",
    "Spanish": "🇪🇸 Español",
    "Portuguese": "🇵🇹 Português",
    "Danish": "🇩🇰 Dansk",
    "Dutch": "🇳🇱 Nederlands",
    "Russian": "🇷🇺 Русский",
}


# ============================================================
# 5. TRANSLATIONS
# ============================================================

TRANSLATIONS = {

    "en": {
        "language": "Language",
        "description": "Assess landscape characteristics using the CLASS 2.0 artificial intelligence model.",
        "upload_title": "Upload landscape images",
        "upload_help": "Upload one or more JPG, JPEG or PNG images.",
        "upload_label": "Choose landscape images",
        "results_title": "Assessment results",
        "results_description": "The following scores represent the seven landscape dimensions assessed by CLASS 2.0.",
        "image": "Image",
        "total_score": "Overall landscape score",
        "download": "Download results as CSV",
        "processing": "Processing landscape images...",
        "loading_model": "Loading CLASS 2.0 model...",
        "model_ready": "CLASS 2.0 model loaded successfully.",
        "model_error": "Unable to load the CLASS 2.0 model.",
        "no_images": "Upload images to begin the assessment.",
        "error_image": "Could not process image",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Landscape dimensions",
        "number_images": "Number of images assessed",
        "average_score": "Mean landscape score",
        "individual_results": "Individual image results",
        "score_range": "Score range: 1–6",
    },

    "fr": {
        "language": "Langue",
        "description": "Évaluez les caractéristiques du paysage à l'aide du modèle d'intelligence artificielle CLASS 2.0.",
        "upload_title": "Télécharger des images de paysage",
        "upload_help": "Téléchargez une ou plusieurs images JPG, JPEG ou PNG.",
        "upload_label": "Choisir des images de paysage",
        "results_title": "Résultats de l'évaluation",
        "results_description": "Les scores représentent les sept dimensions du paysage évaluées par CLASS 2.0.",
        "image": "Image",
        "total_score": "Score global du paysage",
        "download": "Télécharger les résultats au format CSV",
        "processing": "Analyse des images de paysage...",
        "loading_model": "Chargement du modèle CLASS 2.0...",
        "model_ready": "Modèle CLASS 2.0 chargé avec succès.",
        "model_error": "Impossible de charger le modèle CLASS 2.0.",
        "no_images": "Téléchargez des images pour commencer l'évaluation.",
        "error_image": "Impossible de traiter l'image",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Dimensions du paysage",
        "number_images": "Nombre d'images évaluées",
        "average_score": "Score moyen du paysage",
        "individual_results": "Résultats individuels des images",
        "score_range": "Échelle des scores : 1–6",
    },

    "it": {
        "language": "Lingua",
        "description": "Valuta le caratteristiche del paesaggio utilizzando il modello di intelligenza artificiale CLASS 2.0.",
        "upload_title": "Carica immagini del paesaggio",
        "upload_help": "Carica una o più immagini JPG, JPEG o PNG.",
        "upload_label": "Seleziona immagini del paesaggio",
        "results_title": "Risultati della valutazione",
        "results_description": "I punteggi rappresentano le sette dimensioni del paesaggio valutate da CLASS 2.0.",
        "image": "Immagine",
        "total_score": "Punteggio complessivo del paesaggio",
        "download": "Scarica i risultati in formato CSV",
        "processing": "Elaborazione delle immagini del paesaggio...",
        "loading_model": "Caricamento del modello CLASS 2.0...",
        "model_ready": "Modello CLASS 2.0 caricato correttamente.",
        "model_error": "Impossibile caricare il modello CLASS 2.0.",
        "no_images": "Carica immagini per iniziare la valutazione.",
        "error_image": "Impossibile elaborare l'immagine",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Dimensioni del paesaggio",
        "number_images": "Numero di immagini valutate",
        "average_score": "Punteggio medio del paesaggio",
        "individual_results": "Risultati delle singole immagini",
        "score_range": "Intervallo dei punteggi: 1–6",
    },

    "de": {
        "language": "Sprache",
        "description": "Bewerten Sie Landschaftsmerkmale mit dem künstlichen Intelligenzmodell CLASS 2.0.",
        "upload_title": "Landschaftsbilder hochladen",
        "upload_help": "Laden Sie ein oder mehrere JPG-, JPEG- oder PNG-Bilder hoch.",
        "upload_label": "Landschaftsbilder auswählen",
        "results_title": "Bewertungsergebnisse",
        "results_description": "Die Werte repräsentieren die sieben von CLASS 2.0 bewerteten Landschaftsdimensionen.",
        "image": "Bild",
        "total_score": "Gesamtbewertung der Landschaft",
        "download": "Ergebnisse als CSV herunterladen",
        "processing": "Landschaftsbilder werden verarbeitet...",
        "loading_model": "CLASS 2.0-Modell wird geladen...",
        "model_ready": "CLASS 2.0-Modell erfolgreich geladen.",
        "model_error": "Das CLASS 2.0-Modell konnte nicht geladen werden.",
        "no_images": "Laden Sie Bilder hoch, um die Bewertung zu starten.",
        "error_image": "Bild konnte nicht verarbeitet werden",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Landschaftsdimensionen",
        "number_images": "Anzahl bewerteter Bilder",
        "average_score": "Durchschnittliche Landschaftsbewertung",
        "individual_results": "Einzelergebnisse der Bilder",
        "score_range": "Bewertungsskala: 1–6",
    },

    "sv": {
        "language": "Språk",
        "description": "Bedöm landskapets egenskaper med hjälp av CLASS 2.0-modellen för artificiell intelligens.",
        "upload_title": "Ladda upp landskapsbilder",
        "upload_help": "Ladda upp en eller flera JPG-, JPEG- eller PNG-bilder.",
        "upload_label": "Välj landskapsbilder",
        "results_title": "Bedömningsresultat",
        "results_description": "Poängen representerar de sju landskapsdimensioner som bedöms av CLASS 2.0.",
        "image": "Bild",
        "total_score": "Övergripande landskapspoäng",
        "download": "Ladda ner resultat som CSV",
        "processing": "Bearbetar landskapsbilder...",
        "loading_model": "Laddar CLASS 2.0-modellen...",
        "model_ready": "CLASS 2.0-modellen har laddats.",
        "model_error": "Det gick inte att ladda CLASS 2.0-modellen.",
        "no_images": "Ladda upp bilder för att börja bedömningen.",
        "error_image": "Det gick inte att bearbeta bilden",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Landskapsdimensioner",
        "number_images": "Antal bedömda bilder",
        "average_score": "Genomsnittlig landskapspoäng",
        "individual_results": "Resultat för enskilda bilder",
        "score_range": "Poängskala: 1–6",
    },

    "zh": {
        "language": "语言",
        "description": "使用 CLASS 2.0 人工智能模型评估景观特征。",
        "upload_title": "上传景观图像",
        "upload_help": "上传一张或多张 JPG、JPEG 或 PNG 图像。",
        "upload_label": "选择景观图像",
        "results_title": "评估结果",
        "results_description": "以下分数代表 CLASS 2.0 评估的七个景观维度。",
        "image": "图像",
        "total_score": "景观综合得分",
        "download": "下载 CSV 格式的结果",
        "processing": "正在处理景观图像...",
        "loading_model": "正在加载 CLASS 2.0 模型...",
        "model_ready": "CLASS 2.0 模型加载成功。",
        "model_error": "无法加载 CLASS 2.0 模型。",
        "no_images": "请上传图像以开始评估。",
        "error_image": "无法处理图像",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "景观维度",
        "number_images": "已评估图像数量",
        "average_score": "平均景观得分",
        "individual_results": "单张图像评估结果",
        "score_range": "评分范围：1–6",
    },

    "ko": {
        "language": "언어",
        "description": "CLASS 2.0 인공지능 모델을 사용하여 경관 특성을 평가합니다.",
        "upload_title": "경관 이미지 업로드",
        "upload_help": "하나 이상의 JPG, JPEG 또는 PNG 이미지를 업로드하세요.",
        "upload_label": "경관 이미지 선택",
        "results_title": "평가 결과",
        "results_description": "다음 점수는 CLASS 2.0이 평가하는 7가지 경관 차원을 나타냅니다.",
        "image": "이미지",
        "total_score": "종합 경관 점수",
        "download": "CSV 결과 다운로드",
        "processing": "경관 이미지 처리 중...",
        "loading_model": "CLASS 2.0 모델 로딩 중...",
        "model_ready": "CLASS 2.0 모델이 성공적으로 로드되었습니다.",
        "model_error": "CLASS 2.0 모델을 로드할 수 없습니다.",
        "no_images": "평가를 시작하려면 이미지를 업로드하세요.",
        "error_image": "이미지를 처리할 수 없습니다",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "경관 차원",
        "number_images": "평가된 이미지 수",
        "average_score": "평균 경관 점수",
        "individual_results": "개별 이미지 결과",
        "score_range": "점수 범위: 1–6",
    },

    "es": {
        "language": "Idioma",
        "description": "Evalúe las características del paisaje mediante el modelo de inteligencia artificial CLASS 2.0.",
        "upload_title": "Subir imágenes del paisaje",
        "upload_help": "Suba una o varias imágenes JPG, JPEG o PNG.",
        "upload_label": "Seleccionar imágenes del paisaje",
        "results_title": "Resultados de la evaluación",
        "results_description": "Las puntuaciones representan las siete dimensiones del paisaje evaluadas por CLASS 2.0.",
        "image": "Imagen",
        "total_score": "Puntuación general del paisaje",
        "download": "Descargar resultados en CSV",
        "processing": "Procesando imágenes del paisaje...",
        "loading_model": "Cargando el modelo CLASS 2.0...",
        "model_ready": "Modelo CLASS 2.0 cargado correctamente.",
        "model_error": "No se pudo cargar el modelo CLASS 2.0.",
        "no_images": "Suba imágenes para comenzar la evaluación.",
        "error_image": "No se pudo procesar la imagen",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Dimensiones del paisaje",
        "number_images": "Número de imágenes evaluadas",
        "average_score": "Puntuación media del paisaje",
        "individual_results": "Resultados individuales de las imágenes",
        "score_range": "Rango de puntuación: 1–6",
    },

    "pt": {
        "language": "Idioma",
        "description": "Avalie as características da paisagem utilizando o modelo de inteligência artificial CLASS 2.0.",
        "upload_title": "Carregar imagens da paisagem",
        "upload_help": "Carregue uma ou mais imagens JPG, JPEG ou PNG.",
        "upload_label": "Selecionar imagens da paisagem",
        "results_title": "Resultados da avaliação",
        "results_description": "As pontuações representam as sete dimensões da paisagem avaliadas pelo CLASS 2.0.",
        "image": "Imagem",
        "total_score": "Pontuação geral da paisagem",
        "download": "Descarregar resultados em CSV",
        "processing": "A processar imagens da paisagem...",
        "loading_model": "A carregar o modelo CLASS 2.0...",
        "model_ready": "Modelo CLASS 2.0 carregado com sucesso.",
        "model_error": "Não foi possível carregar o modelo CLASS 2.0.",
        "no_images": "Carregue imagens para iniciar a avaliação.",
        "error_image": "Não foi possível processar a imagem",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Dimensões da paisagem",
        "number_images": "Número de imagens avaliadas",
        "average_score": "Pontuação média da paisagem",
        "individual_results": "Resultados individuais das imagens",
        "score_range": "Intervalo de pontuação: 1–6",
    },

    "da": {
        "language": "Sprog",
        "description": "Vurder landskabets egenskaber ved hjælp af CLASS 2.0-modellen for kunstig intelligens.",
        "upload_title": "Upload landskabsbilleder",
        "upload_help": "Upload et eller flere JPG-, JPEG- eller PNG-billeder.",
        "upload_label": "Vælg landskabsbilleder",
        "results_title": "Vurderingsresultater",
        "results_description": "Resultaterne repræsenterer de syv landskabsdimensioner, der vurderes af CLASS 2.0.",
        "image": "Billede",
        "total_score": "Samlet landskabsscore",
        "download": "Download resultater som CSV",
        "processing": "Behandler landskabsbilleder...",
        "loading_model": "Indlæser CLASS 2.0-modellen...",
        "model_ready": "CLASS 2.0-modellen er indlæst.",
        "model_error": "CLASS 2.0-modellen kunne ikke indlæses.",
        "no_images": "Upload billeder for at starte vurderingen.",
        "error_image": "Billedet kunne ikke behandles",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Landskabsdimensioner",
        "number_images": "Antal vurderede billeder",
        "average_score": "Gennemsnitlig landskabsscore",
        "individual_results": "Individuelle billedresultater",
        "score_range": "Scoreskala: 1–6",
    },

    "nl": {
        "language": "Taal",
        "description": "Beoordeel landschapskenmerken met behulp van het kunstmatige-intelligentiemodel CLASS 2.0.",
        "upload_title": "Landschapsafbeeldingen uploaden",
        "upload_help": "Upload een of meer JPG-, JPEG- of PNG-afbeeldingen.",
        "upload_label": "Landschapsafbeeldingen selecteren",
        "results_title": "Beoordelingsresultaten",
        "results_description": "De scores vertegenwoordigen de zeven landschapsdimensies die door CLASS 2.0 worden beoordeeld.",
        "image": "Afbeelding",
        "total_score": "Algemene landschapsscore",
        "download": "Resultaten downloaden als CSV",
        "processing": "Landschapsafbeeldingen verwerken...",
        "loading_model": "CLASS 2.0-model laden...",
        "model_ready": "CLASS 2.0-model succesvol geladen.",
        "model_error": "Het CLASS 2.0-model kon niet worden geladen.",
        "no_images": "Upload afbeeldingen om de beoordeling te starten.",
        "error_image": "Afbeelding kon niet worden verwerkt",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Landschapsdimensies",
        "number_images": "Aantal beoordeelde afbeeldingen",
        "average_score": "Gemiddelde landschapsscore",
        "individual_results": "Individuele afbeeldingsresultaten",
        "score_range": "Scorebereik: 1–6",
    },

    "ru": {
        "language": "Язык",
        "description": "Оцените характеристики ландшафта с помощью модели искусственного интеллекта CLASS 2.0.",
        "upload_title": "Загрузить изображения ландшафта",
        "upload_help": "Загрузите одно или несколько изображений JPG, JPEG или PNG.",
        "upload_label": "Выбрать изображения ландшафта",
        "results_title": "Результаты оценки",
        "results_description": "Оценки отражают семь измерений ландшафта, анализируемых CLASS 2.0.",
        "image": "Изображение",
        "total_score": "Общая оценка ландшафта",
        "download": "Скачать результаты в формате CSV",
        "processing": "Обработка изображений ландшафта...",
        "loading_model": "Загрузка модели CLASS 2.0...",
        "model_ready": "Модель CLASS 2.0 успешно загружена.",
        "model_error": "Не удалось загрузить модель CLASS 2.0.",
        "no_images": "Загрузите изображения, чтобы начать оценку.",
        "error_image": "Не удалось обработать изображение",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Измерения ландшафта",
        "number_images": "Количество оценённых изображений",
        "average_score": "Средняя оценка ландшафта",
        "individual_results": "Результаты отдельных изображений",
        "score_range": "Диапазон оценок: 1–6",
    },
}


# ============================================================
# 6. TRANSLATED DIMENSION NAMES
# ============================================================

DIMENSION_TRANSLATIONS = {

    "en": DIMENSION_LABELS,

    "fr": [
        "Couches du paysage",
        "Forme du relief",
        "Biodiversité",
        "Couleur et lumière",
        "Compatibilité",
        "Éléments archétypaux",
        "Caractère de paix et de silence",
    ],

    "it": [
        "Strati del paesaggio",
        "Morfologia del terreno",
        "Biodiversità",
        "Colore e luce",
        "Compatibilità",
        "Elementi archetipici",
        "Carattere di pace e silenzio",
    ],

    "de": [
        "Landschaftsschichten",
        "Landform",
        "Biodiversität",
        "Farbe und Licht",
        "Kompatibilität",
        "Archetypische Elemente",
        "Charakter von Frieden und Stille",
    ],

    "sv": [
        "Landskapets lager",
        "Landform",
        "Biologisk mångfald",
        "Färg och ljus",
        "Kompatibilitet",
        "Arketypiska element",
        "Karaktär av frid och tystnad",
    ],

    "zh": [
        "景观层次",
        "地貌",
        "生物多样性",
        "色彩与光线",
        "兼容性",
        "原型元素",
        "宁静与静谧的特征",
    ],

    "ko": [
        "경관의 층위",
        "지형",
        "생물다양성",
        "색상과 빛",
        "호환성",
        "원형적 요소",
        "평화와 고요의 특성",
    ],

    "es": [
        "Capas del paisaje",
        "Forma del terreno",
        "Biodiversidad",
        "Color y luz",
        "Compatibilidad",
        "Elementos arquetípicos",
        "Carácter de paz y silencio",
    ],

    "pt": [
        "Camadas da paisagem",
        "Forma do relevo",
        "Biodiversidade",
        "Cor e luz",
        "Compatibilidade",
        "Elementos arquetípicos",
        "Caráter de paz e silêncio",
    ],

    "da": [
        "Landskabets lag",
        "Landform",
        "Biodiversitet",
        "Farve og lys",
        "Kompatibilitet",
        "Arketypiske elementer",
        "Karakter af fred og stilhed",
    ],

    "nl": [
        "Landschapslagen",
        "Landvorm",
        "Biodiversiteit",
        "Kleur en licht",
        "Compatibiliteit",
        "Archetypische elementen",
        "Karakter van rust en stilte",
    ],

    "ru": [
        "Слои ландшафта",
        "Форма рельефа",
        "Биоразнообразие",
        "Цвет и свет",
        "Совместимость",
        "Архетипические элементы",
        "Характер покоя и тишины",
    ],
}


# ============================================================
# 7. FIX LANGUAGE SESSION STATE
# ============================================================

# This is the important fix for:
# KeyError: 'English'
#
# Previous application versions may have stored "English",
# whereas the current language options use "🇬🇧 English".

stored_language = st.session_state.get("language", "🇬🇧 English")

if stored_language in LANGUAGE_ALIASES:
    stored_language = LANGUAGE_ALIASES[stored_language]

if stored_language not in LANGUAGES:
    stored_language = "🇬🇧 English"

# Migrate old session-state value before creating the widget.
st.session_state["language"] = stored_language


# ============================================================
# 8. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #FFFFFF;
        color: #17324D;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
    }

    h1, h2, h3 {
        color: #17324D;
    }

    [data-testid="stFileUploader"] section {
        background-color: #F7F9FB;
        border: 1px dashed #A9B9C8;
        border-radius: 10px;
    }

    .stDownloadButton button {
        background-color: #17324D;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.2rem;
    }

    .stDownloadButton button:hover {
        background-color: #284F70;
        color: white;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8EF;
        border-radius: 8px;
    }

    [data-testid="stMetric"] {
        background-color: #F7F9FB;
        border: 1px solid #E2E8EF;
        padding: 15px;
        border-radius: 8px;
    }

    [data-testid="stMetricLabel"] {
        color: #526477;
    }

    [data-testid="stMetricValue"] {
        color: #17324D;
    }

    .footer {
        text-align: center;
        color: #8493A2;
        font-size: 13px;
        padding-top: 35px;
        padding-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. HEADER AND LANGUAGE SELECTOR
# ============================================================

logo_col, title_col, language_col = st.columns(
    [1.5, 4.5, 2],
    vertical_alignment="center",
)

with logo_col:

    if logo_image is not None:
        st.image(logo_image, width=190)
    else:
        st.markdown(
            """
            <div style="
                font-size: 19px;
                font-weight: 700;
                color: #17324D;
            ">
                NeuroLandscape
            </div>
            """,
            unsafe_allow_html=True,
        )


with title_col:

    st.html(
        """
        <div style="padding-top: 2px; padding-bottom: 4px;">
            <h1 style="
                font-size: 42px;
                font-weight: 750;
                color: #17324D;
                margin: 0;
                padding: 0;
                line-height: 1.15;
            ">
                CLASS 2.0
            </h1>

            <p style="
                font-size: 17px;
                color: #526477;
                margin-top: 5px;
                margin-bottom: 0;
            ">
                NeuroLandscape | Contemplative Landscape Automated Scoring System
            </p>
        </div>
        """
    )


with language_col:

    st.selectbox(
        "Language / Sprache / Langue",
        options=list(LANGUAGES.keys()),
        key="language",
    )


# IMPORTANT:
# Read the language AFTER the selectbox has been rendered.
# This ensures the current selection is used on every rerun.

selected_language = st.session_state["language"]
lang_code = LANGUAGES[selected_language]

T = TRANSLATIONS[lang_code]
translated_dimensions = DIMENSION_TRANSLATIONS[lang_code]


# ============================================================
# 10. INTRODUCTION
# ============================================================

st.divider()

st.markdown(
    f"""
    <div style="
        font-size: 16px;
        color: #526477;
        margin-bottom: 20px;
    ">
        {T["description"]}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 11. IMAGE PREPROCESSING
# ============================================================

preprocess_image = transforms.Compose([

    transforms.Resize((512, 512)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

])


# ============================================================
# 12. LOAD CONTEXT EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_context_embeddings():

    context_embeddings = {}

    for label in MODEL_LABELS:

        # Original Hugging Face filename convention.
        filename = f"context_embeddings/{label}.pt"

        embedding_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
        )

        embedding = torch.load(
            embedding_path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(embedding, dict):

            if "embedding" in embedding:
                embedding = embedding["embedding"]

            elif "context_embedding" in embedding:
                embedding = embedding["context_embedding"]

            else:
                raise ValueError(
                    f"Unexpected embedding dictionary: {filename}"
                )

        if not torch.is_tensor(embedding):
            embedding = torch.tensor(embedding)

        context_embeddings[label] = embedding.float().squeeze()

    return context_embeddings


# ============================================================
# 13. ORIGINAL MODEL ARCHITECTURE
# ============================================================

class MultiContextSwinRegressor(nn.Module):

    def __init__(self, context_embeddings):

        super().__init__()

        self.swin = swin_v2_b(
            weights=Swin_V2_B_Weights.IMAGENET1K_V1
        )

        self.swin.head = nn.Identity()

        # Preserve the original context embedding structure.

        self.context_embeddings = nn.ParameterDict({

            label: nn.Parameter(
                context_embeddings[label].float().unsqueeze(0),
                requires_grad=False,
            ).squeeze(0)

            for label in context_embeddings

        })

        self.fusion_heads = nn.ModuleDict({

            label: nn.Sequential(

                nn.Linear(
                    1024 + context_embeddings[label].numel(),
                    256,
                ),

                nn.ReLU(),

                nn.Dropout(0.2),

                nn.Linear(256, 1),

            )

            for label in context_embeddings

        })


    def forward(self, x):

        image_features = self.swin(x)

        outputs = []

        for label in MODEL_LABELS:

            context = self.context_embeddings[label]

            context = context.unsqueeze(0).expand(
                image_features.shape[0],
                -1,
            )

            combined = torch.cat(
                [image_features, context],
                dim=1,
            )

            score = self.fusion_heads[label](combined)

            outputs.append(score)

        return torch.cat(outputs, dim=1)


# ============================================================
# 14. LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    context_embeddings = load_context_embeddings()

    model = MultiContextSwinRegressor(
        context_embeddings
    )

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="swin_regressor.pt",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 15. PREDICTION FUNCTION
# ============================================================

def predict_image(model, image):

    image = image.convert("RGB")

    tensor = preprocess_image(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        prediction = model(tensor)

        prediction = prediction * 6.0

        prediction = torch.clamp(
            prediction,
            min=1.0,
            max=6.0,
        )

    return prediction.squeeze(0).cpu().numpy()


# ============================================================
# 16. INITIALISE MODEL
# ============================================================

try:

    with st.spinner(T["loading_model"]):
        model = load_model()

    st.success(T["model_ready"])

except Exception as e:

    st.error(T["model_error"])
    st.exception(e)
    st.stop()


# ============================================================
# 17. IMAGE UPLOAD
# ============================================================

st.markdown(
    f"### {T['upload_title']}"
)

st.caption(T["upload_help"])

uploaded_files = st.file_uploader(
    T["upload_label"],
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)


# ============================================================
# 18. PROCESS IMAGES
# ============================================================

if uploaded_files:

    results = []

    with st.spinner(T["processing"]):

        progress_bar = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                scores = predict_image(
                    model,
                    image,
                )

                result = {
                    T["image"]: uploaded_file.name,
                }

                for dimension, score in zip(
                    translated_dimensions,
                    scores,
                ):

                    result[dimension] = float(score)

                result[T["total_score"]] = float(
                    np.mean(scores)
                )

                results.append(result)

            except Exception as e:

                st.warning(
                    f"{T['error_image']}: {uploaded_file.name}"
                )

                st.exception(e)

            progress_bar.progress(
                (index + 1) / len(uploaded_files)
            )

        progress_bar.empty()


    # ========================================================
    # 19. DISPLAY RESULTS
    # ========================================================

    if results:

        df = pd.DataFrame(results)

        st.divider()

        st.markdown(
            f"### {T['results_title']}"
        )

        st.caption(
            T["results_description"]
        )

        # Summary metrics.
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                T["number_images"],
                len(df),
            )

        with metric_col2:

            st.metric(
                T["average_score"],
                f"{df[T['total_score']].mean():.2f} / 6",
            )

        # Individual image results.
        st.markdown(
            f"#### {T['individual_results']}"
        )

        st.dataframe(
            df.style.format({
                col: "{:.3f}"
                for col in df.select_dtypes(
                    include="number"
                ).columns
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Dimension summary.
        st.markdown(
            f"#### {T['dimensions']}"
        )

        summary_df = pd.DataFrame({

            "Dimension": translated_dimensions,

            "Mean": [
                df[dimension].mean()
                for dimension in translated_dimensions
            ],

            "SD": [
                df[dimension].std()
                for dimension in translated_dimensions
            ],

        })

        summary_df["SD"] = summary_df["SD"].fillna(0)

        st.dataframe(
            summary_df.style.format({
                "Mean": "{:.3f}",
                "SD": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(T["score_range"])

        # CSV export.
        csv_data = df.to_csv(
            index=False,
            encoding="utf-8-sig",
        )

        st.download_button(
            label=f"⬇ {T['download']}",
            data=csv_data.encode("utf-8-sig"),
            file_name="CLASS_2.0_landscape_assessment.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:

        st.warning(T["no_images"])

else:

    st.info(T["no_images"])


# ============================================================
# 20. FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        {T["footer"]}
        <br>
        Contemplative Landscape Automated Scoring System
    </div>
    """,
    unsafe_allow_html=True,
)
