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
import html
import warnings
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import altair as alt
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

BOOK_URL = (
    "https://www.routledge.com/Neuroscience-for-Designing-Green-Spaces-"
    "Contemplative-Landscapes/Olszewska-Guizzo/p/book/9781032280639"
)

FURTHER_READING = [
    {
        "title": "Revised Contemplative Landscape Model (CLM): A reliable and valid evaluation tool for mental health-promoting urban green spaces",
        "authors": "Olszewska-Guizzo, Sia & Escoffier",
        "journal": "Urban Forestry & Urban Greening",
        "year": 2023,
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S1618866723001875",
    },
    {
        "title": "Therapeutic Garden With Contemplative Features Induces Desirable Changes in Mood and Brain Activity in Depressed Adults",
        "authors": "Olszewska-Guizzo, Fogel, Escoffier, Sia, Nakazawa, Kumagai, Dan & Ho",
        "journal": "Frontiers in Psychiatry",
        "year": 2022,
        "url": "https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.757056/full",
    },
    {
        "title": "Can Exposure to Certain Urban Green Spaces Trigger Frontal Alpha Asymmetry in the Brain? Preliminary Findings from a Passive Task EEG Study",
        "authors": "Olszewska-Guizzo, Sia, Fogel & Ho",
        "journal": "International Journal of Environmental Research and Public Health",
        "year": 2020,
        "url": "https://www.mdpi.com/1660-4601/17/2/394",
    },
]

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
    "🇵🇱 Polski": "pl",
    "🇯🇵 日本語": "ja",
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
    "Polish": "🇵🇱 Polski",
    "Japanese": "🇯🇵 日本語",
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

    "pl": {
        "language": "Język",
        "description": "Oceń cechy krajobrazu za pomocą modelu sztucznej inteligencji CLASS 2.0.",
        "upload_title": "Prześlij zdjęcia krajobrazu",
        "upload_help": "Prześlij jedno lub więcej zdjęć w formacie JPG, JPEG lub PNG.",
        "upload_label": "Wybierz zdjęcia krajobrazu",
        "results_title": "Wyniki oceny",
        "results_description": "Wyniki odpowiadają siedmiu wymiarom krajobrazu ocenianym przez CLASS 2.0.",
        "image": "Zdjęcie",
        "total_score": "Ogólna ocena krajobrazu",
        "download": "Pobierz wyniki jako CSV",
        "processing": "Przetwarzanie zdjęć krajobrazu...",
        "loading_model": "Wczytywanie modelu CLASS 2.0...",
        "model_ready": "Model CLASS 2.0 został wczytany.",
        "model_error": "Nie można wczytać modelu CLASS 2.0.",
        "no_images": "Prześlij zdjęcia, aby rozpocząć ocenę.",
        "error_image": "Nie można przetworzyć zdjęcia",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "Wymiary krajobrazu",
        "number_images": "Liczba ocenionych zdjęć",
        "average_score": "Średnia ocena krajobrazu",
        "individual_results": "Wyniki poszczególnych zdjęć",
        "score_range": "Zakres ocen: 1–6",
    },

    "ja": {
        "language": "言語",
        "description": "人工知能モデル CLASS 2.0 を使って景観の特性を評価します。",
        "upload_title": "景観画像をアップロード",
        "upload_help": "JPG、JPEG、PNG 形式の画像を 1 枚以上アップロードしてください。",
        "upload_label": "景観画像を選択",
        "results_title": "評価結果",
        "results_description": "以下のスコアは、CLASS 2.0 が評価する景観の 7 つの次元を表しています。",
        "image": "画像",
        "total_score": "景観の総合スコア",
        "download": "結果を CSV でダウンロード",
        "processing": "景観画像を処理しています...",
        "loading_model": "CLASS 2.0 モデルを読み込んでいます...",
        "model_ready": "CLASS 2.0 モデルを読み込みました。",
        "model_error": "CLASS 2.0 モデルを読み込めませんでした。",
        "no_images": "画像をアップロードして評価を開始してください。",
        "error_image": "画像を処理できませんでした",
        "footer": "NeuroLandscape | CLASS 2.0",
        "dimensions": "景観の次元",
        "number_images": "評価した画像の枚数",
        "average_score": "景観の平均スコア",
        "individual_results": "画像ごとの結果",
        "score_range": "スコアの範囲：1〜6",
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

    "pl": [
        "Warstwy krajobrazu",
        "Ukształtowanie terenu",
        "Bioróżnorodność",
        "Kolor i światło",
        "Kompatybilność",
        "Elementy archetypowe",
        "Charakter spokoju i ciszy",
    ],

    "ja": [
        "景観の層",
        "地形",
        "生物多様性",
        "色と光",
        "適合性",
        "原型的要素",
        "平和と静寂の特性",
    ],
}


# ============================================================
# 6b. ADDITIONAL INTERFACE TEXT
# ============================================================

UI_TRANSLATIONS = {
    "en": {
        "subtitle": "Contemplative Landscape Automated Scoring System",
        "tab_overview": "Overview",
        "tab_images": "Per image",
        "tab_table": "Data table",
        "profile_title": "Dimension profile",
        "profile_caption": "Mean score per dimension across all assessed images.",
        "heatmap_title": "Image comparison",
        "strongest": "Strongest dimension",
        "weakest": "Weakest dimension",
        "about_title": "About the seven dimensions",
        "about_caption": "CLASS 2.0 scores each image on seven dimensions of contemplative landscapes.",
        "read_more": "Read more in the book",
        "english_only": "",
        "col_dimension": "Dimension",
        "col_mean": "Mean",
        "col_sd": "Standard deviation",
        "score": "Score",
    },
    "fr": {
        "subtitle": "Système automatisé d'évaluation des paysages contemplatifs",
        "tab_overview": "Vue d'ensemble",
        "tab_images": "Par image",
        "tab_table": "Tableau de données",
        "profile_title": "Profil des dimensions",
        "profile_caption": "Score moyen par dimension pour l'ensemble des images évaluées.",
        "heatmap_title": "Comparaison des images",
        "strongest": "Dimension la plus forte",
        "weakest": "Dimension la plus faible",
        "about_title": "À propos des sept dimensions",
        "about_caption": "CLASS 2.0 évalue chaque image selon sept dimensions des paysages contemplatifs.",
        "read_more": "Lire la suite dans le livre",
        "english_only": "Le texte complet est disponible uniquement en anglais.",
        "col_dimension": "Dimension",
        "col_mean": "Moyenne",
        "col_sd": "Écart type",
        "score": "Score",
    },
    "it": {
        "subtitle": "Sistema automatizzato di valutazione dei paesaggi contemplativi",
        "tab_overview": "Panoramica",
        "tab_images": "Per immagine",
        "tab_table": "Tabella dati",
        "profile_title": "Profilo delle dimensioni",
        "profile_caption": "Punteggio medio per dimensione su tutte le immagini valutate.",
        "heatmap_title": "Confronto tra immagini",
        "strongest": "Dimensione più forte",
        "weakest": "Dimensione più debole",
        "about_title": "Le sette dimensioni",
        "about_caption": "CLASS 2.0 valuta ogni immagine secondo sette dimensioni dei paesaggi contemplativi.",
        "read_more": "Approfondisci nel libro",
        "english_only": "Il testo completo è disponibile solo in inglese.",
        "col_dimension": "Dimensione",
        "col_mean": "Media",
        "col_sd": "Deviazione standard",
        "score": "Punteggio",
    },
    "de": {
        "subtitle": "Automatisiertes Bewertungssystem für kontemplative Landschaften",
        "tab_overview": "Übersicht",
        "tab_images": "Pro Bild",
        "tab_table": "Datentabelle",
        "profile_title": "Dimensionsprofil",
        "profile_caption": "Durchschnittliche Bewertung pro Dimension über alle bewerteten Bilder.",
        "heatmap_title": "Bildvergleich",
        "strongest": "Stärkste Dimension",
        "weakest": "Schwächste Dimension",
        "about_title": "Die sieben Dimensionen",
        "about_caption": "CLASS 2.0 bewertet jedes Bild anhand von sieben Dimensionen kontemplativer Landschaften.",
        "read_more": "Mehr dazu im Buch",
        "english_only": "Der vollständige Text ist nur auf Englisch verfügbar.",
        "col_dimension": "Dimension",
        "col_mean": "Mittelwert",
        "col_sd": "Standardabweichung",
        "score": "Bewertung",
    },
    "sv": {
        "subtitle": "Automatiserat bedömningssystem för kontemplativa landskap",
        "tab_overview": "Översikt",
        "tab_images": "Per bild",
        "tab_table": "Datatabell",
        "profile_title": "Dimensionsprofil",
        "profile_caption": "Genomsnittlig poäng per dimension för alla bedömda bilder.",
        "heatmap_title": "Jämförelse av bilder",
        "strongest": "Starkaste dimension",
        "weakest": "Svagaste dimension",
        "about_title": "Om de sju dimensionerna",
        "about_caption": "CLASS 2.0 bedömer varje bild utifrån sju dimensioner av kontemplativa landskap.",
        "read_more": "Läs mer i boken",
        "english_only": "Hela texten finns endast på engelska.",
        "col_dimension": "Dimension",
        "col_mean": "Medelvärde",
        "col_sd": "Standardavvikelse",
        "score": "Poäng",
    },
    "zh": {
        "subtitle": "冥想景观自动评分系统",
        "tab_overview": "概览",
        "tab_images": "单张图像",
        "tab_table": "数据表",
        "profile_title": "维度概况",
        "profile_caption": "所有已评估图像在各维度上的平均得分。",
        "heatmap_title": "图像比较",
        "strongest": "最强维度",
        "weakest": "最弱维度",
        "about_title": "关于七个维度",
        "about_caption": "CLASS 2.0 依据冥想景观的七个维度为每张图像评分。",
        "read_more": "在书中阅读更多",
        "english_only": "全文仅提供英文版本。",
        "col_dimension": "维度",
        "col_mean": "平均值",
        "col_sd": "标准差",
        "score": "得分",
    },
    "ko": {
        "subtitle": "명상적 경관 자동 평가 시스템",
        "tab_overview": "개요",
        "tab_images": "이미지별",
        "tab_table": "데이터 표",
        "profile_title": "차원 프로필",
        "profile_caption": "평가된 모든 이미지의 차원별 평균 점수입니다.",
        "heatmap_title": "이미지 비교",
        "strongest": "가장 강한 차원",
        "weakest": "가장 약한 차원",
        "about_title": "일곱 가지 차원 소개",
        "about_caption": "CLASS 2.0은 명상적 경관의 일곱 가지 차원으로 각 이미지를 평가합니다.",
        "read_more": "책에서 더 읽기",
        "english_only": "전체 텍스트는 영어로만 제공됩니다.",
        "col_dimension": "차원",
        "col_mean": "평균",
        "col_sd": "표준편차",
        "score": "점수",
    },
    "es": {
        "subtitle": "Sistema automatizado de evaluación de paisajes contemplativos",
        "tab_overview": "Resumen",
        "tab_images": "Por imagen",
        "tab_table": "Tabla de datos",
        "profile_title": "Perfil de dimensiones",
        "profile_caption": "Puntuación media por dimensión en todas las imágenes evaluadas.",
        "heatmap_title": "Comparación de imágenes",
        "strongest": "Dimensión más fuerte",
        "weakest": "Dimensión más débil",
        "about_title": "Las siete dimensiones",
        "about_caption": "CLASS 2.0 puntúa cada imagen según siete dimensiones de los paisajes contemplativos.",
        "read_more": "Leer más en el libro",
        "english_only": "El texto completo solo está disponible en inglés.",
        "col_dimension": "Dimensión",
        "col_mean": "Media",
        "col_sd": "Desviación estándar",
        "score": "Puntuación",
    },
    "pt": {
        "subtitle": "Sistema automatizado de avaliação de paisagens contemplativas",
        "tab_overview": "Visão geral",
        "tab_images": "Por imagem",
        "tab_table": "Tabela de dados",
        "profile_title": "Perfil das dimensões",
        "profile_caption": "Pontuação média por dimensão em todas as imagens avaliadas.",
        "heatmap_title": "Comparação de imagens",
        "strongest": "Dimensão mais forte",
        "weakest": "Dimensão mais fraca",
        "about_title": "As sete dimensões",
        "about_caption": "O CLASS 2.0 pontua cada imagem segundo sete dimensões das paisagens contemplativas.",
        "read_more": "Leia mais no livro",
        "english_only": "O texto completo está disponível apenas em inglês.",
        "col_dimension": "Dimensão",
        "col_mean": "Média",
        "col_sd": "Desvio padrão",
        "score": "Pontuação",
    },
    "da": {
        "subtitle": "Automatiseret vurderingssystem for kontemplative landskaber",
        "tab_overview": "Oversigt",
        "tab_images": "Pr. billede",
        "tab_table": "Datatabel",
        "profile_title": "Dimensionsprofil",
        "profile_caption": "Gennemsnitlig score pr. dimension for alle vurderede billeder.",
        "heatmap_title": "Sammenligning af billeder",
        "strongest": "Stærkeste dimension",
        "weakest": "Svageste dimension",
        "about_title": "Om de syv dimensioner",
        "about_caption": "CLASS 2.0 vurderer hvert billede ud fra syv dimensioner af kontemplative landskaber.",
        "read_more": "Læs mere i bogen",
        "english_only": "Hele teksten findes kun på engelsk.",
        "col_dimension": "Dimension",
        "col_mean": "Gennemsnit",
        "col_sd": "Standardafvigelse",
        "score": "Score",
    },
    "nl": {
        "subtitle": "Geautomatiseerd beoordelingssysteem voor contemplatieve landschappen",
        "tab_overview": "Overzicht",
        "tab_images": "Per afbeelding",
        "tab_table": "Gegevenstabel",
        "profile_title": "Dimensieprofiel",
        "profile_caption": "Gemiddelde score per dimensie over alle beoordeelde afbeeldingen.",
        "heatmap_title": "Vergelijking van afbeeldingen",
        "strongest": "Sterkste dimensie",
        "weakest": "Zwakste dimensie",
        "about_title": "Over de zeven dimensies",
        "about_caption": "CLASS 2.0 beoordeelt elke afbeelding op zeven dimensies van contemplatieve landschappen.",
        "read_more": "Lees meer in het boek",
        "english_only": "De volledige tekst is alleen in het Engels beschikbaar.",
        "col_dimension": "Dimensie",
        "col_mean": "Gemiddelde",
        "col_sd": "Standaardafwijking",
        "score": "Score",
    },
    "ru": {
        "subtitle": "Автоматизированная система оценки созерцательных ландшафтов",
        "tab_overview": "Обзор",
        "tab_images": "По изображениям",
        "tab_table": "Таблица данных",
        "profile_title": "Профиль измерений",
        "profile_caption": "Средняя оценка по каждому измерению для всех оценённых изображений.",
        "heatmap_title": "Сравнение изображений",
        "strongest": "Сильнейшее измерение",
        "weakest": "Слабейшее измерение",
        "about_title": "О семи измерениях",
        "about_caption": "CLASS 2.0 оценивает каждое изображение по семи измерениям созерцательного ландшафта.",
        "read_more": "Читать подробнее в книге",
        "english_only": "Полный текст доступен только на английском языке.",
        "col_dimension": "Измерение",
        "col_mean": "Среднее",
        "col_sd": "Стандартное отклонение",
        "score": "Оценка",
    },
    "pl": {
        "subtitle": "Zautomatyzowany system oceny krajobrazów kontemplacyjnych",
        "tab_overview": "Przegląd",
        "tab_images": "Według zdjęcia",
        "tab_table": "Tabela danych",
        "profile_title": "Profil wymiarów",
        "profile_caption": "Średnia ocena każdego wymiaru dla wszystkich ocenionych zdjęć.",
        "heatmap_title": "Porównanie zdjęć",
        "strongest": "Najsilniejszy wymiar",
        "weakest": "Najsłabszy wymiar",
        "about_title": "O siedmiu wymiarach",
        "about_caption": "CLASS 2.0 ocenia każde zdjęcie według siedmiu wymiarów krajobrazów kontemplacyjnych.",
        "read_more": "Więcej w książce",
        "english_only": "Opis metody jest dostępny tylko w języku angielskim.",
        "col_dimension": "Wymiar",
        "col_mean": "Średnia",
        "col_sd": "Odchylenie standardowe",
        "score": "Ocena",
    },
    "ja": {
        "subtitle": "瞑想的景観 自動スコアリングシステム",
        "tab_overview": "概要",
        "tab_images": "画像ごと",
        "tab_table": "データ表",
        "profile_title": "次元プロファイル",
        "profile_caption": "評価したすべての画像における各次元の平均スコア。",
        "heatmap_title": "画像の比較",
        "strongest": "最も高い次元",
        "weakest": "最も低い次元",
        "about_title": "7 つの次元について",
        "about_caption": "CLASS 2.0 は、瞑想的景観の 7 つの次元で各画像を評価します。",
        "read_more": "書籍で詳しく読む",
        "english_only": "手法の説明は英語のみでご覧いただけます。",
        "col_dimension": "次元",
        "col_mean": "平均",
        "col_sd": "標準偏差",
        "score": "スコア",
    },
}

for _code, _extra in UI_TRANSLATIONS.items():
    TRANSLATIONS[_code].update(_extra)


# Short descriptions, in the same order as DIMENSION_LABELS.
DIMENSION_DESCRIPTIONS = {
    "en": [
        "Depth of view across foreground, middle ground and background: the 'borrowed scenery' that draws the eye into the distance.",
        "The shape of the terrain and the openness of the sky: how land and horizon frame a sense of vastness.",
        "The richness and variety of plant and animal life visible in the scene.",
        "The interplay of natural light, shadow and colour that gives the landscape its atmosphere.",
        "How harmoniously the elements of the scene fit together as a coherent whole.",
        "Timeless elements such as water, trees, paths or stones that carry deep symbolic meaning.",
        "The overall sense of tranquillity and freedom from noise and disturbance that the scene conveys.",
    ],
    "fr": [
        "Profondeur de vue entre premier plan, plan intermédiaire et arrière-plan : le « paysage emprunté » qui attire le regard au loin.",
        "La forme du terrain et l'ouverture du ciel : comment le relief et l'horizon créent un sentiment d'immensité.",
        "La richesse et la variété de la vie végétale et animale visible dans la scène.",
        "Le jeu de la lumière naturelle, des ombres et des couleurs qui donne au paysage son atmosphère.",
        "L'harmonie avec laquelle les éléments de la scène forment un ensemble cohérent.",
        "Des éléments intemporels comme l'eau, les arbres, les chemins ou les pierres, porteurs d'une profonde signification symbolique.",
        "Le sentiment global de tranquillité, à l'abri du bruit et des perturbations, que transmet la scène.",
    ],
    "it": [
        "Profondità visiva tra primo piano, piano intermedio e sfondo: il «paesaggio preso in prestito» che porta lo sguardo lontano.",
        "La forma del terreno e l'apertura del cielo: come terra e orizzonte creano un senso di vastità.",
        "La ricchezza e la varietà della vita vegetale e animale visibile nella scena.",
        "L'interazione tra luce naturale, ombra e colore che conferisce al paesaggio la sua atmosfera.",
        "Quanto armoniosamente gli elementi della scena si fondono in un insieme coerente.",
        "Elementi senza tempo come acqua, alberi, sentieri o pietre, portatori di un profondo significato simbolico.",
        "Il senso complessivo di tranquillità e di assenza di rumore e disturbo trasmesso dalla scena.",
    ],
    "de": [
        "Sichttiefe über Vorder-, Mittel- und Hintergrund: die „geborgte Landschaft“, die den Blick in die Ferne lenkt.",
        "Die Form des Geländes und die Weite des Himmels: wie Land und Horizont ein Gefühl von Weite erzeugen.",
        "Der Reichtum und die Vielfalt des sichtbaren Pflanzen- und Tierlebens.",
        "Das Zusammenspiel von natürlichem Licht, Schatten und Farbe, das der Landschaft ihre Atmosphäre verleiht.",
        "Wie harmonisch sich die Elemente der Szene zu einem stimmigen Ganzen fügen.",
        "Zeitlose Elemente wie Wasser, Bäume, Wege oder Steine mit tiefer symbolischer Bedeutung.",
        "Das Gesamtgefühl von Ruhe und Freiheit von Lärm und Störungen, das die Szene vermittelt.",
    ],
    "sv": [
        "Djupet i vyn över förgrund, mellangrund och bakgrund: det ”lånade landskapet” som drar blicken mot fjärran.",
        "Terrängens form och himlens öppenhet: hur land och horisont skapar en känsla av vidd.",
        "Rikedomen och variationen av växt- och djurliv som syns i scenen.",
        "Samspelet mellan naturligt ljus, skugga och färg som ger landskapet dess atmosfär.",
        "Hur harmoniskt scenens element passar ihop till en sammanhängande helhet.",
        "Tidlösa element som vatten, träd, stigar eller stenar med djup symbolisk betydelse.",
        "Den övergripande känslan av stillhet och frihet från buller och störningar som scenen förmedlar.",
    ],
    "zh": [
        "前景、中景与远景之间的视野深度：将视线引向远方的“借景”。",
        "地形的形态与天空的开阔度：大地与地平线如何营造辽阔之感。",
        "画面中可见的动植物生命的丰富性与多样性。",
        "自然光、阴影与色彩的相互作用，赋予景观独特的氛围。",
        "画面中各元素作为一个整体相互协调的程度。",
        "水、树木、小径或石头等永恒元素，承载着深刻的象征意义。",
        "画面所传达的整体宁静感，以及远离噪音与干扰的感受。",
    ],
    "ko": [
        "전경, 중경, 원경에 걸친 시야의 깊이: 시선을 먼 곳으로 이끄는 ‘차경’.",
        "지형의 형태와 하늘의 개방감: 땅과 지평선이 광활함을 만들어 내는 방식.",
        "장면에서 보이는 동식물 생명의 풍부함과 다양성.",
        "경관에 분위기를 부여하는 자연광, 그림자, 색채의 상호작용.",
        "장면의 요소들이 하나의 통일된 전체로 얼마나 조화롭게 어우러지는지.",
        "물, 나무, 길, 돌과 같이 깊은 상징적 의미를 지닌 시대를 초월한 요소.",
        "장면이 전달하는 전반적인 고요함과 소음 및 방해로부터의 자유로움.",
    ],
    "es": [
        "Profundidad de la vista entre primer plano, plano medio y fondo: el «paisaje prestado» que lleva la mirada a lo lejos.",
        "La forma del terreno y la apertura del cielo: cómo la tierra y el horizonte crean una sensación de inmensidad.",
        "La riqueza y variedad de la vida vegetal y animal visible en la escena.",
        "La interacción de la luz natural, la sombra y el color que da al paisaje su atmósfera.",
        "La armonía con la que los elementos de la escena forman un conjunto coherente.",
        "Elementos atemporales como el agua, los árboles, los caminos o las piedras, con un profundo significado simbólico.",
        "La sensación general de tranquilidad y ausencia de ruido y perturbaciones que transmite la escena.",
    ],
    "pt": [
        "Profundidade de vista entre primeiro plano, plano intermédio e fundo: a «paisagem emprestada» que leva o olhar ao longe.",
        "A forma do terreno e a abertura do céu: como a terra e o horizonte criam uma sensação de vastidão.",
        "A riqueza e a variedade da vida vegetal e animal visível na cena.",
        "A interação entre luz natural, sombra e cor que confere à paisagem a sua atmosfera.",
        "A harmonia com que os elementos da cena formam um todo coerente.",
        "Elementos intemporais como a água, as árvores, os caminhos ou as pedras, com profundo significado simbólico.",
        "A sensação geral de tranquilidade e ausência de ruído e perturbação que a cena transmite.",
    ],
    "da": [
        "Dybden i udsigten over forgrund, mellemgrund og baggrund: det „lånte landskab“, der trækker blikket ud i det fjerne.",
        "Terrænets form og himlens åbenhed: hvordan land og horisont skaber en følelse af vidde.",
        "Rigdommen og mangfoldigheden af plante- og dyreliv, der ses i scenen.",
        "Samspillet mellem naturligt lys, skygge og farve, der giver landskabet dets stemning.",
        "Hvor harmonisk scenens elementer passer sammen til en sammenhængende helhed.",
        "Tidløse elementer som vand, træer, stier eller sten med dyb symbolsk betydning.",
        "Den samlede følelse af ro og frihed fra støj og forstyrrelser, som scenen formidler.",
    ],
    "nl": [
        "Diepte van het zicht over voorgrond, middengrond en achtergrond: het ‘geleende landschap’ dat de blik naar de verte trekt.",
        "De vorm van het terrein en de openheid van de lucht: hoe land en horizon een gevoel van weidsheid scheppen.",
        "De rijkdom en verscheidenheid aan planten- en dierenleven in het beeld.",
        "Het samenspel van natuurlijk licht, schaduw en kleur dat het landschap zijn sfeer geeft.",
        "Hoe harmonieus de elementen van het beeld samen een samenhangend geheel vormen.",
        "Tijdloze elementen zoals water, bomen, paden of stenen met een diepe symbolische betekenis.",
        "Het algehele gevoel van rust en vrijheid van lawaai en verstoring dat het beeld uitstraalt.",
    ],
    "ru": [
        "Глубина обзора через передний, средний и дальний планы: «заимствованный пейзаж», уводящий взгляд вдаль.",
        "Форма рельефа и открытость неба: как земля и горизонт создают ощущение простора.",
        "Богатство и разнообразие растительной и животной жизни, видимой на изображении.",
        "Взаимодействие естественного света, тени и цвета, придающее ландшафту атмосферу.",
        "Насколько гармонично элементы сцены складываются в единое целое.",
        "Вечные элементы — вода, деревья, тропы или камни — несущие глубокий символический смысл.",
        "Общее ощущение покоя и свободы от шума и помех, которое передаёт сцена.",
    ],
    "pl": [
        "Głębia widoku przez pierwszy plan, plan środkowy i tło: „zapożyczony krajobraz”, który prowadzi wzrok w dal.",
        "Kształt terenu i otwartość nieba: jak ląd i horyzont tworzą poczucie przestrzeni.",
        "Bogactwo i różnorodność życia roślinnego i zwierzęcego widocznego w scenie.",
        "Gra naturalnego światła, cienia i koloru, która nadaje krajobrazowi nastrój.",
        "Jak harmonijnie elementy sceny łączą się w spójną całość.",
        "Ponadczasowe elementy, takie jak woda, drzewa, ścieżki czy kamienie, niosące głębokie znaczenie symboliczne.",
        "Ogólne poczucie spokoju i wolności od hałasu i zakłóceń, jakie przekazuje scena.",
    ],
    "ja": [
        "前景・中景・遠景にわたる眺めの奥行き。視線を遠くへ導く「借景」。",
        "地形の形と空の開放感。大地と地平線がつくり出す広がりの感覚。",
        "風景の中に見える植物や動物の豊かさと多様性。",
        "景観に雰囲気を与える自然光、影、色彩の相互作用。",
        "風景の要素が一つのまとまりとして、どれだけ調和しているか。",
        "水、樹木、小道、石など、深い象徴的意味をもつ普遍的な要素。",
        "風景が伝える全体的な静けさと、騒音や妨げから解放された感覚。",
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
# 8. THEME AND CUSTOM CSS
# ============================================================

PRIMARY = "#17324D"
ACCENT = "#4F7A5A"
MUTED = "#5A6B7B"
SURFACE = "#F5F7F3"
BORDER = "#E1E6DF"

APP_DIR = Path(__file__).parent

st.markdown(
    """
    <style>

    .stApp {
        background: transparent;
        color: #17324D;
    }

    [data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stToolbar"] {
        background: transparent !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    .bg-video {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        object-fit: cover;
        z-index: -1;
    }

    @media (prefers-reduced-motion: reduce) {
        .bg-video { display: none; }
        .stApp {
            background: url("app/static/contemplative_poster.jpg") center / cover fixed;
        }
    }

    .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem;
        max-width: 1200px;
        margin-top: 3rem;
        margin-bottom: 3rem;
        background: rgba(255, 255, 255, 0.94);
        border-radius: 18px;
        box-shadow: 0 20px 60px rgba(10, 30, 45, 0.25);
        backdrop-filter: blur(8px);
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 1.25rem 1rem 2rem 1rem;
            margin-top: 2.5rem;
            border-radius: 12px;
        }
    }

    .info-card {
        border: 1px solid #E1E6DF;
        border-radius: 12px;
        padding: 16px 18px;
        background: #FFFFFF;
        height: 100%;
    }

    .info-card h4 {
        font-size: 15px;
        margin: 0 0 6px 0;
    }

    .info-card p, .info-card li {
        font-size: 14px;
        color: #5A6B7B;
        line-height: 1.55;
        margin: 0;
    }

    .info-card ol {
        padding-left: 18px;
        margin: 0;
    }

    .big-number {
        font-size: 30px;
        font-weight: 750;
        color: #17324D;
        margin: 0;
        line-height: 1.1;
    }

    .big-number-label {
        font-size: 12px;
        font-weight: 700;
        color: #4F7A5A;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 4px 0 0 0;
    }

    .scale-strip {
        display: flex;
        gap: 6px;
        margin: 10px 0 4px 0;
    }

    .scale-strip span {
        flex: 1;
        text-align: center;
        padding: 8px 0;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        color: #FFFFFF;
    }

    .compare-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    .compare-table th, .compare-table td {
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid #E1E6DF;
        vertical-align: top;
    }

    .compare-table th {
        color: #4F7A5A;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .compare-table td:first-child {
        font-weight: 600;
        color: #17324D;
        white-space: nowrap;
    }

    .reading-item {
        border-left: 3px solid #4F7A5A;
        padding: 4px 0 4px 14px;
        margin-bottom: 16px;
    }

    .reading-item a {
        color: #17324D;
        font-weight: 650;
        font-size: 15px;
        text-decoration: none;
    }

    .reading-item a:hover {
        text-decoration: underline;
    }

    .reading-meta {
        font-size: 13px;
        color: #5A6B7B;
        margin: 3px 0 0 0;
    }

    .book-cover img {
        border-radius: 6px;
        box-shadow: 0 12px 30px rgba(10, 30, 45, 0.25);
    }

    html, body, [class*="css"] {
        font-family: "Helvetica Neue", Arial, Helvetica, sans-serif;
    }

    h1, h2, h3, h4 {
        color: #17324D;
        letter-spacing: -0.01em;
    }

    .class-title {
        font-size: 40px;
        font-weight: 750;
        color: #17324D;
        margin: 0;
        line-height: 1.1;
    }

    .class-subtitle {
        font-size: 15px;
        font-weight: 600;
        color: #4F7A5A;
        margin: 6px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .class-description {
        font-size: 16px;
        color: #5A6B7B;
        margin: 8px 0 0 0;
        line-height: 1.5;
        max-width: 640px;
    }

    .section-eyebrow {
        font-size: 12px;
        font-weight: 700;
        color: #4F7A5A;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 2px;
    }

    [data-testid="stFileUploader"] section {
        background-color: #F5F7F3;
        border: 1.5px dashed #A9BBA9;
        border-radius: 12px;
        padding: 1.5rem;
    }

    .stat-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 8px 0 16px 0;
    }

    .stat-card {
        flex: 1 1 200px;
        background-color: #F5F7F3;
        border: 1px solid #E1E6DF;
        border-radius: 12px;
        padding: 14px 16px;
    }

    .stat-label {
        font-size: 13px;
        color: #5A6B7B;
        margin: 0;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #17324D;
        margin: 2px 0 0 0;
        line-height: 1.2;
    }

    .stat-sub {
        font-size: 13px;
        color: #4F7A5A;
        font-weight: 600;
        margin: 2px 0 0 0;
    }

    .overall-score {
        display: flex;
        align-items: baseline;
        gap: 6px;
        margin-bottom: 10px;
    }

    .overall-score-value {
        font-size: 34px;
        font-weight: 750;
        color: #17324D;
        line-height: 1;
    }

    .overall-score-max {
        font-size: 15px;
        color: #5A6B7B;
    }

    .overall-score-label {
        font-size: 13px;
        color: #5A6B7B;
        margin: 0;
    }

    .score-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 0;
    }

    .score-label {
        flex: 0 0 44%;
        font-size: 14px;
        color: #17324D;
        line-height: 1.25;
    }

    .score-track {
        flex: 1 1 auto;
        height: 10px;
        background-color: #E6EBE4;
        border-radius: 999px;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
        background-color: #4F7A5A;
        border-radius: 999px;
    }

    .score-value {
        flex: 0 0 40px;
        text-align: right;
        font-size: 14px;
        font-weight: 600;
        color: #17324D;
        font-variant-numeric: tabular-nums;
    }

    .dimension-description {
        background-color: #F5F7F3;
        border-left: 3px solid #4F7A5A;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-size: 16px;
        color: #17324D;
        line-height: 1.55;
        margin: 6px 0 12px 0;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }

    .stDownloadButton button {
        background-color: #17324D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
    }

    .stDownloadButton button:hover {
        background-color: #284F70;
        color: white;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E1E6DF;
        border-radius: 8px;
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

st.markdown(
    '<video class="bg-video" autoplay muted loop playsinline preload="auto" '
    'poster="app/static/contemplative_poster.jpg" aria-hidden="true">'
    '<source src="app/static/contemplative_bg.mp4" type="video/mp4"></video>',
    unsafe_allow_html=True,
)


# ============================================================
# 9. HEADER AND LANGUAGE SELECTOR
# ============================================================

logo_col, title_col, language_col = st.columns(
    [1, 5, 2],
    vertical_alignment="center",
)

with logo_col:

    if logo_image is not None:
        st.image(logo_image, width=110)
    else:
        st.markdown(
            f'<div style="font-size:19px;font-weight:700;color:{PRIMARY};">NeuroLandscape</div>',
            unsafe_allow_html=True,
        )

# The title is filled in after the language is known.
title_slot = title_col.empty()

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
dimension_descriptions = DIMENSION_DESCRIPTIONS[lang_code]


# ============================================================
# 10. INTRODUCTION
# ============================================================

title_slot.markdown(
    f"""
    <div>
        <h1 class="class-title">CLASS 2.0</h1>
        <p class="class-subtitle">NeuroLandscape &middot; {html.escape(T["subtitle"])}</p>
        <p class="class-description">{html.escape(T["description"])}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


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

except Exception as e:

    st.error(T["model_error"])
    st.exception(e)
    st.stop()


# Cached on the file contents, so changing the language or
# switching tabs does not re-run the model.
@st.cache_data(show_spinner=False, max_entries=256)
def score_image_bytes(file_bytes):

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    return predict_image(model, image)


@st.cache_data(show_spinner=False, max_entries=256)
def make_thumbnail(file_bytes, max_size=900):

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)

    return buffer.getvalue()


# ============================================================
# 17. RENDERING HELPERS
# ============================================================

def score_bars_html(labels, scores):

    rows = []

    for label, score in zip(labels, scores):

        width = max(0.0, min(100.0, float(score) / 6.0 * 100.0))

        rows.append(
            '<div class="score-row">'
            f'<span class="score-label">{html.escape(label)}</span>'
            f'<div class="score-track" aria-hidden="true"><div class="score-fill" style="width:{width:.1f}%"></div></div>'
            f'<span class="score-value">{float(score):.2f}</span>'
            "</div>"
        )

    return "".join(rows)


def stat_card_html(label, value, sub=""):

    sub_html = f'<p class="stat-sub">{html.escape(sub)}</p>' if sub else ""

    return (
        '<div class="stat-card">'
        f'<p class="stat-label">{html.escape(label)}</p>'
        f'<p class="stat-value">{html.escape(value)}</p>'
        f"{sub_html}"
        "</div>"
    )


def profile_chart(summary):

    base = alt.Chart(summary).encode(
        y=alt.Y(
            "dimension:N",
            sort=None,
            title=None,
            axis=alt.Axis(labelLimit=260, labelFontSize=13, labelColor=PRIMARY),
        ),
    )

    bars = base.mark_bar(
        color=ACCENT,
        cornerRadiusEnd=4,
        height=22,
    ).encode(
        x=alt.X(
            "mean:Q",
            scale=alt.Scale(domain=[0, 6]),
            title=T["score"],
            axis=alt.Axis(tickCount=6, gridColor=BORDER),
        ),
        tooltip=[
            alt.Tooltip("dimension:N", title=T["col_dimension"]),
            alt.Tooltip("mean:Q", title=T["col_mean"], format=".2f"),
            alt.Tooltip("sd:Q", title=T["col_sd"], format=".2f"),
        ],
    )

    layers = [bars]

    if (summary["sd"] > 0).any():
        layers.append(
            base.mark_rule(color=PRIMARY, strokeWidth=1.5).encode(
                x="low:Q",
                x2="high:Q",
            )
        )

    layers.append(
        base.mark_text(
            align="left",
            dx=6,
            color=PRIMARY,
            fontWeight="bold",
        ).encode(
            x="high:Q",
            text=alt.Text("mean:Q", format=".2f"),
        )
    )

    return (
        alt.layer(*layers)
        .properties(height=len(summary) * 42)
        .configure_view(strokeWidth=0)
        .configure_axis(domainColor=BORDER, titleColor=MUTED)
    )


def comparison_heatmap(long_df, image_order):

    base = alt.Chart(long_df).encode(
        x=alt.X(
            "dimension:N",
            sort=translated_dimensions,
            title=None,
            axis=alt.Axis(labelAngle=-30, labelLimit=170, labelColor=PRIMARY),
        ),
        y=alt.Y(
            "image:N",
            sort=image_order,
            title=None,
            axis=alt.Axis(labelLimit=220, labelColor=PRIMARY),
        ),
    )

    rects = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color(
            "score:Q",
            scale=alt.Scale(domain=[1, 6], range=["#EEF2EC", ACCENT, PRIMARY]),
            legend=alt.Legend(title=T["score"]),
        ),
        tooltip=[
            alt.Tooltip("image:N", title=T["image"]),
            alt.Tooltip("dimension:N", title=T["col_dimension"]),
            alt.Tooltip("score:Q", title=T["score"], format=".2f"),
        ],
    )

    text = base.mark_text(fontSize=12, fontWeight="bold").encode(
        text=alt.Text("score:Q", format=".2f"),
        color=alt.condition(
            "datum.score >= 3.8",
            alt.value("#FFFFFF"),
            alt.value(PRIMARY),
        ),
    )

    return (
        (rects + text)
        .properties(height=max(120, len(image_order) * 38))
        .configure_view(strokeWidth=0)
    )


# ============================================================
# 18. IMAGE UPLOAD
# ============================================================

st.markdown(f"### {T['upload_title']}")

st.caption(T["upload_help"])

uploaded_files = st.file_uploader(
    T["upload_label"],
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


# ============================================================
# 19. PROCESS IMAGES
# ============================================================

results = []

if uploaded_files:

    with st.spinner(T["processing"]):

        progress_bar = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                file_bytes = uploaded_file.getvalue()

                scores = score_image_bytes(file_bytes)

                results.append({
                    "name": uploaded_file.name,
                    "label": f"{index + 1}. {uploaded_file.name}",
                    "scores": scores,
                    "overall": float(np.mean(scores)),
                    "thumbnail": make_thumbnail(file_bytes),
                })

            except Exception as e:

                st.warning(f"{T['error_image']}: {uploaded_file.name}")
                st.exception(e)

            progress_bar.progress((index + 1) / len(uploaded_files))

        progress_bar.empty()


# ============================================================
# 20. DISPLAY RESULTS
# ============================================================

if results:

    df = pd.DataFrame([
        {
            T["image"]: result["name"],
            **{
                dimension: float(score)
                for dimension, score in zip(translated_dimensions, result["scores"])
            },
            T["total_score"]: result["overall"],
        }
        for result in results
    ])

    means = df[translated_dimensions].mean()
    sds = df[translated_dimensions].std().fillna(0)

    summary = pd.DataFrame({
        "dimension": translated_dimensions,
        "mean": means.values,
        "sd": sds.values,
    })
    summary["low"] = (summary["mean"] - summary["sd"]).clip(lower=0)
    summary["high"] = (summary["mean"] + summary["sd"]).clip(upper=6)

    strongest_index = int(np.argmax(means.values))
    weakest_index = int(np.argmin(means.values))

    st.divider()

    st.markdown(f"### {T['results_title']}")
    st.caption(f"{T['results_description']} {T['score_range']}")

    st.markdown(
        '<div class="stat-grid">'
        + stat_card_html(T["number_images"], str(len(df)))
        + stat_card_html(T["average_score"], f"{df[T['total_score']].mean():.2f} / 6")
        + stat_card_html(
            T["strongest"],
            f"{means.values[strongest_index]:.2f}",
            translated_dimensions[strongest_index],
        )
        + stat_card_html(
            T["weakest"],
            f"{means.values[weakest_index]:.2f}",
            translated_dimensions[weakest_index],
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    overview_tab, images_tab, table_tab = st.tabs([
        T["tab_overview"],
        T["tab_images"],
        T["tab_table"],
    ])

    with overview_tab:

        st.markdown(f"#### {T['profile_title']}")
        st.caption(T["profile_caption"])

        st.altair_chart(profile_chart(summary), width="stretch")

        if len(results) > 1:

            st.markdown(f"#### {T['heatmap_title']}")

            long_df = pd.DataFrame([
                {
                    "image": result["label"],
                    "dimension": dimension,
                    "score": float(score),
                }
                for result in results
                for dimension, score in zip(translated_dimensions, result["scores"])
            ])

            st.altair_chart(
                comparison_heatmap(long_df, [result["label"] for result in results]),
                width="stretch",
            )

    with images_tab:

        for result in results:

            with st.container(border=True):

                image_col, score_col = st.columns([2, 3], gap="large")

                with image_col:
                    st.image(
                        result["thumbnail"],
                        caption=result["label"],
                        width="stretch",
                    )

                with score_col:
                    st.markdown(
                        f'<p class="overall-score-label">{html.escape(T["total_score"])}</p>'
                        '<div class="overall-score">'
                        f'<span class="overall-score-value">{result["overall"]:.2f}</span>'
                        '<span class="overall-score-max">/ 6</span>'
                        "</div>"
                        + score_bars_html(translated_dimensions, result["scores"]),
                        unsafe_allow_html=True,
                    )

    with table_tab:

        st.markdown(f"#### {T['individual_results']}")

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                column: st.column_config.ProgressColumn(
                    column,
                    min_value=0,
                    max_value=6,
                    format="%.2f",
                )
                for column in translated_dimensions + [T["total_score"]]
            },
        )

        st.markdown(f"#### {T['dimensions']}")

        st.dataframe(
            summary[["dimension", "mean", "sd"]].rename(columns={
                "dimension": T["col_dimension"],
                "mean": T["col_mean"],
                "sd": T["col_sd"],
            }),
            width="stretch",
            hide_index=True,
            column_config={
                T["col_mean"]: st.column_config.NumberColumn(format="%.3f"),
                T["col_sd"]: st.column_config.NumberColumn(format="%.3f"),
            },
        )

    csv_data = df.to_csv(index=False)

    st.download_button(
        label=T["download"],
        data=csv_data.encode("utf-8-sig"),
        file_name="CLASS_2.0_landscape_assessment.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )

elif uploaded_files:

    st.warning(T["no_images"])

else:

    st.info(T["no_images"])


# ============================================================
# 21. ABOUT THE DIMENSIONS
# ============================================================

st.divider()

st.markdown(f'<p class="section-eyebrow">CLASS 2.0</p>', unsafe_allow_html=True)
st.markdown(f"### {T['about_title']}")
st.caption(T["about_caption"])

dimension_tabs = st.tabs(translated_dimensions)

for tab, label, description in zip(
    dimension_tabs,
    DIMENSION_LABELS,
    dimension_descriptions,
):

    with tab:

        st.markdown(
            f'<div class="dimension-description">{html.escape(description)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"[{T['read_more']} →]({BOOK_URL})")


# ============================================================
# 22. THE CONTEMPLATIVE LANDSCAPE MODEL
# ============================================================

st.divider()

st.markdown('<p class="section-eyebrow">Method</p>', unsafe_allow_html=True)
st.markdown("### The Contemplative Landscape Model (CLM)")

if T["english_only"]:
    st.caption(T["english_only"])

st.markdown(
    "The Contemplative Landscape Model measures how far a landscape view has the "
    "potential to positively influence the mental health and well-being of people "
    "who are passively exposed to it. Its seven dimensions were identified from "
    "landscape-design literature and neuroscience experiments showing that highly "
    "contemplative scenes are linked to brain activity associated with relaxation, "
    "lower cognitive strain and positive mood. The **CLM** "
    "(Olszewska-Guizzo, Sia & Escoffier, 2023) was validated as a reliable "
    "evaluation tool for mental health-promoting urban green spaces."
)

metric_cols = st.columns(4)

for col, (value, caption) in zip(
    metric_cols,
    [
        ("7", "Dimensions"),
        ("1–6", "Point scale"),
        ("ω .893", "McDonald's omega"),
        ("α .890", "Cronbach's alpha"),
    ],
):
    col.markdown(
        f'<div class="info-card"><p class="big-number">{html.escape(value)}</p>'
        f'<p class="big-number-label">{html.escape(caption)}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("#### How a scene is scored")

scale_colors = ["#B8C7B1", "#9DB596", "#7FA07A", "#64885F", "#4F7A5A", "#2F5A3C"]

st.markdown(
    '<div class="scale-strip" role="img" aria-label="Scale from 1, least contemplative, to 6, most contemplative">'
    + "".join(
        f'<span style="background:{color}">{i}</span>'
        for i, color in enumerate(scale_colors, start=1)
    )
    + "</div>",
    unsafe_allow_html=True,
)
st.caption("1 = least contemplative · 6 = most contemplative")

score_cols = st.columns(2)

with score_cols[0]:
    st.markdown(
        '<div class="info-card"><h4>Manual CLM assessment</h4><ol>'
        "<li>A rater with formal training in landscape architecture, architecture or urban design views the scene, on site or as a photo or video.</li>"
        "<li>Each of the seven dimensions is rated on a 1–6 scale: Layers of the Landscape, Landform, Biodiversity, Color and Light, Compatibility, Archetypal Elements, and Character of Peace and Silence.</li>"
        "<li>The overall CLM score is the average of the seven dimension scores.</li>"
        "<li>For research use, scores from several experts are pooled; the CLM validation used a panel of 13 raters, whose averaged ratings showed excellent agreement.</li>"
        "</ol></div>",
        unsafe_allow_html=True,
    )

with score_cols[1]:
    st.markdown(
        '<div class="info-card"><h4>How CLASS 2.0 automates it</h4><ol>'
        "<li>Each uploaded photo is resized and passed through a Swin Transformer V2 vision model that encodes the composition, vegetation, light and structure of the scene.</li>"
        "<li>A learned context embedding for each CLM dimension steers a regression head, which predicts that dimension's score on the same 1–6 scale used by expert raters.</li>"
        "<li>The seven predictions are averaged into an overall CLM score, exactly like the manual method.</li>"
        "<li>When you upload several photos of one place, CLASS reports the mean and spread of each dimension across all views.</li>"
        "</ol></div>",
        unsafe_allow_html=True,
    )

st.markdown("#### Manual vs. automated")

st.markdown(
    """
    <table class="compare-table">
        <thead><tr><th scope="col"></th><th scope="col">Manual CLM</th><th scope="col">CLASS 2.0</th></tr></thead>
        <tbody>
            <tr><td>Rater</td><td>Trained landscape or design experts</td><td>Deep-learning model trained to predict expert CLM ratings</td></tr>
            <tr><td>Input</td><td>On-site view, photo or video</td><td>One or more photographs</td></tr>
            <tr><td>Scale</td><td>Seven dimensions, 1–6 each</td><td>Seven dimensions, 1–6 each</td></tr>
            <tr><td>Overall score</td><td>Mean of the seven dimensions</td><td>Mean of the seven dimensions</td></tr>
            <tr><td>Scaling up</td><td>Each scene needs expert time; panels improve consistency</td><td>Seconds per image and identical results for the same image, so many sites can be screened quickly</td></tr>
        </tbody>
    </table>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "CLASS 2.0 is designed for screening and comparing sites at scale. "
    "For design decisions, read its scores alongside an expert CLM assessment."
)


# ============================================================
# 23. THE BOOK
# ============================================================

st.divider()

book_cols = st.columns([1, 2.4], gap="large", vertical_alignment="center")

with book_cols[0]:
    st.markdown('<div class="book-cover">', unsafe_allow_html=True)
    st.image(
        str(APP_DIR / "book_cover.jpg"),
        caption=None,
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with book_cols[1]:
    st.markdown('<p class="section-eyebrow">The book</p>', unsafe_allow_html=True)
    st.markdown("### Neuroscience for Designing Green Spaces: Contemplative Landscapes")
    st.markdown("**Agnieszka Olszewska-Guizzo** · Routledge, 2023")
    st.markdown(
        "The textbook behind the seven CLM dimensions. It explains each landscape "
        "quality in depth, the neuroscience evidence behind it, and how designers "
        "can use these features to create green spaces that support mental health."
    )
    st.link_button(
        "View the book on Routledge",
        BOOK_URL,
        icon=":material/menu_book:",
        type="primary",
    )


# ============================================================
# 24. FURTHER READING
# ============================================================

st.divider()

st.markdown('<p class="section-eyebrow">Research</p>', unsafe_allow_html=True)
st.markdown("### Further reading")

for paper in FURTHER_READING:
    st.markdown(
        '<div class="reading-item">'
        f'<a href="{html.escape(paper["url"])}" target="_blank" rel="noopener noreferrer">{html.escape(paper["title"])}</a>'
        f'<p class="reading-meta">{html.escape(paper["authors"])} · <em>{html.escape(paper["journal"])}</em> ({paper["year"]})</p>'
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# 25. FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        {html.escape(T["footer"])}
        <br>
        {html.escape(T["subtitle"])}
    </div>
    """,
    unsafe_allow_html=True,
)
