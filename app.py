
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from PIL import Image
from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd
import csv
import io

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CLASS 2.0 | NeuroLandscape",
    page_icon="🌿",
    layout="wide"
)

HF_REPO = "DOReilly2/swin_regressor"
DEVICE = "cpu"

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "neurolandscape_logo.png"

# Original model identifiers: DO NOT TRANSLATE

dimension_labels = [
    "Layers of the Landscape_embedding",
    "Landform_embedding",
    "Biodiversity_embedding",
    "Color and Light_embedding",
    "Compatibility_embedding",
    "Archetypal Elements_embedding",
    "Character of Peace and Silence_embedding"
]


# ============================================================
# 2. MULTILINGUAL TRANSLATION DICTIONARY
# ============================================================

TRANSLATIONS = {

    "English": {
        "title": "CLASS 2.0",
        "subtitle": "Computational Landscape Assessment",
        "description": "An AI-powered system for assessing landscape quality.",
        "upload": "Upload landscape images",
        "upload_help": "Upload one or more landscape photographs for assessment.",
        "results": "Assessment Results",
        "copy_table": "Copy-Paste Table (Excel Friendly)",
        "results_label": "Results",
        "download": "Download as CSV",
        "success": "Model loaded successfully.",
        "processing": "Processing landscape images...",
        "complete": "Assessment completed successfully!",
        "image_name": "Image Name",
        "language": "Interface Language",
        "welcome": "Upload landscape images to generate environmental quality assessments.",
        "invalid_image": "Could not process image",
        "dimensions": [
            "Layers of the Landscape",
            "Landform",
            "Biodiversity",
            "Color and Light",
            "Compatibility",
            "Archetypal Elements",
            "Character of Peace and Silence"
        ]
    },

    "French": {
        "title": "CLASS 2.0",
        "subtitle": "Évaluation computationnelle du paysage",
        "description": "Un système d'intelligence artificielle pour évaluer la qualité des paysages.",
        "upload": "Télécharger des images de paysages",
        "upload_help": "Téléchargez une ou plusieurs photographies de paysages pour évaluation.",
        "results": "Résultats de l'évaluation",
        "copy_table": "Tableau à copier (compatible avec Excel)",
        "results_label": "Résultats",
        "download": "Télécharger au format CSV",
        "success": "Modèle chargé avec succès.",
        "processing": "Traitement des images de paysages...",
        "complete": "Évaluation terminée avec succès !",
        "image_name": "Nom de l'image",
        "language": "Langue de l'interface",
        "welcome": "Téléchargez des images de paysages pour générer des évaluations de la qualité environnementale.",
        "invalid_image": "Impossible de traiter l'image",
        "dimensions": [
            "Couches du paysage",
            "Forme du relief",
            "Biodiversité",
            "Couleur et lumière",
            "Compatibilité",
            "Éléments archétypaux",
            "Caractère de paix et de silence"
        ]
    },

    "Italian": {
        "title": "CLASS 2.0",
        "subtitle": "Valutazione computazionale del paesaggio",
        "description": "Un sistema basato sull'intelligenza artificiale per valutare la qualità del paesaggio.",
        "upload": "Carica immagini del paesaggio",
        "upload_help": "Carica una o più fotografie del paesaggio per la valutazione.",
        "results": "Risultati della valutazione",
        "copy_table": "Tabella da copiare (compatibile con Excel)",
        "results_label": "Risultati",
        "download": "Scarica come CSV",
        "success": "Modello caricato correttamente.",
        "processing": "Elaborazione delle immagini del paesaggio...",
        "complete": "Valutazione completata con successo!",
        "image_name": "Nome dell'immagine",
        "language": "Lingua dell'interfaccia",
        "welcome": "Carica immagini del paesaggio per generare valutazioni della qualità ambientale.",
        "invalid_image": "Impossibile elaborare l'immagine",
        "dimensions": [
            "Strati del paesaggio",
            "Forma del terreno",
            "Biodiversità",
            "Colore e luce",
            "Compatibilità",
            "Elementi archetipici",
            "Carattere di pace e silenzio"
        ]
    },

    "German": {
        "title": "CLASS 2.0",
        "subtitle": "Computergestützte Landschaftsbewertung",
        "description": "Ein KI-gestütztes System zur Bewertung der Landschaftsqualität.",
        "upload": "Landschaftsbilder hochladen",
        "upload_help": "Laden Sie ein oder mehrere Landschaftsfotos zur Bewertung hoch.",
        "results": "Bewertungsergebnisse",
        "copy_table": "Kopierbare Tabelle (Excel-kompatibel)",
        "results_label": "Ergebnisse",
        "download": "Als CSV herunterladen",
        "success": "Modell erfolgreich geladen.",
        "processing": "Landschaftsbilder werden verarbeitet...",
        "complete": "Bewertung erfolgreich abgeschlossen!",
        "image_name": "Bildname",
        "language": "Oberflächensprache",
        "welcome": "Laden Sie Landschaftsbilder hoch, um Bewertungen der Umweltqualität zu erstellen.",
        "invalid_image": "Bild konnte nicht verarbeitet werden",
        "dimensions": [
            "Landschaftsebenen",
            "Reliefform",
            "Biodiversität",
            "Farbe und Licht",
            "Kompatibilität",
            "Archetypische Elemente",
            "Charakter von Frieden und Stille"
        ]
    },

    "Swedish": {
        "title": "CLASS 2.0",
        "subtitle": "Datorbaserad landskapsbedömning",
        "description": "Ett AI-baserat system för bedömning av landskapskvalitet.",
        "upload": "Ladda upp landskapsbilder",
        "upload_help": "Ladda upp ett eller flera landskapsfotografier för bedömning.",
        "results": "Bedömningsresultat",
        "copy_table": "Kopiera tabell (Excel-kompatibel)",
        "results_label": "Resultat",
        "download": "Ladda ner som CSV",
        "success": "Modellen har laddats in.",
        "processing": "Bearbetar landskapsbilder...",
        "complete": "Bedömningen har slutförts!",
        "image_name": "Bildnamn",
        "language": "Gränssnittsspråk",
        "welcome": "Ladda upp landskapsbilder för att generera bedömningar av miljökvalitet.",
        "invalid_image": "Bilden kunde inte bearbetas",
        "dimensions": [
            "Landskapets lager",
            "Landform",
            "Biologisk mångfald",
            "Färg och ljus",
            "Kompatibilitet",
            "Arketypiska element",
            "Karaktär av lugn och stillhet"
        ]
    },

    "Chinese": {
        "title": "CLASS 2.0",
        "subtitle": "计算景观评估",
        "description": "一个基于人工智能的景观质量评估系统。",
        "upload": "上传景观图像",
        "upload_help": "上传一张或多张景观照片进行评估。",
        "results": "评估结果",
        "copy_table": "可复制表格（兼容 Excel）",
        "results_label": "结果",
        "download": "下载 CSV 文件",
        "success": "模型加载成功。",
        "processing": "正在处理景观图像...",
        "complete": "评估已成功完成！",
        "image_name": "图像名称",
        "language": "界面语言",
        "welcome": "上传景观图像以生成环境质量评估结果。",
        "invalid_image": "无法处理图像",
        "dimensions": [
            "景观层次",
            "地貌",
            "生物多样性",
            "色彩与光线",
            "协调性",
            "原型元素",
            "宁静与寂静的特征"
        ]
    },

    "Korean": {
        "title": "CLASS 2.0",
        "subtitle": "컴퓨터 기반 경관 평가",
        "description": "인공지능 기반 경관 품질 평가 시스템입니다.",
        "upload": "경관 이미지 업로드",
        "upload_help": "평가를 위해 하나 이상의 경관 사진을 업로드하세요.",
        "results": "평가 결과",
        "copy_table": "복사 가능한 표 (Excel 호환)",
        "results_label": "결과",
        "download": "CSV 파일 다운로드",
        "success": "모델이 성공적으로 로드되었습니다.",
        "processing": "경관 이미지를 처리하고 있습니다...",
        "complete": "평가가 성공적으로 완료되었습니다!",
        "image_name": "이미지 이름",
        "language": "인터페이스 언어",
        "welcome": "환경 품질 평가를 생성하려면 경관 이미지를 업로드하세요.",
        "invalid_image": "이미지를 처리할 수 없습니다",
        "dimensions": [
            "경관의 층위",
            "지형",
            "생물다양성",
            "색상과 빛",
            "적합성",
            "원형적 요소",
            "평온함과 고요함의 특성"
        ]
    },

    "Spanish": {
        "title": "CLASS 2.0",
        "subtitle": "Evaluación computacional del paisaje",
        "description": "Un sistema basado en inteligencia artificial para evaluar la calidad del paisaje.",
        "upload": "Subir imágenes del paisaje",
        "upload_help": "Suba una o varias fotografías del paisaje para su evaluación.",
        "results": "Resultados de la evaluación",
        "copy_table": "Tabla para copiar (compatible con Excel)",
        "results_label": "Resultados",
        "download": "Descargar como CSV",
        "success": "Modelo cargado correctamente.",
        "processing": "Procesando imágenes del paisaje...",
        "complete": "¡Evaluación completada correctamente!",
        "image_name": "Nombre de la imagen",
        "language": "Idioma de la interfaz",
        "welcome": "Suba imágenes del paisaje para generar evaluaciones de la calidad ambiental.",
        "invalid_image": "No se pudo procesar la imagen",
        "dimensions": [
            "Capas del paisaje",
            "Forma del relieve",
            "Biodiversidad",
            "Color y luz",
            "Compatibilidad",
            "Elementos arquetípicos",
            "Carácter de paz y silencio"
        ]
    },

    "Portuguese": {
        "title": "CLASS 2.0",
        "subtitle": "Avaliação computacional da paisagem",
        "description": "Um sistema baseado em inteligência artificial para avaliar a qualidade da paisagem.",
        "upload": "Carregar imagens da paisagem",
        "upload_help": "Carregue uma ou mais fotografias da paisagem para avaliação.",
        "results": "Resultados da avaliação",
        "copy_table": "Tabela para copiar (compatível com Excel)",
        "results_label": "Resultados",
        "download": "Descarregar como CSV",
        "success": "Modelo carregado com sucesso.",
        "processing": "A processar imagens da paisagem...",
        "complete": "Avaliação concluída com sucesso!",
        "image_name": "Nome da imagem",
        "language": "Idioma da interface",
        "welcome": "Carregue imagens da paisagem para gerar avaliações da qualidade ambiental.",
        "invalid_image": "Não foi possível processar a imagem",
        "dimensions": [
            "Camadas da paisagem",
            "Forma do relevo",
            "Biodiversidade",
            "Cor e luz",
            "Compatibilidade",
            "Elementos arquetípicos",
            "Caráter de paz e silêncio"
        ]
    },

    "Danish": {
        "title": "CLASS 2.0",
        "subtitle": "Computerbaseret landskabsvurdering",
        "description": "Et AI-baseret system til vurdering af landskabskvalitet.",
        "upload": "Upload landskabsbilleder",
        "upload_help": "Upload et eller flere landskabsfotografier til vurdering.",
        "results": "Vurderingsresultater",
        "copy_table": "Kopierbar tabel (Excel-kompatibel)",
        "results_label": "Resultater",
        "download": "Download som CSV",
        "success": "Modellen er indlæst.",
        "processing": "Behandler landskabsbilleder...",
        "complete": "Vurderingen er gennemført!",
        "image_name": "Billednavn",
        "language": "Grænsefladesprog",
        "welcome": "Upload landskabsbilleder for at generere vurderinger af miljøkvalitet.",
        "invalid_image": "Billedet kunne ikke behandles",
        "dimensions": [
            "Landskabets lag",
            "Landform",
            "Biodiversitet",
            "Farve og lys",
            "Kompatibilitet",
            "Arketypiske elementer",
            "Karakter af fred og stilhed"
        ]
    },

    "Dutch": {
        "title": "CLASS 2.0",
        "subtitle": "Computationele landschapsbeoordeling",
        "description": "Een AI-gestuurd systeem voor de beoordeling van landschapskwaliteit.",
        "upload": "Landschapsafbeeldingen uploaden",
        "upload_help": "Upload een of meerdere landschapsfoto's voor beoordeling.",
        "results": "Beoordelingsresultaten",
        "copy_table": "Kopieerbare tabel (Excel-compatibel)",
        "results_label": "Resultaten",
        "download": "Downloaden als CSV",
        "success": "Model succesvol geladen.",
        "processing": "Landschapsafbeeldingen verwerken...",
        "complete": "Beoordeling succesvol afgerond!",
        "image_name": "Afbeeldingsnaam",
        "language": "Interfacetaal",
        "welcome": "Upload landschapsafbeeldingen om beoordelingen van de milieukwaliteit te genereren.",
        "invalid_image": "Afbeelding kon niet worden verwerkt",
        "dimensions": [
            "Lagen van het landschap",
            "Landvorm",
            "Biodiversiteit",
            "Kleur en licht",
            "Compatibiliteit",
            "Archetypische elementen",
            "Karakter van rust en stilte"
        ]
    },

    "Russian": {
        "title": "CLASS 2.0",
        "subtitle": "Компьютерная оценка ландшафта",
        "description": "Система на основе искусственного интеллекта для оценки качества ландшафта.",
        "upload": "Загрузить изображения ландшафта",
        "upload_help": "Загрузите одну или несколько фотографий ландшафта для оценки.",
        "results": "Результаты оценки",
        "copy_table": "Таблица для копирования (совместима с Excel)",
        "results_label": "Результаты",
        "download": "Скачать в формате CSV",
        "success": "Модель успешно загружена.",
        "processing": "Обработка изображений ландшафта...",
        "complete": "Оценка успешно завершена!",
        "image_name": "Название изображения",
        "language": "Язык интерфейса",
        "welcome": "Загрузите изображения ландшафта для получения оценок качества окружающей среды.",
        "invalid_image": "Не удалось обработать изображение",
        "dimensions": [
            "Слои ландшафта",
            "Форма рельефа",
            "Биоразнообразие",
            "Цвет и свет",
            "Совместимость",
            "Архетипические элементы",
            "Характер покоя и тишины"
        ]
    }
}


# ============================================================
# 3. LANGUAGE CODES
# ============================================================

LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
    "Italian": "it",
    "German": "de",
    "Swedish": "sv",
    "Chinese": "zh",
    "Korean": "ko",
    "Spanish": "es",
    "Portuguese": "pt",
    "Danish": "da",
    "Dutch": "nl",
    "Russian": "ru"
}


# ============================================================
# 4. LANGUAGE SELECTION
# ============================================================

if "language" not in st.session_state:
    st.session_state.language = "English"

# ============================================================
# 5. BRANDED NEUROLANDSCAPE HEADER
# ============================================================

logo_col, title_col, language_col = st.columns(
    [1.5, 3.5, 2]
)

with logo_col:

    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            width=190
        )
    else:
        st.markdown("### NeuroLandscape")

with title_col:

    st.title("CLASS 2.0")
    st.caption("NeuroLandscape | Digital Landscape Assessment")

with language_col:

    st.selectbox(
        "Interface Language / Langue / Sprache",
        options=list(TRANSLATIONS.keys()),
        key="language"
    )

T = TRANSLATIONS[st.session_state.language]

st.divider()

st.subheader(T["subtitle"])
st.write(T["description"])


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image) -> torch.Tensor:

    val_transform = transforms.Compose([

        transforms.Resize((512, 512)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    image = image.convert("RGB")

    return val_transform(image).unsqueeze(0).to(DEVICE)


# ============================================================
# 7. LOAD CONTEXT EMBEDDINGS
# ============================================================

@st.cache_resource
def load_context_embeddings():

    embeddings = {}

    for label in dimension_labels:

        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"context_embeddings/{label}.pt"
        )

        emb = torch.load(
            path,
            map_location="cpu"
        )

        embeddings[label] = emb.squeeze()

    return embeddings


context_embeddings = load_context_embeddings()


# ============================================================
# 8. MODEL ARCHITECTURE
# ============================================================

class MultiContextSwinRegressor(nn.Module):

    def __init__(self, context_embeddings: dict):

        super().__init__()

        self.swin = swin_v2_b(
            weights=Swin_V2_B_Weights.IMAGENET1K_V1
        )

        self.swin.head = nn.Identity()

        self.context_embeddings = nn.ParameterDict({

            label: nn.Parameter(
                context_embeddings[label]
                .float()
                .unsqueeze(0),
                requires_grad=False
            ).squeeze(0)

            for label in context_embeddings

        })

        self.fusion_heads = nn.ModuleDict({

            label: nn.Sequential(

                nn.Linear(
                    1024 + self.context_embeddings[label].shape[0],
                    256
                ),

                nn.ReLU(),

                nn.Dropout(0.2),

                nn.Linear(256, 1)

            )

            for label in self.context_embeddings

        })

    def forward(self, image):

        image_feat = self.swin(image)

        outputs = []

        for label in self.context_embeddings:

            context = self.context_embeddings[label].expand(
                image_feat.size(0), -1
            )

            fused = torch.cat(
                [image_feat, context],
                dim=1
            )

            score = self.fusion_heads[label](fused)

            outputs.append(score)

        return torch.cat(outputs, dim=1)


# ============================================================
# 9. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="swin_regressor.pt"
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    model = MultiContextSwinRegressor(
        context_embeddings
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# ============================================================
# 10. MULTILINGUAL USER INTERFACE
# ============================================================

st.success(T["success"])

st.write(T["welcome"])

uploaded_files = st.file_uploader(

    T["upload"],

    type=["jpg", "jpeg", "png"],

    accept_multiple_files=True,

    help=T["upload_help"]

)


# ============================================================
# 11. IMAGE PREDICTION
# ============================================================

if uploaded_files:

    results = []

    progress = st.progress(0)

    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):

        try:

            image = Image.open(uploaded_file)

            image_tensor = preprocess_image(image)

            with torch.no_grad():

                predicted_scores = model(image_tensor)

                predicted_scores = predicted_scores.squeeze(0) * 6

                predicted_scores = torch.clamp(
                    predicted_scores,
                    1.0,
                    6.0
                )

                predicted_scores = predicted_scores.cpu().tolist()

            results.append(

                [uploaded_file.name] +

                [
                    round(float(score), 2)
                    for score in predicted_scores
                ]

            )

        except Exception as e:

            st.error(
                f"{T['invalid_image']}: "
                f"{uploaded_file.name}: {str(e)}"
            )

        progress.progress(
            (index + 1) / len(uploaded_files)
        )

    status.success(T["complete"])


    # ========================================================
    # 12. MULTILINGUAL RESULTS
    # ========================================================

    if results:

        st.divider()

        st.subheader(T["results"])

        headers = [
            T["image_name"]
        ] + T["dimensions"]

        results_df = pd.DataFrame(
            results,
            columns=headers
        )

        # Display results table

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 13. EXCEL-FRIENDLY CSV
        # ====================================================

        csv_buffer = io.StringIO()

        writer = csv.writer(csv_buffer)

        writer.writerow(headers)

        writer.writerows(results)

        table_text = csv_buffer.getvalue()

        st.subheader(T["copy_table"])

        st.text_area(
            T["results_label"],
            table_text,
            height=400
        )


        # ====================================================
        # 14. MULTILINGUAL CSV DOWNLOAD
        # ====================================================

        language_code = LANGUAGE_CODES[
            st.session_state.language
        ]

        st.download_button(

            label=T["download"],

            data="\ufeff" + table_text,

            file_name=f"CLASS2_predictions_{language_code}.csv",

            mime="text/csv",

            use_container_width=True

        )
