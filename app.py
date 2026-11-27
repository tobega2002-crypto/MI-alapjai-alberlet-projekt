import os
import tempfile
import streamlit as st
from PIL import Image

from src.decision_engine import RentCheckEngine
from src.image_analyzer import create_demo_images

st.set_page_config(
    page_title="RentCheck AI – Albérlet-értékbecslő",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 RentCheck AI – Intelligens Albérlet-értékbecslő és Hirdetés-elemző")
st.markdown(
    "Multimodális MI döntéstámogató rendszer: **Gépi tanulásos árbecslés (Regresszió)** + "
    "**NLP szövegelemzés** + **OpenCV képfeldolgozás**."
)

# Motor inicializálása gyorsítótárazva
@st.cache_resource
def load_engine():
    return RentCheckEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Hiba a modell betöltésekor: {e}")
    st.stop()

# Oldalsáv: Hirdetés paramétereinek megadása
st.sidebar.header("📋 Hirdetés adatai")

kerulet = st.sidebar.selectbox(
    "Kerület",
    [
        "I. kerület", "II. kerület", "V. kerület", "VI. kerület", "VII. kerület",
        "VIII. kerület", "IX. kerület", "XI. kerület", "XIII. kerület", "XIV. kerület"
    ],
    index=7,
)
alapterulet = st.sidebar.slider("Alapterület (m²)", min_value=20, max_value=130, value=54)
szobak = st.sidebar.selectbox("Szobák száma", [1, 2, 3, 4], index=1)
allapot = st.sidebar.selectbox(
    "Ingatlan állapota",
    ["Új építésű", "Felújított", "Jó állapotú", "Felújítandó"],
    index=1,
)
erkely = st.sidebar.checkbox("Erkély / Terasz", value=True)
lift = st.sidebar.checkbox("Lift van", value=True)
butorozott = st.sidebar.checkbox("Bútorozott", value=True)

advertised_price = st.sidebar.number_input(
    "Hirdetett bérleti díj (Ft/hó)",
    min_value=80000,
    max_value=1500000,
    value=245000,
    step=5000,
)

# Fő panel két oszlopban
col_input, col_result = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("1. Hirdetés szöveges leírása (NLP)")
    default_desc = (
        f"Kiadó a {kerulet}ben egy {alapterulet} nm-es, {szobak} szobás, {allapot.lower()} lakás. "
        "Világos, napfényes, alacsony rezsi, metró 2 perc, klimatizált, azonnal költözhető."
    )
    description = st.text_area("Másold be a hirdetés szövegét:", value=default_desc, height=130)

    st.subheader("2. Lakásfotók feltöltése (OpenCV)")
    sample_mode = st.radio(
        "Képforrás kiválasztása:",
        [
            "Saját képek feltöltése",
            "Demó teszt: Világos, éles szobakép",
            "Demó teszt: Sötét, életlen szobakép",
        ],
    )

    uploaded_files = None
    if sample_mode == "Saját képek feltöltése":
        uploaded_files = st.file_uploader(
            "Válassz ki 1 vagy több fotót (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

    analyze_btn = st.button("🔍 Hirdetés elemzése MI-vel", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 MI Kiértékelés és Döntéstámogatás")

    if analyze_btn:
        temp_paths = []
        good_sample, bad_sample = create_demo_images()

        if sample_mode == "Demó teszt: Világos, éles szobakép":
            temp_paths = [good_sample]
        elif sample_mode == "Demó teszt: Sötét, életlen szobakép":
            temp_paths = [bad_sample]
        elif uploaded_files:
            for uf in uploaded_files:
                suffix = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    temp_paths.append(tmp.name)

        features = {
            "kerulet": kerulet,
            "alapterulet_nm": float(alapterulet),
            "szobak_szama": szobak,
            "allapot": allapot,
            "erkely": int(erkely),
            "lift": int(lift),
            "butorozott": int(butorozott),
        }

        res = engine.evaluate_listing(
            listing_features=features,
            advertised_price=int(advertised_price),
            description=description,
            image_paths=temp_paths,
        )

        # Fő verdikt megjelenítése
        if res["verdict_level"] == "success":
            st.success(f"**{res['verdict']}** — Összesített Deal Score: **{res['deal_score']} / 100**")
        elif res["verdict_level"] == "warning":
            st.warning(f"**{res['verdict']}** — Összesített Deal Score: **{res['deal_score']} / 100**")
        else:
            st.error(f"**{res['verdict']}** — Összesített Deal Score: **{res['deal_score']} / 100**")

        st.info(f"💡 **Ajánlás:** {res['advice']}")

        # Kiemelt mérőszámok
        m1, m2, m3 = st.columns(3)
        m1.metric("Hirdetett ár", f"{res['advertised_price']:,} Ft".replace(",", " "))
        m2.metric(
            "MI becsült reális ár",
            f"{res['predicted_price']:,} Ft".replace(",", " "),
            delta=f"{res['price_diff_pct']:+.1f}% eltérés",
            delta_color="inverse",
        )
        m3.metric("Deal Score", f"{res['deal_score']} / 100")

        if res["red_flags"]:
            st.markdown("**⚠️ Észlelt kockázatok (Red Flags):**")
            for flag in res["red_flags"]:
                st.markdown(f"- {flag}")

        # Részletes modul eredmények füleken
        tab1, tab2, tab3 = st.tabs(["📝 NLP Szövegelemzés", "📷 Képfeldolgozás (OpenCV)", "📈 Modell háttér"])

        with tab1:
            nlp = res["nlp_result"]
            st.write(f"**Szöveges megbízhatósági pontszám:** {nlp['nlp_score']} / 100 (Tisztított szavak száma: {nlp['word_count']})")
            st.write("**Pozitív kulcsszavak:**", ", ".join(nlp["positive_hits"]) if nlp["positive_hits"] else "Nincs")
            st.write("**Negatív / kockázati kifejezések:**", ", ".join(nlp["negative_hits"]) if nlp["negative_hits"] else "Nincs")
            st.write("**Legfontosabb TF-IDF kifejezések:**", ", ".join(nlp["tfidf_keywords"]) if nlp["tfidf_keywords"] else "Nincs")

        with tab2:
            img_res = res["image_result"]
            st.write(f"**Átlagos képi minőség pontszám:** {img_res['avg_image_score']} / 100")
            if temp_paths:
                cols = st.columns(min(len(temp_paths), 3))
                for idx, p in enumerate(temp_paths):
                    with cols[idx % 3]:
                        st.image(Image.open(p), caption=os.path.basename(p), use_container_width=True)
            for item in img_res["images"]:
                st.markdown(
                    f"- **{item['file']}**: Pontszám: `{item['image_score']}/100` | "
                    f"Fényerő: `{item['brightness']}` | Kontraszt: `{item['contrast']}` | "
                    f"Élesség (Laplacian): `{item['sharpness_laplacian']}` — *{', '.join(item['issues'])}*"
                )

        with tab3:
            if os.path.exists("reports/model_evaluation.png"):
                st.image("reports/model_evaluation.png", caption="Regressziós modellek összehasonlítása (Teszt halmaz)")

        # Ideiglenes fájlok takarítása
        if sample_mode == "Saját képek feltöltése":
            for p in temp_paths:
                if os.path.exists(p):
                    os.remove(p)
    else:
        st.write("Állítsd be a bal oldalon a hirdetés paramétereit, majd kattints a **Hirdetés elemzése MI-vel** gombra!")