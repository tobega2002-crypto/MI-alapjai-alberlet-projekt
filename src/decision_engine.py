import os
import joblib
import pandas as pd
from typing import Dict, List

from src.nlp_analyzer import analyze_listing_text
from src.image_analyzer import analyze_image_batch


class RentCheckEngine:
    def __init__(
        self,
        model_path: str = "models/price_model.pkl",
        data_path: str = "data/alberletek.csv",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"A betanított árbecslő modell nem található: {model_path}. Futtasd előbb a src/price_model.py fájlt!"
            )
        self.price_model = joblib.load(model_path)

        # Referencia szövegkorpusz betöltése a TF-IDF súlyozáshoz
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            self.reference_corpus = df["leiras"].dropna().tolist()[:300]
        else:
            self.reference_corpus = []

    def evaluate_listing(
        self,
        listing_features: Dict,
        advertised_price: int,
        description: str,
        image_paths: List[str],
    ) -> Dict:
        """
        Teljes körű multimodális kiértékelés:
        1. ML Regresszió -> Reális piaci bérleti díj becslése
        2. NLP modul -> Szöveges leírás minősége és kulcsszavak
        3. OpenCV modul -> Fotók fényereje, kontrasztja, élessége
        4. Üzleti logika -> Áreltérés + Deal Score + Döntési ajánlás
        """
        # 1. Árbecslés (Regresszió)
        input_df = pd.DataFrame([listing_features])
        predicted_price = float(self.price_model.predict(input_df)[0])
        predicted_price_rounded = int(round(predicted_price / 1000) * 1000)

        price_diff_ft = advertised_price - predicted_price_rounded
        price_diff_pct = ((advertised_price - predicted_price) / predicted_price) * 100

        # Ár-pontszám (0 - 100): 0% eltérés = 75 pont; olcsóbb = több pont, drágább = kevesebb pont
        price_score = max(0.0, min(100.0, 75.0 - (price_diff_pct * 1.8)))

        # 2. Szövegelemzés (NLP)
        nlp_result = analyze_listing_text(description, self.reference_corpus)
        nlp_score = nlp_result["nlp_score"]

        # 3. Képelemzés (Computer Vision)
        image_result = analyze_image_batch(image_paths)
        image_score = image_result["avg_image_score"]

        # 4. Súlyozott Deal Score számítása (50% Ár, 25% Szöveg, 25% Képek)
        deal_score = int(round(0.50 * price_score + 0.25 * nlp_score + 0.25 * image_score))

        # Üzleti döntési szabályrendszer (Red Flag / Best Deal / Fair Price)
        red_flags = []
        if price_diff_pct > 15:
            red_flags.append(f"A hirdetett ár {price_diff_pct:.1f}%-kal magasabb a becsült piaci árnál.")
        if price_diff_pct < -22 and (nlp_score < 50 or image_score < 55):
            red_flags.append("Gyanúsan alacsony ár gyenge minőségű leírással vagy fotókkal (lehetséges csalás / rejtett hiba).")
        if nlp_score < 45:
            red_flags.append("A hirdetés szövege kockázatos kulcsszavakat tartalmaz vagy hiányos.")
        if image_score < 55:
            red_flags.append("A feltöltött fotók túl sötétek vagy életlenek.")

        if price_diff_pct < -22 and (nlp_score < 50 or image_score < 55):
            verdict = "GYANÚS HIRDETÉS (Red Flag)"
            verdict_level = "error"
            advice = (
                "Az ár irreálisan alacsony a kerületi átlaghoz képest, miközben a leírás vagy a képek "
                "minősége gyenge. Fokozott óvatosság ajánlott (ne utalj előre foglalót!)."
            )
        elif deal_score >= 76 and price_diff_pct <= 5:
            verdict = "KIVÁLÓ VÉTEL (Best Deal)"
            verdict_level = "success"
            advice = (
                "Kedvező ár-érték arányú ajánlat: a bérleti díj a piaci érték alatt vagy annak közelében van, "
                "a leírás és a fotók is megbízhatóak. Érdemes mielőbb megtekinteni!"
            )
        elif deal_score >= 52 and price_diff_pct <= 15:
            verdict = "REÁLIS AJÁNLAT (Fair Price)"
            verdict_level = "warning"
            advice = (
                "Piaci árkategóriába illeszkedő hirdetés. A kisebb hiányosságok vagy az enyhe felár miatt "
                "érdemes rákérdezni a pontos rezsiköltségre és alkudni a bérleti díjból."
            )
        else:
            verdict = "TÚLÁRAZOTT / KOCKÁZATOS (Overpriced)"
            verdict_level = "error"
            advice = (
                "A hirdetés ár-érték aránya kedvezőtlen: a kért bérleti díj meghaladja a paraméterek alapján "
                "indokolt szintet, vagy jelentős minőségi kockázatok látszanak."
            )

        return {
            "advertised_price": advertised_price,
            "predicted_price": predicted_price_rounded,
            "price_diff_ft": price_diff_ft,
            "price_diff_pct": round(price_diff_pct, 1),
            "price_score": int(round(price_score)),
            "nlp_result": nlp_result,
            "image_result": image_result,
            "deal_score": deal_score,
            "verdict": verdict,
            "verdict_level": verdict_level,
            "advice": advice,
            "red_flags": red_flags,
        }