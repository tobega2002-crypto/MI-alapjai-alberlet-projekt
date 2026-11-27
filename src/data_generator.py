import numpy as np
import pandas as pd
import os

def generate_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # Budapesti kerületek és alap négyzetméterár-szorzóik (Ft/m2/hó körülbelüli arány)
    districts = {
        "I. kerület": 5200,
        "II. kerület": 5100,
        "V. kerület": 5800,
        "VI. kerület": 5000,
        "VII. kerület": 4700,
        "VIII. kerület": 4100,
        "IX. kerület": 4600,
        "XI. kerület": 4800,
        "XIII. kerület": 4900,
        "XIV. kerület": 4200,
    }

    conditions = ["Új építésű", "Felújított", "Jó állapotú", "Felújítandó"]
    condition_mult = {
        "Új építésű": 1.18,
        "Felújított": 1.10,
        "Jó állapotú": 1.00,
        "Felújítandó": 0.78,
    }

    district_choices = np.random.choice(list(districts.keys()), size=n_samples)
    areas = np.random.randint(24, 115, size=n_samples)
    rooms = np.clip(np.round(areas / 28).astype(int), 1, 4)
    cond_choices = np.random.choice(conditions, size=n_samples, p=[0.2, 0.35, 0.35, 0.1])
    has_balcony = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
    has_elevator = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    furnished = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])

    prices = []
    descriptions = []

    pos_phrases = [
        "világos, napfényes lakás",
        "alacsony rezsi, egyedi mérőórákkal",
        "kiváló tömegközlekedés, metró 2 perc",
        "teljesen gépesített és bútorozott",
        "azonnal költözhető, frissen festve",
        "klimatizált, modern kialakítás",
    ]
    neg_phrases = [
        "felújításra szoruló állapot",
        "magas közös költség, régi nyílászárók",
        "sötét, belső udvarra néző",
        "bútorozatlan, gépesítés nélkül",
        "3 havi kaució szükséges, csak hosszú távra",
        "zajongó utcafront, lift nincs",
    ]

    for i in range(n_samples):
        base_rate = districts[district_choices[i]]
        mult = condition_mult[cond_choices[i]]
        
        # Reális piaci ár számítása
        fair_price = (
            areas[i] * base_rate * mult
            + rooms[i] * 12000
            + has_balcony[i] * 15000
            + has_elevator[i] * 10000
            + furnished[i] * 18000
        )
        
        # Piaci zaj (+/- 8%) + néhány szándékosan túlárazott vagy alulárazott hirdetés
        noise = np.random.normal(1.0, 0.06)
        if np.random.rand() < 0.12:  # 12% túlárazott hirdetés
            noise = np.random.uniform(1.22, 1.45)
        elif np.random.rand() < 0.08:  # 8% alkalmi vétel / gyanúsan olcsó
            noise = np.random.uniform(0.68, 0.82)

        final_price = int(round((fair_price * noise) / 1000) * 1000)
        prices.append(final_price)

        # Szöveges leírás összeállítása az állapot alapján
        if cond_choices[i] in ["Új építésű", "Felújított"]:
            chosen_phrases = np.random.choice(pos_phrases, size=3, replace=False)
        elif cond_choices[i] == "Felújítandó":
            chosen_phrases = np.random.choice(neg_phrases, size=3, replace=False)
        else:
            chosen_phrases = [
                np.random.choice(pos_phrases),
                np.random.choice(neg_phrases),
            ]

        desc = (
            f"Kiadó a {district_choices[i]}ben egy {areas[i]} nm-es, {rooms[i]} szobás, "
            f"{cond_choices[i].lower()} ingatlan. Jellemzők: {', '.join(chosen_phrases)}."
        )
        descriptions.append(desc)

    df = pd.DataFrame({
        "kerulet": district_choices,
        "alapterulet_nm": areas.astype(float),
        "szobak_szama": rooms,
        "allapot": cond_choices,
        "erkely": has_balcony,
        "lift": has_elevator,
        "butorozott": furnished,
        "leiras": descriptions,
        "ar_ft": prices,
    })

    # Szándékos hiányzó értékek (NaN) beszúrása az adattisztítás demonstrálásához
    nan_indices = np.random.choice(df.index, size=25, replace=False)
    df.loc[nan_indices[:15], "alapterulet_nm"] = np.nan
    df.loc[nan_indices[15:], "allapot"] = np.nan

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "alberletek.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Adathalmaz sikeresen legenerálva: {output_path} ({len(df)} sor)")
    return df

if __name__ == "__main__":
    generate_dataset()