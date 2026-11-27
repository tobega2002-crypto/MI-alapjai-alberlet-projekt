import os
import cv2
import numpy as np
from typing import Dict, List


def analyze_single_image(image_path: str) -> Dict:
    """
    Egyetlen lakásfotó minőségi elemzése OpenCV segítségével:
    - Fényerő (szürkeárnyalatos átlag)
    - Kontraszt (szórás)
    - Élesség (Laplacian operátor varianciája)
    - Színtelítettség (HSV S-csatorna átlag)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Nem sikerült beolvasni a képet: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Fényerő (0 - 255)
    brightness = float(np.mean(gray))

    # 2. Kontraszt (intenzitás szórása)
    contrast = float(np.std(gray))

    # 3. Élesség (Laplacian variancia - minél nagyobb, annál élesebb a kép)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 4. Átlagos színtelítettség (0 - 255)
    saturation = float(np.mean(hsv[:, :, 1]))

    # Pontszámítás (0 - 100)
    score = 100.0
    issues = []

    if brightness < 75:
        score -= 28
        issues.append("Túl sötét felvétel (gyenge megvilágítás)")
    elif brightness > 225:
        score -= 15
        issues.append("Túlexponált / kiégett világos részek")

    if contrast < 32:
        score -= 18
        issues.append("Alacsony kontraszt, fakó kép")

    if laplacian_var < 85:
        score -= 30
        issues.append("Elmosódott / életlen fotó (alacsony Laplacian-variancia)")

    if saturation < 25:
        score -= 10
        issues.append("Színtelen, szürkés árnyalatvilág")

    final_score = int(max(0, min(100, round(score))))

    return {
        "file": os.path.basename(image_path),
        "image_score": final_score,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness_laplacian": round(laplacian_var, 1),
        "saturation": round(saturation, 1),
        "issues": issues if issues else ["Megfelelő megvilágítás és képélesség"],
    }


def analyze_image_batch(image_paths: List[str]) -> Dict:
    """Több feltöltött lakásfotó összesített vizuális értékelése."""
    if not image_paths:
        return {
            "avg_image_score": 40,
            "images": [],
            "summary_issues": ["Nincs feltöltött fotó a hirdetéshez"],
        }

    results = []
    all_issues = []
    for path in image_paths:
        res = analyze_single_image(path)
        results.append(res)
        for iss in res["issues"]:
            if "Megfelelő" not in iss and iss not in all_issues:
                all_issues.append(iss)

    avg_score = int(round(np.mean([r["image_score"] for r in results])))
    return {
        "avg_image_score": avg_score,
        "images": results,
        "summary_issues": all_issues if all_issues else ["Jó minőségű, világos és éles fotók"],
    }


def create_demo_images(output_dir: str = "data"):
    """Két szintetikus tesztképet generál (egy világos-éles és egy sötét-homályos szobaképet)."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Világos, éles, kontrasztos szoba szimuláció
    good_img = np.full((400, 600, 3), (210, 220, 230), dtype=np.uint8)
    cv2.rectangle(good_img, (50, 60), (220, 260), (255, 250, 230), -1)   # ablak
    cv2.rectangle(good_img, (260, 180), (540, 350), (90, 130, 170), -1)  # kanapé
    cv2.rectangle(good_img, (80, 300), (230, 380), (60, 90, 120), -1)    # asztal
    cv2.putText(good_img, "Nappali - Vilagos", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    good_path = os.path.join(output_dir, "sample_good_room.jpg")
    cv2.imwrite(good_path, good_img)

    # 2. Sötét, életlen szoba szimuláció
    bad_img = np.full((400, 600, 3), (42, 44, 48), dtype=np.uint8)
    cv2.rectangle(bad_img, (260, 180), (540, 350), (55, 58, 62), -1)
    bad_img = cv2.GaussianBlur(bad_img, (21, 21), 0)
    bad_path = os.path.join(output_dir, "sample_dark_blurry.jpg")
    cv2.imwrite(bad_path, bad_img)

    return good_path, bad_path


if __name__ == "__main__":
    good_p, bad_p = create_demo_images()
    print("Jó kép elemzése:", analyze_single_image(good_p))
    print("Rossz kép elemzése:", analyze_single_image(bad_p))