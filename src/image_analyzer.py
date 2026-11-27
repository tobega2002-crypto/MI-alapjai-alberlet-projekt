import os
import cv2
import numpy as np
from typing import Dict, List


def analyze_single_image(image_path: str) -> Dict:
    """
    Valódi ingatlanfotókra kalibrált OpenCV képminőség-elemző:
    - Egységes 640px szélességre skálázás (felbontásfüggetlen mérés)
    - Fényerő és sötét területek aránya
    - Kontraszt (hisztogram-terjedés)
    - Skálázott élesség (Laplacian variancia)
    - Színhőmérséklet és telítettség
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Nem sikerült beolvasni a képet: {image_path}")

    # Egységes méretre skálázás, hogy a HD és kis képek ugyanazt a skálát adják
    h, w = img.shape[:2]
    target_w = 640
    target_h = int(h * (target_w / w))
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # 1. Átlagos fényerő és sötét pixelek aránya
    brightness = float(np.mean(gray))
    dark_pixel_ratio = float(np.mean(gray < 50)) * 100  # %

    # 2. Kontraszt (10. és 90. percentilis különbsége - robusztusabb a kiégett ablakokra)
    p10, p90 = np.percentile(gray, [10, 90])
    contrast = float(p90 - p10)

    # 3. Élesség (640px-re normalizált Laplacian variancia)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 4. Átlagos színtelítettség
    saturation = float(np.mean(hsv[:, :, 1]))

    # Kalibrált pontozás (valós webes lakásfotókhoz igazítva)
    score = 92.0
    issues = []

    if brightness < 85 or dark_pixel_ratio > 35:
        score -= 25
        issues.append("Sötét, alulvilágított helyiség (magas árnyékarány)")
    elif brightness > 230:
        score -= 12
        issues.append("Túlexponált, vakítóan világos kép")
    elif 115 <= brightness <= 195:
        score += 5  # Ideális, természetes megvilágítás jutalmazása

    if contrast < 65:
        score -= 15
        issues.append("Alacsony kontraszt, lapos/fakó felvétel")

    # 640px szélességnél a valóban elmosódott képek jellemzően 35 alatt vannak
    if laplacian_var < 35:
        score -= 25
        issues.append("Elmosódott, életlen fotó")
    elif laplacian_var > 120:
        score += 3  # Részletgazdag, éles kép

    if saturation < 15:
        score -= 8
        issues.append("Szinte teljesen szürkeárnyalatos / fakó színvilág")

    final_score = int(max(0, min(100, round(score))))

    return {
        "file": os.path.basename(image_path),
        "image_score": final_score,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness_laplacian": round(laplacian_var, 1),
        "saturation": round(saturation, 1),
        "issues": issues if issues else ["Megfelelő megvilágítás, kontraszt és képélesség"],
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

    good_img = np.full((400, 600, 3), (175, 185, 195), dtype=np.uint8)
    cv2.rectangle(good_img, (50, 60), (220, 260), (245, 240, 220), -1)
    cv2.rectangle(good_img, (260, 180), (540, 350), (70, 110, 150), -1)
    cv2.rectangle(good_img, (80, 300), (230, 380), (50, 80, 110), -1)
    cv2.putText(good_img, "Nappali - Vilagos", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    good_path = os.path.join(output_dir, "sample_good_room.jpg")
    cv2.imwrite(good_path, good_img)

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