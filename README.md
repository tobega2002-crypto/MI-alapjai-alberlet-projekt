# RentCheck AI – Intelligens Albérlet-értékbecslő és Hirdetés-elemző

Multimodális mesterséges intelligencia alapú döntéstámogató rendszer, amely lakáshirdetések piaci árát, szöveges megbízhatóságát és fotóinak vizuális minőségét értékeli ki.

## Alkalmazott MI témakörök
1. **Adatelőkészítés és adatbányászat (`pandas`, `scikit-learn`):** Hiányzó értékek imputálása, skálázás, One-Hot kódolás.
2. **Felügyelt gépi tanulás – Regresszió (`LinearRegression`, `RandomForestRegressor`):** Reális havi bérleti díj becslése ($R^2 = 0.865$).
3. **Természetes nyelvfeldolgozás – NLP (`TfidfVectorizer`, kulcsszó-elemzés):** Szöveges leírások kockázat- és előnyelemzése.
4. **Számítógépes látás – Computer Vision (`OpenCV`):** Fényerő, RMS kontraszt, színtelítettség és Laplacian-alapú képélesség mérése.

## Telepítés és futtatás
```bash
pip install -r requirements.txt
python src/data_generator.py
python src/price_model.py
streamlit run app.py