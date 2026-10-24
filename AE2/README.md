# AE2 – Obtención y reconocimiento del dataset

Proyecto de analítica descriptiva con **datasocial/tiktok-5.6B-videos** (Hugging Face, licencia CC BY-NC 4.0).
El dataset completo pesa ~460 GB: el notebook trabaja con una **muestra** documentada.

## Estructura
```
AE2/
├── AE2_Exploracion_Dataset.ipynb   # carga de la muestra, descripción básica y resumen
├── requirements.txt
├── documento/Documento_AE2_Necesidad_y_Diseno.docx
└── resultados/resumen_dataframe.txt   # se genera al ejecutar el notebook
```

## Cómo ejecutar
1. `pip install -r requirements.txt`
2. Abrir el notebook desde la carpeta `AE2/` y ejecutar todas las celdas (*Run All*).
3. Se crea/regenera `resultados/resumen_dataframe.txt` (UTF-8).
4. Subir a GitHub el notebook ejecutado, el `.txt` y el `.docx`.

Ajustes: `N_FILAS` (tamaño de la muestra) y `SEED` en la celda de parámetros.
No se cargan `caption`, `mentions`, `on_screen_text` ni `hashtag_ids` por privacidad.
