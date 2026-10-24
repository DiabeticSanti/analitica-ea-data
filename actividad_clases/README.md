# Actividad de clases (continuación de AE2)

`HFDatasetLoader` → DataFrame → transformación mínima → `DatasetSummarizer` → `resultados/resumen_dataframe.txt`

- `hf_dataset_loader.py`: clase `HFDatasetLoader(dataset_id, split, config, columns, max_rows, seed)`.
- `dataset_summarizer.py`: clase `DatasetSummarizer(output_dir, filename)`.
- `actividad_clases_tiktok.ipynb`: coordina ambas con el dataset de AE2 y selecciona las 9 variables del diseño del análisis.

Ejecutar: `pip install -r requirements.txt`, abrir el notebook **desde esta carpeta** y *Run All*.
