"""Clase para obtener un dataset de Hugging Face como DataFrame de pandas."""
from __future__ import annotations

import random
from typing import List, Optional

import pandas as pd


class HFDatasetLoader:
    """Obtiene un dataset de Hugging Face Datasets y lo devuelve como DataFrame.

    Parámetros
    ----------
    dataset_id : str
        Identificador en el Hub, p. ej. ``"scikit-learn/iris"``.
    split : str, opcional
        Partición a cargar. Si no se indica y el dataset tiene un único split, se usa
        ese; si tiene varios, se pide indicarlo.
    config : str, opcional
        Configuración (subset), cuando el dataset define varias.
    columns : list[str], opcional
        Columnas a cargar (útil para omitir columnas pesadas o con datos personales).
    max_rows : int, opcional
        Si se indica, se carga una MUESTRA de hasta ``max_rows`` filas en lugar del
        dataset completo (necesario para datasets enormes). La muestra se toma de un
        archivo Parquet elegido al azar con ``seed`` y de varios grupos de filas
        repartidos en él; si no es posible, se usan las primeras filas por streaming.
    seed : int
        Semilla para la selección del archivo (reproducibilidad).
    row_groups : int
        Cantidad de grupos de filas repartidos en el archivo al muestrear.

    Tras ``load()`` quedan disponibles: ``df``, ``split_used``, ``revision``,
    ``is_sample``, ``sample_note`` y ``source_file``.
    """

    def __init__(self, dataset_id: str, split: Optional[str] = None,
                 config: Optional[str] = None, columns: Optional[List[str]] = None,
                 max_rows: Optional[int] = None, seed: int = 42, row_groups: int = 5) -> None:
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset_id debe ser un texto no vacío, p. ej. 'scikit-learn/iris'.")
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows debe ser un entero positivo.")
        self.dataset_id = dataset_id.strip()
        self.split = split
        self.config = config
        self.columns = list(columns) if columns else None
        self.max_rows = max_rows
        self.seed = seed
        self.row_groups = max(1, row_groups)

        self.df: Optional[pd.DataFrame] = None
        self.split_used: Optional[str] = split
        self.revision: Optional[str] = None
        self.source_file: Optional[str] = None
        self.is_sample: bool = False
        self.sample_note: str = ""

    # ------------------------------------------------------------------ API
    def load(self) -> pd.DataFrame:
        """Carga los datos (completos o muestra) y devuelve un DataFrame."""
        self.revision = self._get_revision()
        self.df = self._load_full() if self.max_rows is None else self._load_sample()
        return self.df

    # ------------------------------------------------------------ internos
    def _get_revision(self) -> Optional[str]:
        try:
            from huggingface_hub import HfApi
            return HfApi().dataset_info(self.dataset_id).sha
        except Exception:
            return None

    def _load_full(self) -> pd.DataFrame:
        from datasets import Dataset, DatasetDict, load_dataset
        try:
            data = load_dataset(self.dataset_id, self.config, split=self.split)
        except Exception as error:
            raise RuntimeError(
                f"No se pudo cargar '{self.dataset_id}' (split={self.split!r}, "
                f"config={self.config!r}): {error}") from error

        if isinstance(data, DatasetDict):
            nombres = list(data.keys())
            if len(nombres) != 1:
                raise ValueError(f"El dataset tiene varios splits {nombres}. Indique cuál usar con split=...")
            self.split_used = nombres[0]
            data = data[self.split_used]
        if not isinstance(data, Dataset):
            raise TypeError(f"Tipo de dataset no soportado: {type(data).__name__}")
        if self.columns:
            data = data.select_columns(self.columns)
        self.is_sample = False
        self.sample_note = "Dataset completo del split."
        return data.to_pandas()

    def _load_sample(self) -> pd.DataFrame:
        try:
            return self._sample_from_parquet()
        except Exception as error:
            print(f"[HFDatasetLoader] Muestreo por parquet no disponible ({error}); "
                  f"se usa streaming con las primeras filas.")
            return self._sample_from_stream()

    def _sample_from_parquet(self) -> pd.DataFrame:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, HfFileSystem

        archivos = sorted(f for f in HfApi().list_repo_files(self.dataset_id, repo_type="dataset")
                          if f.endswith(".parquet"))
        if self.split:
            por_split = [f for f in archivos if self.split in f]
            archivos = por_split or archivos
        if not archivos:
            raise FileNotFoundError("el repositorio no tiene archivos .parquet")

        archivo = random.Random(self.seed).choice(archivos)
        with HfFileSystem().open(f"datasets/{self.dataset_id}/{archivo}", "rb") as f:
            pf = pq.ParquetFile(f)
            n_grupos = pf.metadata.num_row_groups
            idx = sorted(set(np.linspace(0, n_grupos - 1, min(self.row_groups, n_grupos)).astype(int).tolist()))
            por_grupo = max(1, self.max_rows // len(idx))
            tablas = [pf.read_row_group(i, columns=self.columns).slice(0, por_grupo) for i in idx]
            df = pa.concat_tables(tablas).to_pandas()

        self.split_used = self.split or self.split_used
        self.source_file = archivo
        self.is_sample = True
        self.sample_note = (
            f"Archivo '{archivo}' elegido al azar (semilla {self.seed}) entre {len(archivos)} archivos "
            f"parquet; {len(idx)} grupos de filas repartidos de {n_grupos}; hasta {self.max_rows:,} filas. "
            f"No es una muestra aleatoria simple del dataset completo.")
        return df

    def _sample_from_stream(self) -> pd.DataFrame:
        from datasets import load_dataset
        try:
            data = load_dataset(self.dataset_id, self.config, split=self.split, streaming=True)
        except Exception as error:
            raise RuntimeError(f"No se pudo abrir '{self.dataset_id}' por streaming: {error}") from error
        if isinstance(data, dict):                       # IterableDatasetDict
            nombres = list(data.keys())
            if len(nombres) != 1:
                raise ValueError(f"El dataset tiene varios splits {nombres}. Indique cuál usar con split=...")
            self.split_used = nombres[0]
            data = data[self.split_used]
        if self.columns:
            data = data.select_columns(self.columns)
        df = pd.DataFrame(list(data.take(self.max_rows)))
        self.source_file = "streaming"
        self.is_sample = True
        self.sample_note = (f"Primeras {self.max_rows:,} filas por streaming; el orden original puede "
                            f"hacer que la muestra esté sesgada (p. ej. hacia los registros más antiguos).")
        return df

    def __repr__(self) -> str:
        return (f"HFDatasetLoader(dataset_id={self.dataset_id!r}, split={self.split!r}, "
                f"max_rows={self.max_rows!r})")
