"""Clase que resume un DataFrame de pandas en un archivo .txt."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class DatasetSummarizer:
    """Genera un archivo .txt (UTF-8) que describe el contenido de un DataFrame.

    Recibe los datos ya cargados; no se conecta a Hugging Face.

    Parámetros
    ----------
    output_dir : str | Path
        Carpeta de salida (ruta relativa al proyecto). Se crea si no existe.
    filename : str
        Nombre del archivo de resumen.
    """

    def __init__(self, output_dir: str | Path = "resultados",
                 filename: str = "resumen_dataframe.txt") -> None:
        self.output_dir = Path(output_dir)
        self.filename = filename

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.filename

    @staticmethod
    def count_empty(df: pd.DataFrame) -> Dict[str, int]:
        """Cuenta cadenas vacías y listas vacías por columna (no son NaN para pandas)."""
        def vacio(v):
            if isinstance(v, str):
                return v.strip() == ""
            return hasattr(v, "__len__") and len(v) == 0
        out = {}
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype) in ("string", "str"):
                out[c] = int(df[c].map(vacio).sum())
        return out

    def build_text(self, df: pd.DataFrame, source: Optional[str] = None,
                   description: Optional[str] = None,
                   metadata: Optional[Dict[str, object]] = None) -> str:
        """Construye el texto del resumen sin escribirlo en disco.

        ``metadata`` es un diccionario opcional (dataset, split, revisión, muestra...)
        que se imprime en la sección de origen.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")

        nulos = df.isna().sum()
        L = ["RESUMEN DEL DATAFRAME", "=" * 60,
             f"Generado el: {datetime.now():%Y-%m-%d %H:%M:%S}", "", "0. ORIGEN"]
        if source:
            L.append(f"- Origen: {source}")
        for clave, valor in (metadata or {}).items():
            if valor not in (None, ""):
                L.append(f"- {clave}: {valor}")
        if description:
            L.append(f"- Transformación aplicada: {description}")

        L += ["", "1. DIMENSIONES", f"- Filas: {df.shape[0]}", f"- Columnas: {df.shape[1]}",
              "", "2. COLUMNAS, TIPOS Y VALORES NULOS"]
        if df.shape[1] == 0:
            L.append("(el DataFrame no tiene columnas)")
        else:
            ancho = max(len(str(c)) for c in df.columns)
            L.append(f"{'Columna'.ljust(ancho)} | {'Tipo'.ljust(20)} | Nulos")
            L.append("-" * (ancho + 33))
            for col in df.columns:
                L.append(f"{str(col).ljust(ancho)} | {str(df[col].dtype).ljust(20)} | {int(nulos[col])}")

        total = int(nulos.sum())
        L += ["", f"Total de valores nulos: {total}",
              "- Hay valores nulos." if total else "- No hay valores nulos."]

        vacios = {c: n for c, n in self.count_empty(df).items() if n > 0}
        L += ["", "3. VALORES VACÍOS (cadena vacía o lista vacía; pandas no los cuenta como nulos)"]
        L += [f"- {c}: {n}" for c, n in vacios.items()] if vacios else ["- No se encontraron."]

        num = df.select_dtypes(include="number")
        L += ["", "4. ESTADÍSTICAS DESCRIPTIVAS (columnas numéricas)"]
        L.append(num.describe().to_string() if not num.empty and df.shape[0] > 0
                 else "(no hay columnas numéricas o filas)")

        otras = df.drop(columns=num.columns)
        if not otras.empty and df.shape[0] > 0:
            L += ["", "5. COLUMNAS NO NUMÉRICAS (valores distintos y más frecuente)"]
            for col in otras.columns:
                try:
                    top = otras[col].value_counts(dropna=True).head(1)
                    frec = f"{top.index[0]} ({int(top.iloc[0])})" if not top.empty else "n/d"
                    L.append(f"- {col}: {otras[col].nunique(dropna=True)} distintos; más frecuente: {frec}")
                except TypeError:   # listas u objetos no comparables
                    L.append(f"- {col}: tipo no resumible ({otras[col].dtype})")
        L.append("")
        return "\n".join(L)

    def summarize(self, df: pd.DataFrame, source: Optional[str] = None,
                  description: Optional[str] = None,
                  metadata: Optional[Dict[str, object]] = None) -> Path:
        """Escribe el resumen en ``output_dir/filename`` y devuelve la ruta.

        Crea la carpeta si no existe y sobrescribe el archivo en cada ejecución.
        """
        texto = self.build_text(df, source=source, description=description, metadata=metadata)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(texto, encoding="utf-8")
        return self.output_path
