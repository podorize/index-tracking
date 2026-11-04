"""Caches em pickle legíveis em qualquer pandas 2.x ou 3.x.

Um pickle de DataFrame depende da versão do pandas que o gravou. Aqui, DataFrames, Series e índices viram arrays do numpy (texto como
`object`, datas em nanossegundos) e são remontados na leitura; o resto da estrutura (dicionários, listas, números, textos) fica igual.
"""
import pickle

import numpy as np
import pandas as pd

MARCA = "__pandas__"


def _array(valores):
    a = np.asarray(valores)
    if a.dtype.kind == "M":
        return a.astype("datetime64[ns]")
    if a.dtype.kind in "OUT" or str(getattr(valores, "dtype", "")) in ("str", "string"):
        return np.asarray(valores, dtype=object)
    return a


def _indice(i):
    if isinstance(i, pd.MultiIndex):
        return {MARCA: "multi", "niveis": [_array(i.get_level_values(k)) for k in range(i.nlevels)], "nomes": list(i.names)}
    return {MARCA: "indice", "valores": _array(i), "nome": i.name}


def codificar(x):
    if isinstance(x, pd.DataFrame):
        return {MARCA: "df", "colunas": _indice(x.columns), "indice": _indice(x.index),
                "dados": [_array(x.iloc[:, k]) for k in range(x.shape[1])]}
    if isinstance(x, pd.Series):
        return {MARCA: "serie", "dados": _array(x), "indice": _indice(x.index), "nome": x.name}
    if isinstance(x, pd.Index):
        return _indice(x)
    if isinstance(x, dict):
        return {k: codificar(v) for k, v in x.items()}
    if isinstance(x, list):
        return [codificar(v) for v in x]
    if isinstance(x, tuple):
        return tuple(codificar(v) for v in x)
    if isinstance(x, str):
        return str(x)
    return x


def _remontar_indice(d):
    if d[MARCA] == "multi":
        return pd.MultiIndex.from_arrays(d["niveis"], names=d["nomes"])
    return pd.Index(d["valores"], name=d["nome"])


def decodificar(x):
    if isinstance(x, dict) and MARCA in x:
        tipo = x[MARCA]
        if tipo == "df":
            df = pd.DataFrame({k: v for k, v in enumerate(x["dados"])}, index=_remontar_indice(x["indice"]))
            df.columns = _remontar_indice(x["colunas"])
            return df
        if tipo == "serie":
            return pd.Series(x["dados"], index=_remontar_indice(x["indice"]), name=x["nome"])
        return _remontar_indice(x)
    if isinstance(x, dict):
        return {k: decodificar(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decodificar(v) for v in x]
    if isinstance(x, tuple):
        return tuple(decodificar(v) for v in x)
    return x


def gravar(objeto, caminho):
    with open(caminho, "wb") as f:
        pickle.dump(codificar(objeto), f, protocol=4)


def ler(caminho):
    with open(caminho, "rb") as f:
        return decodificar(pickle.load(f))
