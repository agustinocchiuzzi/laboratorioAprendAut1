"""Arbol de decision ID3 implementado para la materia.

Sigue el pseudocodigo del curso: entropia para medir el desorden, ganancia de
informacion para elegir el atributo por el que corta cada nodo, y reglas de
parada (misma clase, sin atributos, rama sin ejemplos, ganancia menor al
minimo o profundidad maxima) para no sobreajustar.
"""

import numpy as np


class Nodo:
    """Un nodo del arbol: puede ser pregunta (con ramas) o hoja.

    Un nodo nace como hoja con la clase mayoritaria del grupo, su respaldo:
    es la prediccion si queda como hoja, la de una rama sin ejemplos y la que
    usa _clasificar cuando un valor no se vio en train.
    """

    def __init__(self, clase, conteos):
        self.clase = clase
        self.conteos = conteos
        self.atributo = None  # si es None, el nodo es una hoja
        self.ramas = {}       # valor del atributo y su nodo hijo


class ID3:
    """Clasificador ID3. X: enteros (atributos discretizados), y: clases."""

    def __init__(self, min_info_gain=0.0, max_depth=8):
        self.min_info_gain = min_info_gain
        self.max_depth = max_depth

    def fit(self, X, y):
        """Construye el arbol con los datos de entrenamiento."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.clases_ = self.classes_
        self.ganancia_total_ = np.zeros(X.shape[1])
        self.raiz_ = self._construir(X, y, list(range(X.shape[1])), 0)

        # Importancia de cada atributo: ganancia acumulada normalizada.
        total = self.ganancia_total_.sum()
        self.feature_importances_ = (
            self.ganancia_total_ / total if total > 0 else self.ganancia_total_.copy()
        )
        return self

    def _construir(self, X, y, atributos, profundidad):
        """Devuelve el subarbol de estos ejemplos.

        El arbol se ensambla de abajo hacia arriba: cada llamada devuelve su
        nodo y el padre lo cuelga en nodo.ramas. Se corta cuando no queda
        nada por dividir o el corte no aporta informacion.
        """
        conteos = {clase: int(np.sum(y == clase)) for clase in self.clases_}
        clase_mas_comun = max(conteos, key=conteos.get)
        nodo = Nodo(clase_mas_comun, conteos)

        # Regla 1: todos los ejemplos con la misma clase son hoja.
        if len(np.unique(y)) == 1:
            return nodo
        # Regla 2: no quedan atributos: hoja con la clase mayoritaria.
        if not atributos:
            return nodo
        # Regla 4 (adicional): no se supera la profundidad maxima.
        if self.max_depth is not None and profundidad >= self.max_depth:
            return nodo

        # Se elige el atributo disponible con mayor ganancia.
        entropia_de_s = self._entropia(y)
        mejor_atributo = None
        mejor_ganancia = -1.0
        for atributo in atributos:
            ganancia = self._ganancia(X, y, atributo, entropia_de_s)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo

        # Regla 4: sin ganancia suficiente: hoja (poda por ganancia).
        if mejor_atributo is None or mejor_ganancia <= self.min_info_gain:
            return nodo

        nodo.atributo = mejor_atributo
        self.ganancia_total_[mejor_atributo] += mejor_ganancia * len(y)
        atributos_restantes = [a for a in atributos if a != mejor_atributo]

        # Una rama por cada valor distinto del atributo (multiway).
        for valor in np.unique(X[:, mejor_atributo]):
            ejemplos_en_rama = X[:, mejor_atributo] == valor
            sub_X, sub_y = X[ejemplos_en_rama], y[ejemplos_en_rama]

            # Llave int de Python: coincide con la que busca _clasificar.
            valor_llave = int(valor)

            # Regla 3: rama sin ejemplos: hoja con la clase del padre.
            if len(sub_y) == 0:
                nodo.ramas[valor_llave] = Nodo(clase_mas_comun, conteos)
            else:
                nodo.ramas[valor_llave] = self._construir(
                    sub_X, sub_y, atributos_restantes, profundidad + 1
                )
        return nodo

    def _entropia(self, y):
        """Desorden de un conjunto de etiquetas: 0 si son todas iguales."""
        total = len(y)
        entropia = 0.0
        for clase in self.clases_:
            proporcion = np.sum(y == clase) / total
            if proporcion > 0:
                entropia -= proporcion * np.log2(proporcion)
        return entropia

    def _ganancia(self, X, y, atributo, entropia_de_s):
        """Cuanta entropia se ahorra al conocer el valor del atributo."""
        total = len(y)
        entropia_condicional = 0.0
        for valor in np.unique(X[:, atributo]):
            ejemplos_en_rama = X[:, atributo] == valor
            proporcion = np.sum(ejemplos_en_rama) / total
            entropia_condicional += proporcion * self._entropia(y[ejemplos_en_rama])
        return entropia_de_s - entropia_condicional

    def predict(self, X):
        """Clasifica cada fila bajando por el arbol hasta una hoja."""
        X = np.asarray(X)
        return np.array([self._clasificar(fila) for fila in X])

    def _clasificar(self, fila):
        nodo = self.raiz_
        while nodo.atributo is not None:
            valor = int(fila[nodo.atributo])
            hijo = nodo.ramas.get(valor)
            if hijo is None:
                # Valor que no aparecio en train: se usa el respaldo del nodo.
                break
            nodo = hijo
        return nodo.clase

    def get_depth(self):
        """Niveles hasta la hoja mas lejana."""
        return self._profundidad(self.raiz_)

    def _profundidad(self, nodo):
        if not nodo.ramas:
            return 0
        return 1 + max(self._profundidad(hijo) for hijo in nodo.ramas.values())

    def get_n_leaves(self):
        """Cantidad de hojas (clasificaciones posibles)."""
        return self._cantidad_de_hojas(self.raiz_)

    def _cantidad_de_hojas(self, nodo):
        if not nodo.ramas:
            return 1
        return sum(self._cantidad_de_hojas(hijo) for hijo in nodo.ramas.values())