"""Arbol de decision ID3, implementado para la materia.

Sigue el algoritmo de las notas del curso:

- Para elegir el atributo se usa la entropia de Shannon y la ganancia de
  informacion:

      Entropy(S) = - sum_c P(c) * log2(P(c))                      (clase c)
      Gain(S, A) = Entropy(S) - sum_v (|S_v| / |S|) * Entropy(S_v) (valor v)

- El arbol es "multiway": de cada nodo sale una rama por cada valor posible
  del atributo elegido.
- Un atributo se usa a lo sumo una vez en cada rama.
- Como se arma: fit() llama a _construir en la raiz; cada llamada representa
  un subconjunto de ejemplos y devuelve el SUBARBOL de ese subconjunto.
  Las llamadas bajan en profundidad (hasta las hojas), pero el arbol se
  "ensambla" de abajo hacia arriba: cada subarbol devuelto se cuelga del
  padre en el diccionario nodo.ramas.
- Reglas de parada (igual que el pseudocodigo del curso):
    1) todos los ejemplos tienen la misma clase  ->  hoja con esa clase;
    2) no quedan atributos                        ->  hoja con la clase mas comun;
    3) una rama queda sin ejemplos                ->  hoja con la clase mas comun
       del nodo padre;
    4) (adicional) mejor ganancia no supera min_info_gain o se llego al limite
       de profundidad -> hoja. Esto evita el sobreajuste.
"""

import numpy as np


class Nodo:
    """Un nodo del arbol.

    Un nodo "nace" como hoja (atributo = None) y solo se transforma en
    pregunta si _construir encuentra un atributo con suficiente ganancia.

    Atributos:
        clase (str): clase mayoritaria de los ejemplos de este nodo. Es el
            RESPALDO del nodo: se calcula siempre, "por las dudas". Sirve
            para 3 cosas:
            1) como prediccion si este nodo queda como hoja;
            2) como prediccion de una rama que quedo sin ejemplos;
            3) en _clasificar si un valor nunca se vio en train.
        conteos (dict): {clase: cantidad de ejemplos} de este nodo.
        atributo (int | None): indice del atributo por el que pregunta este
            nodo (la columna de X que se mira). Si es None, es una hoja:
            no tiene pregunta y su clase es la prediccion final.
        ramas (dict): {valor del atributo: nodo hijo}. Es el "menu de
            respuestas" de la pregunta: por cada valor posible del atributo
            hay un subarbol (otro Nodo). Una hoja tiene ramas = {} vacio.
    """

    def __init__(self, clase, conteos):
        self.clase = clase
        self.conteos = conteos
        self.atributo = None
        self.ramas = {}


class ID3:
    """Clasificador ID3 que recibe una matriz de atributos discretos.

    X: matriz de enteros, una fila por ejemplo y una columna por atributo.
    y: clases (por ejemplo: "E", "L", "V").
    """

    def __init__(self, min_info_gain=0.0, max_depth=8):
        self.min_info_gain = min_info_gain  # minimo de ganancia para seguir cortando
        self.max_depth = max_depth          # profundidad maxima del arbol (None = sin tope)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Construye el arbol con los datos de entrenamiento."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.clases_ = np.unique(y)                   # valores posibles de la clase
        self.ganancia_total_ = np.zeros(X.shape[1])   # suma de ganancia por atributo
        self.raiz_ = self._construir(X, y, list(range(X.shape[1])), 0)

        # Importancia = ganancia acumulada de cada atributo, normalizada a [0,1].
        total = self.ganancia_total_.sum()
        self.feature_importances_ = (
            self.ganancia_total_ / total if total > 0 else self.ganancia_total_.copy()
        )
        return self

    def _construir(self, X, y, atributos, profundidad):
        """Crea y devuelve el nodo que representa a estos ejemplos.

        Pasos, siempre los mismos en cada nodo:
            1) contar las clases de estos ejemplos y la clase mas comun
               (ese mayoritario queda como respaldo en nodo.clase);
            2) aplicar las reglas de parada: si toca, devolver una HOJA;
            3) calcular la entropia de este conjunto (cuanta "mezcla" hay);
            4) calcular la ganancia de cada atributo disponible y elegir
               el de mayor ganancia;
            5) si vale la pena, dividir: por cada valor distinto del
               atributo (np.unique), llamarse a si mismo con esa particion
               y colgar el resultado en nodo.ramas.

        Ojo con la recursion: las llamadas bajan en profundidad hasta las
        hojas, pero el arbol se ENSAMBLA de abajo hacia arriba. Cada
        llamada devuelve su subarbol y el padre lo engancha con
        "nodo.ramas[valor] = subarbol". Por eso la raiz se "completa"
        recien cuando terminaron todos sus descendientes.
        """
        # Cuenta cuantas veces aparece cada clase entre estos ejemplos.
        conteos = {clase: int(np.sum(y == clase)) for clase in self.clases_}
        # Respaldo: si este nodo no puede decidir con mas informacion,
        # predice esto. En empate gana la primera clase en orden alfabetico
        # (self.clases_ = np.unique(y) viene ordenado): criterio fijo, sin
        # azar, para que el resultado sea reproducible.
        clase_mas_comun = max(conteos, key=conteos.get)

        # El nodo nace como hoja; solo se convierte en nodo interno si
        # encontramos un atributo que aporte suficiente informacion.
        nodo = Nodo(clase_mas_comun, conteos)

        # Regla 1: todos los ejemplos con la misma clase -> hoja.
        if len(np.unique(y)) == 1:
            return nodo

        # Regla 2: no quedan atributos -> hoja con la clase mas comun.
        if not atributos:
            return nodo

        # No seguir creciendo si se alcanzo la profundidad maxima.
        if self.max_depth is not None and profundidad >= self.max_depth:
            return nodo

        # Se calcula la ganancia de cada atributo disponible y se elige el mejor.
        entropia_de_s = self._entropia(y)
        mejor_atributo = None
        mejor_ganancia = -1.0
        for atributo in atributos:
            ganancia = self._ganancia(X, y, atributo, entropia_de_s)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo

        # Regla 4: ninguna ganancia supera el minimo -> hoja (poda por ganancia).
        if mejor_atributo is None or mejor_ganancia <= self.min_info_gain:
            return nodo

        # El nodo pasa a tener pregunta y el atributo deja de usarse en esta rama.
        nodo.atributo = mejor_atributo
        self.ganancia_total_[mejor_atributo] += mejor_ganancia * len(y)

        atributos_restantes = [a for a in atributos if a != mejor_atributo]
        # np.unique(...) devuelve los VALORES DISTINTOS de esa columna
        # (p.ej. [1 2 3] cuando el atributo es baja/media/alta): por cada
        # valor posible creamos una rama (multiway).
        for valor in np.unique(X[:, mejor_atributo]):
            ejemplos_en_rama = X[:, mejor_atributo] == valor
            sub_X, sub_y = X[ejemplos_en_rama], y[ejemplos_en_rama]

            # La rama guarda la llave como 'int(valor)' (np.int64 -> int de
            # Python) para que SIEMPRE coincida con la llave que busca
            # _clasificar al predecir.

            # Regla 3: rama sin ejemplos -> hoja con la clase mas comun del nodo.
            if len(sub_y) == 0:
                nodo.ramas[int(valor)] = Nodo(clase_mas_comun, conteos)
            else:
                # Aqui se ve el ensamblaje: la llamada hija devuelve el
                # subarbol y el padre lo cuelga como rama.
                nodo.ramas[int(valor)] = self._construir(
                    sub_X, sub_y, atributos_restantes, profundidad + 1
                )
        return nodo

    # ------------------------------------------------------------------
    # Medidas de informacion (entropia y ganancia)
    # ------------------------------------------------------------------

    def _entropia(self, y):
        """Entropia de Shannon de un conjunto de ejemplos.

        Entropy(S) = - sum_c P(c) * log2(P(c))
        Mide la "mezcla" de clases: 0 si todas son iguales, maximo si estan
        todas igual de distribuidas.
        """
        total = len(y)
        entropia = 0.0
        for clase in self.clases_:
            proporcion = np.sum(y == clase) / total
            if proporcion > 0:
                entropia -= proporcion * np.log2(proporcion)
        return entropia

    def _ganancia(self, X, y, atributo, entropia_de_s):
        """Ganancia de informacion de un atributo A sobre S.

        Gain(S, A) = Entropy(S) - sum_v (|S_v| / |S|) * Entropy(S_v)
        Bits que se "ahorran" al conocer el valor del atributo. Cuanto mayor,
        mejor separa el atributo las clases.
        """
        total = len(y)
        entropia_condicional = 0.0
        # Igual que en _construir: np.unique nos da los valores distintos
        # de la columna, y por cada valor medimos la entropia de esa parte.
        for valor in np.unique(X[:, atributo]):
            ejemplos_en_rama = X[:, atributo] == valor
            proporcion = np.sum(ejemplos_en_rama) / total
            entropia_condicional += proporcion * self._entropia(y[ejemplos_en_rama])
        return entropia_de_s - entropia_condicional

    # ------------------------------------------------------------------
    # Clasificacion
    # ------------------------------------------------------------------

    def predict(self, X):
        """Clasifica cada fila recorriendo el arbol desde la raiz hasta una hoja.

        Cada fila baja por las ramas que indican sus valores de atributo.
        Ojo: la etiqueta real NO se usa aca. Despues, en la evaluacion, se
        comparan estas predicciones contra el winner real (test) para
        calcular accuracy / F1 / etc.
        """
        X = np.asarray(X)
        return np.array([self._clasificar(fila) for fila in X])

    def _clasificar(self, fila):
        nodo = self.raiz_
        while nodo.atributo is not None:
            valor = int(fila[nodo.atributo])
            hijo = nodo.ramas.get(valor)
            if hijo is None:
                # Valor que no aparecio en train: no hay rama por donde
                # bajar. Quedamos con el respaldo del nodo (nodo.clase,
                # el mayoritario "por las dudas"). Misma idea que la
                # regla 3 del pseudocodigo del curso.
                break
            nodo = hijo
        return nodo.clase

    # ------------------------------------------------------------------
    # Informacion sobre el arbol entrenado
    # ------------------------------------------------------------------

    def get_depth(self):
        """Profundidad maxima: cantidad de niveles hasta la hoja mas lejana."""
        return self._profundidad(self.raiz_)

    def _profundidad(self, nodo):
        if not nodo.ramas:
            return 0
        return 1 + max(self._profundidad(hijo) for hijo in nodo.ramas.values())

    def get_n_leaves(self):
        """Cantidad de hojas (clasificaciones posibles) del arbol."""
        return self._cantidad_de_hojas(self.raiz_)

    def _cantidad_de_hojas(self, nodo):
        if not nodo.ramas:
            return 1
        return sum(self._cantidad_de_hojas(hijo) for hijo in nodo.ramas.values())