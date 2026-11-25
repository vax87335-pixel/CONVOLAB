
import ast
import math
import warnings
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad, IntegrationWarning
from matplotlib.ticker import FixedLocator, FuncFormatter

# ==============================================
# CONVOLAB WEB v0.7.1
# VERSION FINAL CORREGIDA
# ==============================================

st.set_page_config(
    page_title="CONVOLAB",
    page_icon="📈",
    layout="wide"
)

t = sp.Symbol("t", real=True)
u = sp.Symbol("tau", real=True)

PERMITIDAS = {
    "t": t,
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "oo": sp.oo,
    "inf": sp.oo,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "Abs": sp.Abs,
    "sign": sp.sign,
}

st.title("📈 CONVOLAB")
st.subheader("Laboratorio interactivo de señales")

st.caption(
    "Tiempo continuo · Expresiones analíticas · "
    "Operaciones · Manipulación · Convolución · "
    "Derivación · Integración"
)


# ==============================================
# INTERPRETACION DE EXPRESIONES
# ==============================================

def interpretar(texto):

    texto = (
        texto.strip()
        .replace("^", "**")
        .replace("∞", "oo")
        .replace("−", "-")
    )

    if not texto or len(texto) > 500:
        raise ValueError(
            "Escribe una expresión de hasta 500 caracteres."
        )

    arbol = ast.parse(texto, mode="eval")

    if sum(1 for _ in ast.walk(arbol)) > 120:
        raise ValueError(
            "La expresión es demasiado extensa."
        )

    def convertir(nodo):

        if (
            isinstance(nodo, ast.Constant)
            and type(nodo.value) in (int, float)
        ):
            return sp.Rational(str(nodo.value))

        if (
            isinstance(nodo, ast.Name)
            and nodo.id in PERMITIDAS
        ):
            valor = PERMITIDAS[nodo.id]

            if not callable(valor):
                return valor

        if isinstance(nodo, ast.UnaryOp):

            valor = convertir(nodo.operand)

            if isinstance(nodo.op, ast.USub):
                return -valor

            if isinstance(nodo.op, ast.UAdd):
                return valor

        if isinstance(nodo, ast.BinOp):

            izquierda = convertir(nodo.left)
            derecha = convertir(nodo.right)

            if isinstance(nodo.op, ast.Add):
                return izquierda + derecha

            if isinstance(nodo.op, ast.Sub):
                return izquierda - derecha

            if isinstance(nodo.op, ast.Mult):
                return izquierda * derecha

            if isinstance(nodo.op, ast.Div):
                return izquierda / derecha

            if isinstance(nodo.op, ast.Pow):

                if (
                    derecha.is_number
                    and abs(float(derecha)) > 100
                ):
                    raise ValueError(
                        "Utiliza exponentes entre -100 y 100."
                    )

                return izquierda ** derecha

        if (
            isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Name)
            and nodo.func.id in PERMITIDAS
            and not nodo.keywords
            and len(nodo.args) == 1
        ):

            funcion = PERMITIDAS[nodo.func.id]

            if callable(funcion):
                return funcion(
                    convertir(nodo.args[0])
                )

        raise ValueError(
            "Utiliza las expresiones indicadas "
            "en la guía rápida."
        )

    expr = convertir(arbol.body)

    if (
        expr.has(sp.nan, sp.zoo)
        or expr.is_real is False
    ):
        raise ValueError(
            "La expresión debe ser real y estar definida."
        )

    return expr


def interpretar_limite(texto):

    valor = interpretar(texto)

    if valor.free_symbols:
        raise ValueError(
            "El límite debe ser constante."
        )

    if valor.is_extended_real is not True:
        raise ValueError(
            "El límite debe ser real."
        )

    return valor


# ==============================================
# EDITOR DE SEÑALES
# ==============================================

def editor(nombre):

    st.markdown(f"### Señal {nombre}(t)")

    cantidad = st.number_input(
        "Número de tramos",
        min_value=0,
        max_value=8,
        value=1,
        key=f"cantidad_{nombre}"
    )

    tramos = []

    for i in range(cantidad):

        st.markdown(f"**Tramo {i+1}**")

        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:

            expresion = st.text_input(
                "Expresión",
                value="1",
                key=f"{nombre}_expr_{i}"
            )

        with c2:

            inicio = st.text_input(
                "Desde",
                value=str(
                    i * (2 if nombre == "x" else 1)
                ),
                key=f"{nombre}_ini_{i}",
                help="Para infinito negativo escribe -inf."
            )

        with c3:

            final = st.text_input(
                "Hasta",
                value=str(
                    (i + 1) *
                    (2 if nombre == "x" else 1)
                ),
                key=f"{nombre}_fin_{i}",
                help="Para infinito positivo escribe inf."
            )

        try:

            expr = interpretar(expresion)

            if expr.has(sp.oo, -sp.oo):

                raise ValueError(
                    "El infinito solamente se admite "
                    "en los límites."
                )

            a = interpretar_limite(inicio)
            b = interpretar_limite(final)

            if a >= b:
                raise ValueError(
                    "El límite inferior debe ser "
                    "menor que el superior."
                )

            tramos.append((expr, a, b))

        except Exception as error:

            st.error(f"Tramo {i+1}: {error}")
            return None

    tramos.sort(
        key=lambda r: float(r[1])
    )

    for anterior, actual in zip(
        tramos, tramos[1:]
    ):

        if anterior[2] > actual[1]:

            st.error(
                "Los tramos no deben superponerse."
            )

            return None

    return tramos


# ==============================================
# FORMATO MATEMATICO
# ==============================================

def formato_limite(valor):

    if valor == sp.oo:
        return r"+\infty"

    if valor == -sp.oo:
        return r"-\infty"

    return sp.latex(valor)


def condicion_intervalo(
    a,
    b,
    invertido=False
):

    if a == -sp.oo and b == sp.oo:

        return r"t\in\mathbb{R}"

    if a == -sp.oo:

        signo = (
            r"\leq"
            if invertido
            else "<"
        )

        return (
            "t"
            + signo
            + formato_limite(b)
        )

    if b == sp.oo:

        signo = (
            ">"
            if invertido
            else r"\geq"
        )

        return (
            "t"
            + signo
            + formato_limite(a)
        )

    if invertido:

        return (
            formato_limite(a)
            + r"<t\leq "
            + formato_limite(b)
        )

    return (
        formato_limite(a)
        + r"\leq t<"
        + formato_limite(b)
    )


def latex_tramos(
    nombre,
    tramos,
    invertido=False
):

    filas = []

    for expr, a, b in tramos:

        filas.append(
            sp.latex(expr)
            + " & "
            + condicion_intervalo(
                a, b, invertido
            )
        )

    if not (
        len(tramos) == 1
        and tramos[0][1] == -sp.oo
        and tramos[0][2] == sp.oo
    ):

        filas.append(
            r"0 & \text{otro caso}"
        )

    cuerpo = r" \\ ".join(filas)

    return (
        nombre
        + r"(t)=\begin{cases}"
        + cuerpo
        + r"\end{cases}"
    )


# ==============================================
# EVALUACION DE SEÑALES
# ==============================================

def evaluar(
    tramos,
    puntos,
    invertido=False
):

    puntos = np.asarray(
        puntos,
        dtype=float
    )

    salida = np.zeros_like(puntos)

    for expr, a, b in tramos:

        if invertido:

            mask = (
                (puntos > float(a))
                & (puntos <= float(b))
            )

        else:

            mask = (
                (puntos >= float(a))
                & (puntos < float(b))
            )

        if not np.any(mask):
            continue

        funcion = sp.lambdify(
            t, expr, "numpy"
        )

        with np.errstate(all="ignore"):

            valores = funcion(
                puntos[mask]
            )

        salida[mask] = np.broadcast_to(
            np.asarray(
                valores,
                dtype=float
            ),
            salida[mask].shape
        )

    return salida


def fronteras(tramos):

    puntos = set()

    for expr, a, b in tramos:

        if a.is_finite:
            puntos.add(float(a))

        if b.is_finite:
            puntos.add(float(b))

    return sorted(puntos)


# ==============================================
# CONFIGURACION DE EJES
# ==============================================

def configurar_eje_tiempo(
    ax,
    cortes,
    minimo,
    maximo
):

    if minimo >= maximo:

        raise ValueError(
            "Selecciona un intervalo válido."
        )

    ax.set_xlim(minimo, maximo)

    ticks = list(
        range(
            math.ceil(minimo),
            math.floor(maximo) + 1
        )
    )

    if not ticks:
        ticks = [minimo, maximo]

    ax.xaxis.set_major_locator(
        FixedLocator(ticks)
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda valor, posicion: f"{valor:g}"
        )
    )

    ax.tick_params(
        axis="x",
        labelsize=9,
        rotation=0,
        pad=5
    )

    ax.minorticks_off()


def preparar_figura(fig):

    fig.tight_layout()
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()

    for ax in fig.axes:

        etiquetas = ax.get_xticklabels()

        cajas = [
            etiqueta.get_window_extent(renderer)
            for etiqueta in etiquetas
        ]

        if any(
            a.x1 + 5 > b.x0
            for a, b in zip(
                cajas, cajas[1:]
            )
        ):

            for etiqueta in etiquetas:

                etiqueta.set_rotation(90)

                etiqueta.set_horizontalalignment(
                    "center"
                )

        ax.xaxis.labelpad = 8

    fig.tight_layout()


# ==============================================
# GRAFICAS POR INTERVALOS
# ==============================================

def dibujar(
    ax,
    funcion,
    cortes,
    minimo,
    maximo,
    color,
    titulo,
    unir_saltos=True
):

    divisiones = sorted(
        set(
            [minimo, maximo]
            + [
                x for x in cortes
                if minimo < x < maximo
            ]
        )
    )

    for a, b in zip(
        divisiones[:-1],
        divisiones[1:]
    ):

        if a >= b:
            continue

        eps = (b - a) * 1e-8

        xs = np.linspace(
            a + eps,
            b - eps,
            350
        )

        ys = np.asarray(
            funcion(xs),
            dtype=float
        )

        ys = np.broadcast_to(
            ys, xs.shape
        )

        ax.plot(
            xs,
            np.ma.masked_invalid(ys),
            color=color,
            lw=2
        )

    for x in (
        cortes if unir_saltos else []
    ):

        if not minimo < x < maximo:
            continue

        eps = min(
            1e-7,
            (maximo - minimo) * 1e-9
        )

        izquierda = float(
            funcion(
                np.array([x - eps])
            )[0]
        )

        derecha = float(
            funcion(
                np.array([x + eps])
            )[0]
        )

        if (
            np.isfinite(izquierda)
            and np.isfinite(derecha)
            and not np.isclose(
                izquierda,
                derecha,
                rtol=1e-6,
                atol=1e-7
            )
        ):

            ax.plot(
                [x, x],
                [izquierda, derecha],
                color=color,
                lw=1.7
            )

    ax.axhline(
        0,
        color="gray",
        lw=0.7
    )

    ax.axvline(
        0,
        color="gray",
        lw=0.7
    )

    ax.set_title(titulo)
    ax.set_xlabel("Tiempo t")
    ax.set_ylabel("Amplitud")

    configurar_eje_tiempo(
        ax,
        cortes,
        minimo,
        maximo
    )

    ax.grid(alpha=0.25)


# ==============================================
# OPERACIONES ANALITICAS POR TRAMOS
# ==============================================

def expresion_en(tramos, muestra):

    for expr, a, b in tramos:

        if float(a) <= muestra < float(b):
            return expr

    return sp.S.Zero


def punto_muestra(a, b):

    if a == -sp.oo and b == sp.oo:
        return 0.0

    if a == -sp.oo:
        return float(b) - 1

    if b == sp.oo:
        return float(a) + 1

    return (
        float(a) + float(b)
    ) / 2


def combinar_iguales(tramos):

    if not tramos:
        return []

    resultado = [tramos[0]]

    for expr, a, b in tramos[1:]:

        anterior, inicio, fin = resultado[-1]

        if (
            fin == a
            and sp.simplify(expr - anterior) == 0
        ):

            resultado[-1] = (
                anterior,
                inicio,
                b
            )

        else:

            resultado.append(
                (expr, a, b)
            )

    return resultado


def operacion_analitica(
    x,
    k,
    operacion
):

    limites = {
        -sp.oo,
        sp.oo
    }

    for expr, a, b in x + k:

        limites.add(a)
        limites.add(b)

    limites = sorted(
        limites,
        key=float
    )

    resultado = []

    for a, b in zip(
        limites[:-1],
        limites[1:]
    ):

        muestra = punto_muestra(a, b)

        fx = expresion_en(x, muestra)
        fk = expresion_en(k, muestra)

        if operacion == "Suma":

            expr = fx + fk

        elif operacion == "Resta":

            expr = fx - fk

        else:

            expr = fx * fk

        expr = sp.simplify(expr)

        resultado.append(
            (expr, a, b)
        )

    return [
        tramo
        for tramo in combinar_iguales(resultado)
        if tramo[0] != 0
    ]


def sumar_tramos(lista):

    resultado = []

    for tramos in lista:

        resultado = operacion_analitica(
            resultado,
            tramos,
            "Suma"
        )

    return resultado


# ==============================================
# VALIDACION MATEMATICA
# ==============================================

def valor_finito(
    expr,
    punto,
    direccion
):

    valor = sp.simplify(
        sp.limit(
            expr,
            t,
            punto,
            dir=direccion
        )
    )

    if (
        valor.is_finite is not True
        or valor.is_real is not True
    ):

        raise ValueError(
            f"No existe un límite lateral "
            f"real y finito en t = {punto}."
        )

    return valor


def validar_intervalo(
    expr,
    a,
    b
):

    from sympy.calculus.util import continuous_domain

    try:

        dominio = continuous_domain(
            expr,
            t,
            sp.S.Reals
        )

        valido = sp.Interval.open(
            a, b
        ).is_subset(dominio)

    except (
        NotImplementedError,
        ValueError
    ):

        raise ValueError(
            "No fue posible verificar el dominio. "
            "Divide la señal en tramos continuos."
        )

    if valido is not True:

        raise ValueError(
            "La expresión contiene una discontinuidad "
            "interna, singularidad o zona no real."
        )


# ==============================================
# DERIVACION
# ==============================================

def derivar_senal(tramos):

    regulares = []

    for expr, a, b in tramos:

        validar_intervalo(
            expr, a, b
        )

        derivada = sp.simplify(
            sp.diff(expr, t)
        )

        if derivada.has(
            sp.Derivative,
            sp.DiracDelta
        ):

            raise ValueError(
                "Divide la expresión en tramos "
                "suaves antes de derivarla."
            )

        if derivada != 0:

            regulares.append(
                (derivada, a, b)
            )

    limites = sorted(
        {
            v
            for _, a, b in tramos
            for v in (a, b)
            if v.is_finite
        },
        key=float
    )

    impulsos = []

    for punto in limites:

        izquierda = sp.S.Zero
        derecha = sp.S.Zero

        for expr, a, b in tramos:

            if a < punto <= b:

                izquierda = valor_finito(
                    expr, punto, "-"
                )

            if a <= punto < b:

                derecha = valor_finito(
                    expr, punto, "+"
                )

        salto = sp.simplify(
            derecha - izquierda
        )

        if salto != 0:

            impulsos.append(
                (punto, salto)
            )

    return (
        combinar_iguales(regulares),
        impulsos
    )


# ==============================================
# INTEGRACION ACUMULADA
# ==============================================

def integrar_senal(tramos):

    limites = {
        -sp.oo,
        sp.oo,
        sp.S.Zero
    }

    for expr, a, b in tramos:

        validar_intervalo(expr, a, b)

        limites.update((a, b))

    limites = sorted(
        limites,
        key=float
    )

    celdas = []

    for a, b in zip(
        limites[:-1],
        limites[1:]
    ):

        expr = expresion_en(
            tramos,
            punto_muestra(a, b)
        )

        primitiva = sp.integrate(
            expr, t
        )

        if primitiva.has(sp.Integral):

            raise ValueError(
                "No se obtuvo una primitiva "
                "cerrada para todos los tramos."
            )

        primitiva = sp.simplify(
            sp.re(primitiva)
        )

        celdas.append(
            (a, b, primitiva)
        )

    resultados = []

    acumulado = sp.S.Zero

    for a, b, primitiva in celdas:

        if a < 0:
            continue

        referencia = valor_finito(
            primitiva, a, "+"
        )

        expr = sp.simplify(
            acumulado
            + primitiva
            - referencia
        )

        resultados.append(
            (expr, a, b)
        )

        if b != sp.oo:

            acumulado = valor_finito(
                expr, b, "-"
            )

    acumulado = sp.S.Zero

    for a, b, primitiva in reversed(celdas):

        if b > 0:
            continue

        referencia = valor_finito(
            primitiva, b, "-"
        )

        expr = sp.simplify(
            acumulado
            + primitiva
            - referencia
        )

        resultados.append(
            (expr, a, b)
        )

        if a != -sp.oo:

            acumulado = valor_finito(
                expr, a, "+"
            )

    resultados.sort(
        key=lambda tramo: float(tramo[1])
    )

    return [
        tramo
        for tramo in combinar_iguales(resultados)
        if tramo[0] != 0
    ]


# ==============================================
# IMPULSOS DE DIRAC
# ==============================================

def normalizar_impulsos(impulsos):

    pesos = {}

    for p, w in impulsos:

        p = sp.simplify(p)

        pesos[p] = sp.simplify(
            pesos.get(p, 0) + w
        )

    return sorted(
        [
            (p, w)
            for p, w in pesos.items()
            if w != 0
        ],
        key=lambda z: float(z[0])
    )


def editor_mixto(nombre):

    tramos = editor(nombre)

    if tramos is None:
        return None

    with st.expander(
        f"Impulsos de Dirac de {nombre}(t) (opcional)"
    ):

        st.latex(r"A\delta(t-a)")

        st.caption(
            "a: posición del impulso. "
            "A: peso del impulso. "
            "Para ingresar solamente impulsos, "
            "selecciona 0 tramos."
        )

        n = st.number_input(
            "Número de impulsos",
            min_value=0,
            max_value=8,
            value=0,
            key=f"ni_{nombre}"
        )

        impulsos = []

        for j in range(n):

            st.markdown(
                f"**Impulso {j+1}**"
            )

            c1, c2 = st.columns(2)

            try:

                p = interpretar_limite(
                    c1.text_input(
                        "Posición",
                        str(j),
                        key=f"ip_{nombre}_{j}"
                    )
                )

                w = interpretar_limite(
                    c2.text_input(
                        "Peso",
                        "1",
                        key=f"iw_{nombre}_{j}"
                    )
                )

                if (
                    not p.is_finite
                    or not w.is_finite
                ):

                    raise ValueError(
                        "La posición y el peso "
                        "deben ser finitos."
                    )

                impulsos.append((p, w))

            except Exception as error:

                st.error(str(error))
                return None

    return (
        tramos,
        normalizar_impulsos(impulsos)
    )


# ==============================================
# FORMULAS DE SEÑALES MIXTAS
# ==============================================

def mostrar_formula(
    nombre,
    tramos,
    impulsos=(),
    invertido=False
):

    if tramos:

        nombre_regular = (
            nombre + r"_{\mathrm{reg}}"
            if impulsos
            else nombre
        )

        st.latex(
            latex_tramos(
                nombre_regular,
                tramos,
                invertido
            )
        )

    if impulsos:

        distribucion = sum(
            (
                w * sp.DiracDelta(t - p)
                for p, w in impulsos
            ),
            sp.S.Zero
        )

        delta_latex = sp.latex(
            distribucion
        )

        if tramos:

            signo = (
                ""
                if delta_latex.startswith("-")
                else "+"
            )

            st.latex(
                nombre
                + "(t)="
                + nombre
                + r"_{\mathrm{reg}}(t)"
                + signo
                + delta_latex
            )

        else:

            st.latex(
                nombre
                + "(t)="
                + delta_latex
            )

    elif not tramos:

        st.latex(
            nombre + "(t)=0"
        )


# ==============================================
# REPRESENTACION GRAFICA DE IMPULSOS
# ==============================================

def dibujar_impulsos(
    ax,
    impulsos,
    minimo,
    maximo,
    color="crimson",
    leyenda=True
):

    visibles = [
        (p, w)
        for p, w in impulsos
        if minimo <= float(p) <= maximo
    ]

    if not visibles:
        return

    lo, hi = ax.get_ylim()

    escala = max(
        abs(float(w))
        for _, w in visibles
    )

    altura = max(
        abs(lo),
        abs(hi),
        1
    ) * 0.85

    extremos = [0]

    for p, w in visibles:

        x = float(p)

        y = (
            altura * float(w) / escala
        )

        extremos.append(y)

        ax.annotate(
            "",
            xy=(x, y),
            xytext=(x, 0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=2.2
            ),
            zorder=6
        )

        ax.annotate(
            r"$"
            + sp.latex(w)
            + r"\,\delta$",
            (x, y),
            xytext=(
                5,
                7 if y > 0 else -16
            ),
            textcoords="offset points",
            color=color,
            zorder=7
        )

    ax.set_ylim(
        min(
            lo,
            min(extremos) - altura * 0.30
        ),
        max(
            hi,
            max(extremos) + altura * 0.30
        )
    )

    if leyenda:

        ax.plot(
            [],
            [],
            color=color,
            label="Impulsos: peso indicado"
        )

        ax.legend(
            loc="best",
            fontsize=8
        )


# ==============================================
# EXTREMOS ABIERTOS Y CERRADOS
# ==============================================

def dibujar_extremos(
    ax,
    tramos,
    minimo,
    maximo,
    color,
    invertido=False
):

    puntos = sorted(
        {
            p
            for _, a, b in tramos
            for p in (a, b)
            if p.is_finite
            and minimo <= float(p) <= maximo
        },
        key=float
    )

    for p in puntos:

        try:

            izquierda = sp.S.Zero
            derecha = sp.S.Zero

            for expr, a, b in tramos:

                if a < p <= b:

                    izquierda = valor_finito(
                        expr, p, "-"
                    )

                if a <= p < b:

                    derecha = valor_finito(
                        expr, p, "+"
                    )

            if sp.simplify(
                izquierda - derecha
            ) == 0:

                continue

            if invertido:

                cerrado = izquierda
                abierto = derecha

            else:

                cerrado = derecha
                abierto = izquierda

            ax.scatter(
                [float(p)],
                [float(abierto)],
                s=28,
                facecolors=ax.get_facecolor(),
                edgecolors=color,
                linewidths=1.4,
                zorder=8
            )

            ax.scatter(
                [float(p)],
                [float(cerrado)],
                s=28,
                color=color,
                zorder=9
            )

        except Exception:

            continue


# ==============================================
# GRAFICAS MIXTAS
# ==============================================

def graficar_mixta(
    tramos,
    impulsos,
    minimo,
    maximo,
    color,
    titulo,
    saltos=True,
    invertido=False,
    extremos=False
):

    fig, ax = plt.subplots(
        figsize=(11, 4)
    )

    try:

        dibujar(
            ax,
            lambda z: evaluar(
                tramos,
                z,
                invertido
            ),
            fronteras(tramos),
            minimo,
            maximo,
            color,
            titulo,
            unir_saltos=saltos
        )

        if extremos:

            dibujar_extremos(
                ax,
                tramos,
                minimo,
                maximo,
                color,
                invertido
            )

        dibujar_impulsos(
            ax,
            impulsos,
            minimo,
            maximo
        )

        preparar_figura(fig)

        st.pyplot(fig)

    finally:

        plt.close(fig)


# ==============================================
# TRANSFORMACION INTERNA
# g(t) = A*f(B*t+D)
# ==============================================

def transformar(
    tramos,
    impulsos,
    A,
    B,
    D
):

    if B == 0:

        raise ValueError(
            "El factor B debe ser distinto de cero."
        )

    if A == 0:

        return [], []

    salida = []

    for e, a, b in tramos:

        l = (a - D) / B
        r = (b - D) / B

        if B < 0:

            l, r = r, l

        expresion = sp.simplify(
            A * e.subs(
                t,
                B * t + D
            )
        )

        salida.append(
            (expresion, l, r)
        )

    salida.sort(
        key=lambda z: float(z[1])
    )

    nuevos_impulsos = [
        (
            (p - D) / B,
            sp.simplify(
                A * w / abs(B)
            )
        )
        for p, w in impulsos
    ]

    return (
        salida,
        normalizar_impulsos(
            nuevos_impulsos
        )
    )


# ==============================================
# MANIPULACION FACTORIZADA
# y(t) = A*f[B(t-C)]
# ==============================================

def manipulacion_factorizada(
    tramos,
    impulsos,
    A,
    B,
    C
):

    D = -B * C

    return transformar(
        tramos,
        impulsos,
        A,
        B,
        D
    )


def latex_manipulacion(
    nombre,
    A,
    B,
    C
):

    if A == 0:

        return r"y(t)=0"

    amplitud = (
        ""
        if A == 1
        else "-"
        if A == -1
        else sp.latex(A)
    )

    interior = sp.latex(t - C)

    if B == 1:

        argumento = interior

    elif B == -1:

        argumento = (
            r"-\left("
            + interior
            + r"\right)"
        )

    else:

        argumento = (
            sp.latex(B)
            + r"\left("
            + interior
            + r"\right)"
        )

    return (
        "y(t)="
        + amplitud
        + nombre
        + r"\left["
        + argumento
        + r"\right]"
    )


# ==============================================
# OPERACIONES CON IMPULSOS
# ==============================================

def valor_continuo(tramos, p):

    izquierda = sp.S.Zero
    derecha = sp.S.Zero

    for e, a, b in tramos:

        if a < p <= b:

            izquierda = valor_finito(
                e, p, "-"
            )

        if a <= p < b:

            derecha = valor_finito(
                e, p, "+"
            )

    if sp.simplify(
        izquierda - derecha
    ) != 0:

        raise ValueError(
            f"Producto con impulso en una "
            f"discontinuidad (t={p}). "
            "No está definido sin una "
            "convención adicional."
        )

    return derecha


def operacion_mixta(
    x,
    k,
    operacion
):

    xr, xi = x
    kr, ki = k

    reg = operacion_analitica(
        xr,
        kr,
        operacion
    )

    if operacion == "Suma":

        imp = xi + ki

    elif operacion == "Resta":

        imp = xi + [
            (p, -w)
            for p, w in ki
        ]

    else:

        if (
            {p for p, w in xi}
            & {p for p, w in ki}
        ):

            raise ValueError(
                "El producto de impulsos "
                "en la misma posición "
                "no está definido."
            )

        imp = [
            (
                p,
                w * valor_continuo(kr, p)
            )
            for p, w in xi
        ]

        imp += [
            (
                p,
                w * valor_continuo(xr, p)
            )
            for p, w in ki
        ]

    return (
        reg,
        normalizar_impulsos(imp)
    )


def integrar_mixta(
    tramos,
    impulsos
):

    reg = integrar_senal(tramos)

    extras = []

    for p, w in impulsos:

        if p > 0:

            extras.append([
                (w, p, sp.oo)
            ])

        else:

            extras.append([
                (-w, -sp.oo, p)
            ])

    return (
        sumar_tramos([reg] + extras),
        []
    )


# ==============================================
# CONVOLUCION NUMERICA
# ==============================================

def preparar_pares(x, k):

    pares = []

    for fx, a, b in x:

        for fk, c, d in k:

            expresion = (
                fx.subs(t, u)
                * fk.subs(t, t - u)
            )

            funcion = sp.lambdify(
                (u, t),
                expresion,
                "numpy"
            )

            pares.append(
                (
                    funcion,
                    float(a),
                    float(b),
                    float(c),
                    float(d)
                )
            )

    return pares


def convolucion_instante(
    pares,
    tiempo
):

    total = 0.0

    for f, a, b, c, d in pares:

        inferior = max(
            a,
            tiempo - d
        )

        superior = min(
            b,
            tiempo - c
        )

        if inferior >= superior:
            continue

        integral, error = quad(
            lambda z: float(
                f(z, tiempo)
            ),
            inferior,
            superior,
            epsabs=1e-7,
            epsrel=1e-6,
            limit=150
        )

        total += integral

    if not np.isfinite(total):

        raise ValueError(
            "La integral no produjo un valor finito."
        )

    return total


def convolucion_numerica(
    x,
    k,
    minimo,
    maximo,
    puntos_extra=()
):

    tiempos = np.unique(
        np.r_[
            np.linspace(
                minimo,
                maximo,
                151
            ),
            [
                float(v)
                for v in puntos_extra
                if minimo <= float(v) <= maximo
            ]
        ]
    )

    pares = preparar_pares(x, k)

    valores = []

    with warnings.catch_warnings():

        warnings.simplefilter(
            "error",
            IntegrationWarning
        )

        for tiempo in tiempos:

            valores.append(
                convolucion_instante(
                    pares,
                    tiempo
                )
            )

    return (
        tiempos,
        np.asarray(valores)
    )


# ==============================================
# PLAN DE CONVOLUCION ANALITICA
# ==============================================

def plan_convolucion(x, k):

    cortes = {
        -sp.oo,
        sp.oo
    }

    for _, a, b in x:

        for _, c, d in k:

            cortes.update(
                v + z
                for v in (a, b)
                for z in (c, d)
                if v.is_finite
                and z.is_finite
            )

    cortes = sorted(
        cortes,
        key=float
    )

    if len(cortes) > 34:

        raise ValueError(
            "Demasiados intervalos para "
            "el desarrollo simbólico."
        )

    plan = []

    for a, b in zip(
        cortes[:-1],
        cortes[1:]
    ):

        muestra = punto_muestra(a, b)

        partes = []

        for i, (fx, l, r) in enumerate(
            x, 1
        ):

            for j, (fk, c, d) in enumerate(
                k, 1
            ):

                inf = max(
                    (l, t - d),
                    key=lambda z: float(
                        z.subs(t, muestra)
                    )
                )

                sup = min(
                    (r, t - c),
                    key=lambda z: float(
                        z.subs(t, muestra)
                    )
                )

                if (
                    float(inf.subs(t, muestra))
                    >= float(sup.subs(t, muestra))
                ):

                    continue

                producto = sp.simplify(
                    fx.subs(t, u)
                    * fk.subs(t, t - u)
                )

                if producto != 0:

                    partes.append(
                        (
                            i,
                            j,
                            inf,
                            sup,
                            producto
                        )
                    )

        plan.append(
            (a, b, partes)
        )

    return plan


# ==============================================
# RESOLUCION DE INTEGRALES
# ==============================================

def resolver_plan(plan):

    tramos = []
    desarrollo = []

    completo = True

    for a, b, partes in plan:

        total = sp.S.Zero
        pasos = []

        for i, j, l, r, f in partes:

            F = None
            valor = None

            try:

                F = sp.integrate(f, u)

                if F.has(sp.Integral):

                    F = None

                valor = sp.integrate(
                    f,
                    (u, l, r)
                )

                if (
                    valor.has(
                        sp.Integral,
                        sp.Piecewise,
                        sp.oo,
                        -sp.oo,
                        sp.zoo,
                        sp.nan
                    )
                    or valor.is_finite is False
                    or valor.is_real is False
                ):

                    valor = None

            except Exception:

                valor = None

            pasos.append(
                (
                    i,
                    j,
                    l,
                    r,
                    f,
                    F,
                    valor
                )
            )

            if valor is None:

                completo = False

            else:

                total += valor

        total = sp.expand(
            sp.simplify(total)
        )

        desarrollo.append(
            (
                a,
                b,
                pasos,
                total
                if all(
                    p[-1] is not None
                    for p in pasos
                )
                else None
            )
        )

        if total != 0:

            tramos.append(
                (total, a, b)
            )

    return (
        combinar_iguales(tramos)
        if completo
        else None,
        desarrollo
    )


# ==============================================
# CONVOLUCION CON IMPULSOS
# ==============================================

def contribuciones_impulsos(x, k):

    xr, xi = x
    kr, ki = k

    extras = [
        transformar(
            kr,
            [],
            w,
            sp.S.One,
            -p
        )[0]
        for p, w in xi
    ]

    extras += [
        transformar(
            xr,
            [],
            w,
            sp.S.One,
            -p
        )[0]
        for p, w in ki
    ]

    imp = normalizar_impulsos([
        (
            p + q,
            w * v
        )
        for p, w in xi
        for q, v in ki
    ])

    return (
        sumar_tramos(extras),
        imp
    )


# ==============================================
# DESARROLLO MATEMATICO PASO A PASO
# ==============================================

def mostrar_desarrollo(
    plan,
    desarrollo,
    fijo,
    movil,
    x,
    k
):

    with st.expander(
        "Desarrollo matemático paso a paso",
        expanded=False
    ):

        st.markdown(
            "**1. Identificación de las señales**"
        )

        st.write(
            f"Señal fija: {fijo}(t)"
        )

        st.write(
            f"Señal móvil: {movil}(t)"
        )

        st.latex(
            rf"c(t)=\int_{{-\infty}}^{{\infty}}"
            rf"{fijo}(\tau)"
            rf"{movil}(t-\tau)"
            r"\,d\tau"
        )

        st.markdown(
            "**2. Inversión de la señal móvil**"
        )

        for i, (e, a, b) in enumerate(
            k[0], 1
        ):

            st.write(f"Tramo {i}")

            st.latex(
                rf"{movil}(-\tau)="
                + sp.latex(
                    e.subs(t, -u)
                )
                + r",\quad "
                + sp.latex(-b)
                + r"<\tau\leq "
                + sp.latex(-a)
            )

        for p, w in k[1]:

            st.latex(
                sp.latex(
                    w * sp.DiracDelta(-u - p)
                )
            )

        st.markdown(
            "**3. Desplazamiento de la señal móvil**"
        )

        for i, (e, a, b) in enumerate(
            k[0], 1
        ):

            st.write(f"Tramo {i}")

            st.latex(
                rf"{movil}(t-\tau)="
                + sp.latex(
                    e.subs(t, t - u)
                )
                + r",\quad "
                + sp.latex(t - b)
                + r"<\tau\leq "
                + sp.latex(t - a)
            )

        for p, w in k[1]:

            st.latex(
                sp.latex(
                    w * sp.DiracDelta(t - u - p)
                )
            )

        st.markdown(
            "**4. Puntos de cambio de la convolución**"
        )

        cortes = sorted(
            {
                v
                for a, b, _ in plan
                for v in (a, b)
                if v.is_finite
            },
            key=float
        )

        if cortes:

            st.latex(
                r"t\in\left\{"
                + ", ".join(
                    map(sp.latex, cortes)
                )
                + r"\right\}"
            )

        else:

            st.write(
                "No existen puntos de cambio "
                "finitos en las partes regulares."
            )

        st.markdown(
            "**5. Límites de integración**"
        )

        st.latex(
            r"L=\max(a,t-d)"
        )

        st.latex(
            r"U=\min(b,t-c)"
        )

        st.caption(
            "Se integra cuando L < U. "
            "Los extremos aislados no cambian "
            "las integrales de las partes regulares."
        )

        filas = []

        for a, b, pasos, total in desarrollo:

            if not pasos:

                filas.append({
                    "Intervalo": f"{a} a {b}",
                    "Tramos": "—",
                    "Límite inferior": "—",
                    "Límite superior": "—",
                    "Resultado regular": "0"
                })

            for i, j, l, r, f, F, v in pasos:

                filas.append({
                    "Intervalo": f"{a} a {b}",
                    "Tramos": f"{i} × {j}",
                    "Límite inferior": str(l),
                    "Límite superior": str(r),
                    "Resultado regular": (
                        str(v)
                        if v is not None
                        else "No resuelto"
                    )
                })

        if filas:

            st.table(filas)

        st.markdown(
            "**6. Desarrollo de las integrales**"
        )

        for n, (a, b, pasos, total) in enumerate(
            desarrollo,
            1
        ):

            st.divider()

            st.markdown(
                f"#### Intervalo {n}"
            )

            st.latex(
                condicion_intervalo(a, b)
            )

            if not pasos:

                st.write(
                    "No existe superposición "
                    "entre las partes regulares."
                )

                st.latex(
                    r"c_{\mathrm{reg}}(t)=0"
                )

                continue

            for i, j, l, r, f, F, v in pasos:

                st.markdown(
                    f"Tramo fijo {i} × tramo móvil {j}"
                )

                st.latex(
                    r"\int_{"
                    + sp.latex(l)
                    + "}^{"
                    + sp.latex(r)
                    + "}"
                    + sp.latex(f)
                    + r"\,d\tau"
                )

                if F is not None:

                    st.latex(
                        r"\left["
                        + sp.latex(F)
                        + r"\right]_{"
                        + sp.latex(l)
                        + "}^{"
                        + sp.latex(r)
                        + "}"
                    )

                    if not (
                        l.has(sp.oo, -sp.oo)
                        or r.has(sp.oo, -sp.oo)
                    ):

                        evaluacion = (
                            F.subs(u, r)
                            - F.subs(u, l)
                        )

                        st.latex(
                            "="
                            + sp.latex(evaluacion)
                        )

                if v is not None:

                    st.latex(
                        "="
                        + sp.latex(sp.expand(v))
                    )

                else:

                    st.warning(
                        "Integral no resuelta o "
                        "convergencia no establecida."
                    )

            if total is not None:

                st.latex(
                    r"c_{\mathrm{reg}}(t)="
                    + sp.latex(sp.expand(total))
                )

        if x[1] or k[1]:

            st.markdown(
                "**7. Contribuciones de los impulsos**"
            )

            st.latex(
                r"f(t)*A\delta(t-p)=Af(t-p)"
            )

            for p, w in x[1]:

                st.latex(
                    sp.latex(w)
                    + movil
                    + r"_{\mathrm{reg}}"
                    + r"\left("
                    + sp.latex(t - p)
                    + r"\right)"
                )

            for p, w in k[1]:

                st.latex(
                    sp.latex(w)
                    + fijo
                    + r"_{\mathrm{reg}}"
                    + r"\left("
                    + sp.latex(t - p)
                    + r"\right)"
                )

            st.latex(
                r"A\delta(t-p)*B\delta(t-q)"
                r"=AB\delta(t-p-q)"
            )


# ==============================================
# SIMULADOR INTERACTIVO v0.7.1
#
# CORRECCION PRINCIPAL:
# El instante de la animacion es independiente
# del estado del widget slider.
#
# Los botones y las graficas pertenecen
# al mismo fragmento.
#
# STOP y PAUSA conservan el instante actual.
# ==============================================

def simulador(
    x,
    k,
    fijo,
    movil,
    minimo,
    maximo,
    tiempos,
    valores,
    impulsos,
    regular,
    extra,
    clave
):

    with st.expander(
        "Simulación interactiva de la convolución",
        expanded=False
    ):

        st.caption(
            "Desplaza la señal móvil o utiliza "
            "los controles de reproducción. "
            "STOP y Pausar conservan las gráficas "
            "en el instante seleccionado."
        )

        # ======================================
        # INICIALIZACION DEL ESTADO
        # ======================================

        if (
            st.session_state.get("sim_clave")
            != clave
        ):

            st.session_state["sim_clave"] = clave

            st.session_state["sim_instante"] = (
                float(minimo)
            )

            st.session_state["sim_slider_v071"] = (
                float(minimo)
            )

            st.session_state["sim_play"] = False

            st.session_state["sim_estado"] = (
                "Preparado"
            )

            st.session_state["sim_saltar_paso"] = (
                True
            )

        pares = preparar_pares(
            x[0],
            k[0]
        )

        # ======================================
        # FUNCIONES DE LOS BOTONES
        # ======================================

        def reproducir():

            if (
                st.session_state["sim_instante"]
                >= maximo
            ):

                st.session_state["sim_instante"] = (
                    float(minimo)
                )

            st.session_state["sim_play"] = True

            st.session_state["sim_estado"] = (
                "Reproduciendo"
            )

            st.session_state["sim_saltar_paso"] = (
                True
            )

        def pausar():

            st.session_state["sim_play"] = False

            st.session_state["sim_estado"] = (
                "En pausa"
            )

        def detener():

            st.session_state["sim_play"] = False

            st.session_state["sim_estado"] = (
                "Detenido"
            )

        def reiniciar():

            st.session_state["sim_play"] = False

            st.session_state["sim_instante"] = (
                float(minimo)
            )

            st.session_state["sim_estado"] = (
                "Preparado"
            )

        def movimiento_manual():

            st.session_state["sim_play"] = False

            st.session_state["sim_instante"] = (
                float(
                    st.session_state[
                        "sim_slider_v071"
                    ]
                )
            )

            st.session_state["sim_estado"] = (
                "Control manual"
            )

        # ======================================
        # FRAGMENTO INDEPENDIENTE
        #
        # No se utiliza st.rerun() dentro
        # de la reproduccion.
        #
        # No se utiliza un bucle bloqueante.
        # ======================================

        @st.fragment(run_every=0.8)
        def panel_animacion():

            # ----------------------------------
            # Actualizar solamente si reproduce
            # ----------------------------------

            if st.session_state["sim_play"]:

                if st.session_state.get(
                    "sim_saltar_paso",
                    False
                ):

                    st.session_state[
                        "sim_saltar_paso"
                    ] = False

                else:

                    paso = max(
                        0.1,
                        round(
                            (maximo - minimo) / 40,
                            1
                        )
                    )

                    nuevo = min(
                        float(maximo),
                        round(
                            st.session_state[
                                "sim_instante"
                            ] + paso,
                            1
                        )
                    )

                    st.session_state[
                        "sim_instante"
                    ] = nuevo

                    if nuevo >= maximo:

                        st.session_state[
                            "sim_play"
                        ] = False

                        st.session_state[
                            "sim_estado"
                        ] = "Finalizado"

            # ----------------------------------
            # Conservar el instante actual
            # ----------------------------------

            instante = float(
                st.session_state[
                    "sim_instante"
                ]
            )

            instante = min(
                float(maximo),
                max(
                    float(minimo),
                    instante
                )
            )

            st.session_state[
                "sim_instante"
            ] = instante

            # Actualizar el widget antes
            # de crearlo.

            st.session_state[
                "sim_slider_v071"
            ] = instante

            # ----------------------------------
            # CONTROLES DE REPRODUCCION
            # ----------------------------------

            c1, c2, c3, c4 = st.columns(4)

            c1.button(
                "▶ Reproducir",
                key="sim_reproducir_v071",
                on_click=reproducir,
                use_container_width=True
            )

            c2.button(
                "⏸ Pausar",
                key="sim_pausar_v071",
                on_click=pausar,
                use_container_width=True
            )

            c3.button(
                "⏹ STOP",
                key="sim_detener_v071",
                on_click=detener,
                use_container_width=True
            )

            c4.button(
                "↺ Reiniciar",
                key="sim_reiniciar_v071",
                on_click=reiniciar,
                use_container_width=True
            )

            # ----------------------------------
            # CONTROL DESLIZANTE
            # ----------------------------------

            st.slider(
                "Desplazamiento temporal t",
                min_value=float(minimo),
                max_value=float(maximo),
                step=0.1,
                key="sim_slider_v071",
                on_change=movimiento_manual
            )

            estado = st.session_state[
                "sim_estado"
            ]

            st.caption(
                f"Estado: {estado} · "
                f"t = {instante:.2f}"
            )

            # ==================================
            # GRAFICAS DEL INSTANTE ACTUAL
            #
            # Se generan siempre, incluso
            # cuando sim_play es False.
            # ==================================

            try:

                zr = sp.Rational(
                    str(instante)
                )

                xr, xi = x

                kr, ki = transformar(
                    k[0],
                    k[1],
                    sp.S.One,
                    -sp.S.One,
                    zr
                )

                cortes = sorted(
                    set(
                        fronteras(xr)
                        + fronteras(kr)
                    )
                )

                fig, axs = plt.subplots(
                    3,
                    1,
                    figsize=(11, 9)
                )

                try:

                    # ==========================
                    # GRAFICA 1
                    # SUPERPOSICION DE SEÑALES
                    # ==========================

                    dibujar(
                        axs[0],
                        lambda q: evaluar(xr, q),
                        cortes,
                        minimo,
                        maximo,
                        "royalblue",
                        f"Superposición: t = {instante:.2f}"
                    )

                    dibujar(
                        axs[0],
                        lambda q: evaluar(
                            kr,
                            q,
                            invertido=True
                        ),
                        cortes,
                        minimo,
                        maximo,
                        "seagreen",
                        f"Superposición: t = {instante:.2f}"
                    )

                    axs[0].plot(
                        [],
                        [],
                        color="royalblue",
                        label=f"{fijo}(τ): señal fija"
                    )

                    axs[0].plot(
                        [],
                        [],
                        color="seagreen",
                        label=f"{movil}(t−τ): señal móvil"
                    )

                    dibujar_impulsos(
                        axs[0],
                        xi,
                        minimo,
                        maximo,
                        "royalblue",
                        leyenda=False
                    )

                    dibujar_impulsos(
                        axs[0],
                        ki,
                        minimo,
                        maximo,
                        "seagreen",
                        leyenda=False
                    )

                    axs[0].legend(
                        loc="best",
                        fontsize=8
                    )

                    axs[0].set_xlabel(
                        "Variable τ"
                    )

                    # ==========================
                    # GRAFICA 2
                    # PRODUCTO DE LAS SEÑALES
                    # ==========================

                    def prod(q):

                        return (
                            evaluar(xr, q)
                            * evaluar(
                                kr,
                                q,
                                invertido=True
                            )
                        )

                    dibujar(
                        axs[1],
                        prod,
                        cortes,
                        minimo,
                        maximo,
                        "purple",
                        "Producto de las partes regulares"
                    )

                    divisiones = sorted(
                        set(
                            [minimo, maximo]
                            + [
                                c
                                for c in cortes
                                if minimo < c < maximo
                            ]
                        )
                    )

                    for a, b in zip(
                        divisiones[:-1],
                        divisiones[1:]
                    ):

                        eps = (b - a) * 1e-8

                        q = np.linspace(
                            a + eps,
                            b - eps,
                            250
                        )

                        axs[1].fill_between(
                            q,
                            prod(q),
                            alpha=0.20,
                            color="purple"
                        )

                    axs[1].set_xlabel(
                        "Variable τ"
                    )

                    # ==========================
                    # VALOR INSTANTANEO
                    # ==========================

                    if regular is not None:

                        valor_actual = float(
                            evaluar(
                                regular,
                                np.array([instante])
                            )[0]
                        )

                    else:

                        with warnings.catch_warnings():

                            warnings.simplefilter(
                                "error",
                                IntegrationWarning
                            )

                            valor_actual = (
                                convolucion_instante(
                                    pares,
                                    instante
                                )
                                + float(
                                    evaluar(
                                        extra,
                                        np.array([instante])
                                    )[0]
                                )
                            )

                    # ==========================
                    # GRAFICA 3
                    # FORMACION DE CONVOLUCION
                    # ==========================

                    axs[2].plot(
                        tiempos,
                        valores,
                        color="darkviolet",
                        alpha=0.25
                    )

                    mask = tiempos < instante

                    tx = np.r_[
                        tiempos[mask],
                        instante
                    ]

                    vy = np.r_[
                        valores[mask],
                        valor_actual
                    ]

                    axs[2].plot(
                        tx,
                        vy,
                        color="darkviolet",
                        lw=2
                    )

                    axs[2].axvline(
                        instante,
                        color="gray",
                        ls="--"
                    )

                    axs[2].scatter(
                        [instante],
                        [valor_actual],
                        color="darkviolet",
                        zorder=6
                    )

                    axs[2].axhline(
                        0,
                        color="gray",
                        lw=0.7
                    )

                    axs[2].set_title(
                        "Formación de la convolución c(t)"
                    )

                    axs[2].set_xlabel(
                        "Tiempo t"
                    )

                    axs[2].set_ylabel(
                        "Amplitud"
                    )

                    axs[2].grid(
                        alpha=0.25
                    )

                    configurar_eje_tiempo(
                        axs[2],
                        [],
                        minimo,
                        maximo
                    )

                    impulsos_visibles = [
                        (p, w)
                        for p, w in impulsos
                        if float(p) <= instante
                    ]

                    dibujar_impulsos(
                        axs[2],
                        impulsos_visibles,
                        minimo,
                        maximo
                    )

                    preparar_figura(fig)

                    st.pyplot(fig)

                finally:

                    plt.close(fig)

                # ==============================
                # RESULTADO INSTANTANEO
                # ==============================

                st.markdown(
                    "#### Resultado instantáneo"
                )

                st.latex(
                    r"c_{\mathrm{reg}}"
                    + rf"({instante:g})"
                    + "="
                    + f"{valor_actual:.4f}"
                )

                for p, w in impulsos:

                    if np.isclose(
                        float(p),
                        instante,
                        rtol=0,
                        atol=1e-9
                    ):

                        st.info(
                            "En este instante existe "
                            f"un impulso de peso {w}."
                        )

                # ==============================
                # LIMITES DE INTEGRACION
                # ==============================

                st.markdown(
                    "#### Límites de integración actuales"
                )

                filas = []

                for i, (fx, a, b) in enumerate(
                    x[0], 1
                ):

                    for j, (fk, c, d) in enumerate(
                        k[0], 1
                    ):

                        inferior = max(
                            float(a),
                            instante - float(d)
                        )

                        superior = min(
                            float(b),
                            instante - float(c)
                        )

                        if inferior < superior:

                            filas.append({
                                "Tramo fijo": i,
                                "Tramo móvil": j,
                                "Límite inferior": (
                                    f"{inferior:.4f}"
                                    if np.isfinite(inferior)
                                    else "-∞"
                                ),
                                "Límite superior": (
                                    f"{superior:.4f}"
                                    if np.isfinite(superior)
                                    else "+∞"
                                )
                            })

                if filas:

                    st.table(filas)

                else:

                    st.info(
                        "No existe superposición "
                        "entre las partes regulares."
                    )

                if x[1] or k[1]:

                    st.caption(
                        "Las contribuciones de los impulsos "
                        "se calculan mediante las propiedades "
                        "de Dirac."
                    )

            except Exception as error:

                st.session_state[
                    "sim_play"
                ] = False

                st.session_state[
                    "sim_estado"
                ] = "Error"

                st.error(
                    "No se pudo representar "
                    "este instante: "
                    + str(error)
                )

        # Ejecutar el fragmento siempre,
        # sin importar el estado de reproduccion.

        panel_animacion()

        st.caption(
            "La región sombreada representa "
            "el producto de las partes regulares "
            "dentro de la ventana visible. "
            "Las regiones exteriores también "
            "pueden contribuir a la integral."
        )


# ==============================================
# GUIA RAPIDA
# ==============================================

with st.expander(
    "📖 Guía rápida · Cómo ingresar las señales",
    expanded=True
):

    st.markdown("""
Escribe únicamente la expresión en función de **t**.
Coloca sus límites en los campos **Desde** y **Hasta**.

| Para ingresar | Escribe | Evita |
|---|---|---|
| Constante | `2` | `x(t)=2` |
| Rampa | `t+1` | `x=t+1` |
| Multiplicación | `2*t` | `2t` |
| Potencia | `t^2` o `t**2` | `t²` |
| Exponencial | `exp(-t)` | `e-t` |
| Exponencial alternativa | `e^(-t)` | |
| Seno | `sin(pi*t)` | `sen(pi*t)` |
| Coseno | `cos(t)` | |
| Raíz cuadrada | `sqrt(t)` | `√t` |
| Valor absoluto | `Abs(t)` | |
| Logaritmo natural | `log(t)` | `ln(t)` |
| Fracciones | `1/2` | |
| Decimales | `3.5` | `3,5` |
| Infinito positivo | `inf` o `+inf` | `00` |
| Infinito negativo | `-inf` | `-00` |

### ¿Cómo ingresar el infinito?

Escribe **inf**, utilizando las letras:

- **i**: letra i.
- **n**: letra ene.
- **f**: letra efe.

Para infinito negativo escribe **-inf**.

También puedes utilizar `∞`, `-∞`, `oo` o `-oo`.

En `oo` se utilizan dos letras o minúsculas,
no dos números cero.

### Consideraciones

- Los tramos cumplen Desde ≤ t < Hasta.
- Los intervalos no deben superponerse.
- Fuera de los intervalos la señal vale cero.
- Las funciones trigonométricas utilizan radianes.
- Para representar saltos, crea tramos separados.
- Para ingresar solamente impulsos, selecciona 0 tramos.
- Las integrales impropias deben converger.
""")

    st.info(
        "Ejemplo: Expresión: exp(-t), "
        "Desde: 0, Hasta: inf."
    )


# ==============================================
# INGRESO DE SEÑALES
# ==============================================

col1, col2 = st.columns(2)

with col1:

    x = editor_mixto("x")

with col2:

    k = editor_mixto("k")

if x is None or k is None:

    st.stop()


# ==============================================
# EXPRESIONES ANALITICAS
# ==============================================

st.divider()

st.header("Expresiones analíticas")

c1, c2 = st.columns(2)

with c1:

    mostrar_formula("x", *x)

with c2:

    mostrar_formula("k", *k)


# ==============================================
# REPRESENTACION GRAFICA
# ==============================================

st.divider()

st.header("Representación gráfica")

minimo, maximo = st.slider(
    "Intervalo de visualización",
    -30,
    30,
    (-5, 8)
)

if minimo >= maximo:

    st.warning(
        "Selecciona un intervalo válido."
    )

    st.stop()

graficar_mixta(
    *x,
    minimo,
    maximo,
    "royalblue",
    "Señal x(t)"
)

graficar_mixta(
    *k,
    minimo,
    maximo,
    "seagreen",
    "Señal k(t)"
)


# ==============================================
# OPERACIONES ENTRE SEÑALES
# ==============================================

st.divider()

st.header(
    "Operaciones entre señales"
)

op = st.selectbox(
    "Selecciona una operación",
    [
        "Suma",
        "Resta",
        "Multiplicación"
    ],
    key="operaciones_basicas"
)

try:

    res = operacion_mixta(
        x,
        k,
        op
    )

    letra = {
        "Suma": "s",
        "Resta": "r",
        "Multiplicación": "p"
    }[op]

    simbolo = {
        "Suma": "+",
        "Resta": "-",
        "Multiplicación": r"\cdot"
    }[op]

    st.latex(
        rf"{letra}(t)=x(t){simbolo}k(t)"
    )

    st.markdown(
        "#### Resultado analítico"
    )

    mostrar_formula(
        letra,
        *res
    )

    st.markdown(
        "#### Representación gráfica"
    )

    graficar_mixta(
        *res,
        minimo,
        maximo,
        "purple",
        op
    )

except Exception as error:

    st.warning(str(error))


# ==============================================
# MANIPULACION DE SEÑALES
# ==============================================

st.divider()

st.header(
    "Manipulación de señales"
)

with st.expander(
    "📖 Guía rápida · Cómo manipular las señales",
    expanded=False
):

    st.latex(
        r"\boxed{y(t)=A\,f[B(t-C)]}"
    )

    st.markdown("""
La manipulación de señales se realiza mediante
tres parámetros: **A, B y C**.

### Factor de amplitud y/o inversión A

- Si |A| > 1, aumenta la amplitud.
- Si 0 < |A| < 1, disminuye la amplitud.
- Si A < 0, existe inversión de amplitud.
- Si A = 1, la amplitud permanece igual.
- Si A = 0, se obtiene la señal nula.

### Factor de escalamiento temporal y/o inversión B

- Si |B| > 1, existe compresión temporal.
- Si 0 < |B| < 1, existe expansión temporal.
- Si B < 0, existe inversión temporal.
- Si |B| = 1, no existe escalamiento temporal.
- B debe ser distinto de cero.

### Desplazamiento temporal C

- Si C > 0, desplazamiento hacia la derecha.
- Si C < 0, desplazamiento hacia la izquierda.
- Si C = 0, no existe desplazamiento temporal.

### Orden de las transformaciones

1. Realizar el escalamiento temporal mediante B.
2. Realizar la inversión temporal si B es negativo.
3. Desplazar la señal mediante C.
4. Aplicar el escalamiento de amplitud mediante A.
5. Invertir la amplitud si A es negativo.
""")

    st.markdown(
        "### Forma expandida y forma factorizada"
    )

    st.latex(
        r"A\,f(Bt+D)=A\,f[B(t-C)]"
    )

    st.latex(
        r"\boxed{C=-\frac{D}{B}}"
    )

    st.markdown(
        "**Ejemplo de factorización**"
    )

    st.latex(
        r"y(t)=4x\left(-\frac{t}{2}+8\right)"
    )

    st.latex(
        r"y(t)=4x\left[-\frac{1}{2}(t-16)\right]"
    )

    st.latex(
        r"A=4,\qquad B=-\frac12,\qquad C=16"
    )

    st.caption(
        "El desplazamiento efectivo es de "
        "16 unidades hacia la derecha."
    )

    st.markdown(
        "### Ejemplos de ingreso"
    )

    st.table([
        {
            "Transformación": "x(t-2)",
            "A": "1",
            "B": "1",
            "C": "2"
        },
        {
            "Transformación": "x(t+2)",
            "A": "1",
            "B": "1",
            "C": "-2"
        },
        {
            "Transformación": "x(-t)",
            "A": "1",
            "B": "-1",
            "C": "0"
        },
        {
            "Transformación": "x(2t)",
            "A": "1",
            "B": "2",
            "C": "0"
        },
        {
            "Transformación": "x(t/2)",
            "A": "1",
            "B": "1/2",
            "C": "0"
        },
        {
            "Transformación": "-3x(t)",
            "A": "-3",
            "B": "1",
            "C": "0"
        },
        {
            "Transformación": "2x(3t-6)",
            "A": "2",
            "B": "3",
            "C": "2"
        },
        {
            "Transformación": "4x(-t/2+8)",
            "A": "4",
            "B": "-1/2",
            "C": "16"
        }
    ])

    st.markdown(
        "### Transformación de los límites"
    )

    st.latex(
        r"a\leq B(t-C)<b"
    )

    st.write(
        "Si B es positivo:"
    )

    st.latex(
        r"\frac{a}{B}+C"
        r"\leq t<"
        r"\frac{b}{B}+C"
    )

    st.write(
        "Si B es negativo:"
    )

    st.latex(
        r"\frac{b}{B}+C"
        r"<t\leq"
        r"\frac{a}{B}+C"
    )

    st.caption(
        "Cuando B es negativo, se invierten "
        "los límites y las desigualdades."
    )

    st.markdown(
        "### Manipulación de impulsos de Dirac"
    )

    st.latex(
        r"A\delta[B(t-C)-a]"
        r"=\frac{A}{|B|}"
        r"\delta\left(t-C-\frac{a}{B}\right)"
    )


# ==============================================
# EXPRESION GENERAL VISIBLE
# ==============================================

nombre_transformar = st.selectbox(
    "Señal a manipular",
    ["x", "k"],
    format_func=lambda nombre: f"{nombre}(t)",
    key="senal_transformacion"
)

st.markdown(
    "**Expresión general de manipulación**"
)

st.latex(
    rf"\boxed{{y(t)=A\,{nombre_transformar}"
    r"[B(t-C)]}"
)

c1, c2, c3 = st.columns(3)

texto_A = c1.text_input(
    "Factor de amplitud y/o inversión A",
    "1",
    key="manip_A"
)

texto_B = c2.text_input(
    "Factor de escalamiento temporal y/o inversión B",
    "1",
    key="manip_B"
)

texto_C = c3.text_input(
    "Desplazamiento temporal C",
    "0",
    key="manip_C"
)

try:

    A = interpretar_limite(texto_A)
    B = interpretar_limite(texto_B)
    C = interpretar_limite(texto_C)

    if not all(
        v.is_finite
        for v in (A, B, C)
    ):

        raise ValueError(
            "Los factores deben ser finitos."
        )

    if B == 0:

        raise ValueError(
            "El factor B debe ser distinto de cero."
        )

    senal = (
        x
        if nombre_transformar == "x"
        else k
    )

    transformada = manipulacion_factorizada(
        *senal,
        A,
        B,
        C
    )

    st.latex(
        latex_manipulacion(
            nombre_transformar,
            A,
            B,
            C
        )
    )

    with st.expander(
        "Interpretación de la manipulación",
        expanded=False
    ):

        if A == 0:

            st.write(
                "Amplitud: señal nula."
            )

        else:

            st.write(
                "Factor de amplitud: "
                + str(abs(A))
            )

            st.write(
                "Inversión de amplitud: "
                + (
                    "Sí"
                    if A < 0
                    else "No"
                )
            )

        if abs(B) > 1:

            st.write(
                "Compresión temporal por "
                f"un factor de {abs(B)}."
            )

        elif abs(B) < 1:

            st.write(
                "Expansión temporal por "
                f"un factor de {1 / abs(B)}."
            )

        else:

            st.write(
                "Sin escalamiento temporal."
            )

        st.write(
            "Inversión temporal: "
            + (
                "Sí"
                if B < 0
                else "No"
            )
        )

        if C > 0:

            st.write(
                f"Desplazamiento de {C} "
                "unidades hacia la derecha."
            )

        elif C < 0:

            st.write(
                f"Desplazamiento de {abs(C)} "
                "unidades hacia la izquierda."
            )

        else:

            st.write(
                "Sin desplazamiento temporal."
            )

        st.markdown(
            "**Forma expandida equivalente**"
        )

        st.latex(
            r"B(t-C)="
            + sp.latex(
                sp.expand(B * (t - C))
            )
        )

    st.markdown(
        "#### Resultado analítico"
    )

    mostrar_formula(
        "y",
        *transformada,
        invertido=(B < 0)
    )

    st.markdown(
        "#### Representación gráfica"
    )

    graficar_mixta(
        *transformada,
        minimo,
        maximo,
        "teal",
        "Señal transformada y(t)",
        saltos=True,
        invertido=(B < 0),
        extremos=True
    )

    if B < 0:

        st.caption(
            "La inversión temporal cambia "
            "los intervalos a abiertos por la "
            "izquierda y cerrados por la derecha."
        )

    st.caption(
        "Los círculos abiertos representan "
        "extremos excluidos y los círculos "
        "cerrados representan extremos incluidos."
    )

except Exception as error:

    st.warning(str(error))


# ==============================================
# DERIVACION E INTEGRACION
# ==============================================

st.divider()

st.header(
    "Operaciones sobre una señal"
)

sel1, sel2 = st.columns(2)

with sel1:

    nombre = st.selectbox(
        "Señal que deseas utilizar",
        ["x", "k"],
        format_func=lambda nombre: f"{nombre}(t)",
        key="senal_individual"
    )

with sel2:

    op_individual = st.selectbox(
        "Operación sobre la señal",
        [
            "Derivar",
            "Integrar"
        ]
    )

st.caption(
    "Se utiliza únicamente la señal seleccionada."
)

if st.button(
    "Calcular derivada"
    if op_individual == "Derivar"
    else "Calcular integral",
    key="calcular_individual"
):

    try:

        reg, imp = (
            x if nombre == "x"
            else k
        )

        if op_individual == "Derivar":

            if imp:

                raise ValueError(
                    "La derivación de impulsos "
                    "de entrada requiere delta prima. "
                    "Esta operación no está implementada."
                )

            resultado = derivar_senal(reg)

            st.latex(
                rf"d(t)=\frac{{d}}{{dt}}{nombre}(t)"
            )

            st.markdown(
                "#### Resultado analítico"
            )

            mostrar_formula(
                "d",
                *resultado
            )

            with st.expander(
                "Desarrollo matemático de la derivación",
                expanded=False
            ):

                st.markdown(
                    "**Derivación de cada tramo**"
                )

                for expr, a, b in reg:

                    st.latex(
                        r"\frac{d}{dt}\left("
                        + sp.latex(expr)
                        + r"\right)="
                        + sp.latex(
                            sp.diff(expr, t)
                        )
                    )

                    st.latex(
                        condicion_intervalo(a, b)
                    )

                st.markdown(
                    "**Cálculo de los impulsos**"
                )

                st.latex(
                    r"A=f(t_0^+)-f(t_0^-)"
                )

                for punto, peso in resultado[1]:

                    izquierda = sp.S.Zero
                    derecha = sp.S.Zero

                    for expr, a, b in reg:

                        if a < punto <= b:

                            izquierda = valor_finito(
                                expr,
                                punto,
                                "-"
                            )

                        if a <= punto < b:

                            derecha = valor_finito(
                                expr,
                                punto,
                                "+"
                            )

                    st.latex(
                        r"t_0="
                        + sp.latex(punto)
                    )

                    st.latex(
                        r"A="
                        + sp.latex(derecha)
                        + "-("
                        + sp.latex(izquierda)
                        + ")="
                        + sp.latex(peso)
                    )

                    st.latex(
                        sp.latex(
                            peso
                            * sp.DiracDelta(
                                t - punto
                            )
                        )
                    )

                if not resultado[1]:

                    st.info(
                        "La señal no presenta saltos "
                        "que generen impulsos de Dirac."
                    )

            st.markdown(
                "#### Representación gráfica"
            )

            graficar_mixta(
                *resultado,
                minimo,
                maximo,
                "darkorange",
                f"Derivada completa de {nombre}(t)",
                saltos=False
            )

            st.caption(
                "La derivada regular y los impulsos "
                "se representan en una sola gráfica. "
                "Las flechas indican el peso "
                "de los impulsos, no amplitudes finitas."
            )

        else:

            resultado = integrar_mixta(
                reg,
                imp
            )

            st.latex(
                rf"i(t)=\int_0^t"
                rf"{nombre}(\tau)\,d\tau"
            )

            st.markdown(
                "#### Resultado analítico"
            )

            mostrar_formula(
                "i",
                *resultado
            )

            st.markdown(
                "#### Representación gráfica"
            )

            graficar_mixta(
                *resultado,
                minimo,
                maximo,
                "teal",
                "Integral acumulada"
            )

            st.caption(
                "La integral se acumula desde cero. "
                "Las constantes se ajustan entre "
                "los tramos regulares. Para impulsos "
                "se utiliza la convención H(0)=1."
            )

    except Exception as error:

        st.error(str(error))


# ==============================================
# CONVOLUCION
# ==============================================

st.divider()

st.header("Convolución")

movil = st.selectbox(
    "Selecciona la señal móvil",
    ["k", "x"],
    format_func=lambda nombre: f"{nombre}(t)",
    key="senal_movil"
)

fijo = (
    "x"
    if movil == "k"
    else "k"
)

xf, km = (
    (x, k)
    if movil == "k"
    else (k, x)
)

st.caption(
    f"Señal fija: {fijo}(τ) · "
    f"Señal móvil: {movil}(t−τ)"
)

st.latex(
    rf"c(t)=x(t)*k(t)="
    rf"\int_{{-\infty}}^{{+\infty}}"
    rf"{fijo}(\tau)"
    rf"{movil}(t-\tau)\,d\tau"
)


# ==============================================
# PERSISTENCIA DE RESULTADOS
# ==============================================

clave = repr(
    (
        xf,
        km,
        minimo,
        maximo,
        movil
    )
)

if (
    st.session_state.get(
        "conv_clave_activa"
    )
    != clave
):

    st.session_state["sim_play"] = False

    st.session_state[
        "conv_clave_activa"
    ] = clave


if st.button(
    "Calcular convolución",
    type="primary"
):

    try:

        with st.spinner(
            "Calculando convolución..."
        ):

            st.session_state[
                "sim_play"
            ] = False

            st.session_state.pop(
                "sim_clave",
                None
            )

            for expr, a, b in (
                xf[0] + km[0]
            ):

                validar_intervalo(
                    expr,
                    a,
                    b
                )

            plan = plan_convolucion(
                xf[0],
                km[0]
            )

            regular, desarrollo = resolver_plan(
                plan
            )

            extra, imp = contribuciones_impulsos(
                xf,
                km
            )

            cortes = sorted({
                v
                for a, b, _ in plan
                for v in (a, b)
                if v.is_finite
            })

            cortes += fronteras(extra)

            tiempos, valores_num = convolucion_numerica(
                xf[0],
                km[0],
                minimo,
                maximo,
                cortes
            )

            referencia = (
                valores_num
                + evaluar(
                    extra,
                    tiempos
                )
            )

            if regular is not None:

                regular = sumar_tramos([
                    regular,
                    extra
                ])

                valores = evaluar(
                    regular,
                    tiempos
                )

                if not np.allclose(
                    valores,
                    referencia,
                    rtol=1e-4,
                    atol=1e-5
                ):

                    regular = None

                    valores = referencia

            else:

                valores = referencia

            if not np.all(
                np.isfinite(valores)
            ):

                raise ValueError(
                    "La integral no produjo "
                    "valores finitos."
                )

            st.session_state["conv_v071"] = (
                clave,
                plan,
                desarrollo,
                regular,
                imp,
                tiempos,
                valores,
                extra
            )

    except Exception as error:

        st.session_state.pop(
            "conv_v071",
            None
        )

        st.session_state[
            "sim_play"
        ] = False

        st.error(
            "No se pudo calcular "
            f"la convolución: {error}"
        )


# ==============================================
# RESULTADOS DE CONVOLUCION
# ==============================================

if (
    "conv_v071" in st.session_state
    and st.session_state["conv_v071"][0] == clave
):

    (
        _,
        plan,
        desarrollo,
        regular,
        imp,
        tiempos,
        valores,
        extra
    ) = st.session_state["conv_v071"]

    st.markdown(
        "### Resultado gráfico"
    )

    if regular is not None:

        graficar_mixta(
            regular,
            imp,
            minimo,
            maximo,
            "darkviolet",
            "Convolución c(t)"
        )

    else:

        fig, ax = plt.subplots(
            figsize=(11, 4)
        )

        try:

            ax.plot(
                tiempos,
                valores,
                color="darkviolet",
                linewidth=2
            )

            ax.fill_between(
                tiempos,
                valores,
                color="violet",
                alpha=0.15
            )

            ax.axhline(
                0,
                color="gray",
                lw=0.7
            )

            ax.set_title(
                "Convolución numérica c(t)"
            )

            ax.set_xlabel(
                "Tiempo t"
            )

            ax.set_ylabel(
                "Amplitud"
            )

            ax.grid(alpha=0.25)

            configurar_eje_tiempo(
                ax,
                [],
                minimo,
                maximo
            )

            dibujar_impulsos(
                ax,
                imp,
                minimo,
                maximo
            )

            preparar_figura(fig)

            st.pyplot(fig)

        finally:

            plt.close(fig)

    st.markdown(
        "### Resultado analítico"
    )

    if regular is not None:

        mostrar_formula(
            "c",
            regular,
            imp
        )

        st.success(
            "Se obtuvo la expresión analítica "
            "de la convolución."
        )

    else:

        st.warning(
            "No se obtuvo una solución simbólica "
            "completa que coincidiera con "
            "la verificación numérica."
        )

        st.info(
            "La gráfica de la parte regular "
            "corresponde a una solución numérica "
            "en el intervalo de visualización."
        )

        if imp:

            mostrar_formula(
                r"c_{\mathrm{imp}}",
                [],
                imp
            )

    # Desarrollo matematico desplegable.

    mostrar_desarrollo(
        plan,
        desarrollo,
        fijo,
        movil,
        xf,
        km
    )

    # Simulador corregido.

    simulador(
        xf,
        km,
        fijo,
        movil,
        minimo,
        maximo,
        tiempos,
        valores,
        imp,
        regular,
        extra,
        clave
    )


# ==============================================
# PIE DE PAGINA
# ==============================================

st.divider()

st.caption(
    "CONVOLAB Web v0.7.1 | "
    "Proyecto académico de señales continuas | "
    "ig: vax._.07"
)
