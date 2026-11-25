
import ast
import warnings
import time
import math
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad, IntegrationWarning
from matplotlib.ticker import FixedLocator, FormatStrFormatter

# ==============================================
# CONVOLAB WEB v0.6
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
# INTERPRETACIÓN DE EXPRESIONES
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
                        "Usa exponentes entre -100 y 100."
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
            "Usa solo t, números y las funciones de la guía."
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
                    "El infinito solo se admite en los límites."
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

    tramos.sort(key=lambda r: float(r[1]))

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
# FORMATO MATEMÁTICO
# ==============================================

def formato_limite(valor):

    if valor == sp.oo:
        return r"+\infty"

    if valor == -sp.oo:
        return r"-\infty"

    return sp.latex(valor)


def condicion_intervalo(a, b):

    if a == -sp.oo and b == sp.oo:
        return r"t\in\mathbb{R}"

    if a == -sp.oo:
        return rf"t<{formato_limite(b)}"

    if b == sp.oo:
        return rf"t\geq {formato_limite(a)}"

    return (
        rf"{formato_limite(a)}"
        rf"\leq t<{formato_limite(b)}"
    )


def latex_tramos(nombre, tramos):

    filas = []

    for expr, a, b in tramos:

        filas.append(
            rf"{sp.latex(expr)}"
            rf" & {condicion_intervalo(a, b)}"
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
        rf"{nombre}(t)="
        rf"\begin{{cases}}"
        rf"{cuerpo}"
        rf"\end{{cases}}"
    )


# ==============================================
# EVALUACIÓN DE SEÑALES
# ==============================================

def evaluar(tramos, puntos):

    puntos = np.asarray(
        puntos,
        dtype=float
    )

    salida = np.zeros_like(puntos)

    for expr, a, b in tramos:

        mask = (
            (puntos >= float(a))
            & (puntos < float(b))
        )

        if not np.any(mask):
            continue

        f = sp.lambdify(
            t, expr, "numpy"
        )

        with np.errstate(all="ignore"):

            valores = f(
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
# CONFIGURACIÓN DE EJES
# ==============================================

def configurar_eje_tiempo(
    ax, cortes, minimo, maximo
):

    if minimo >= maximo:

        raise ValueError(
            "El inicio de la ventana debe "
            "ser menor que el final."
        )

    ax.set_xlim(minimo, maximo)

    ticks = list(
        range(
            math.ceil(minimo),
            math.floor(maximo) + 1
        )
    )

    ax.xaxis.set_major_locator(
        FixedLocator(ticks)
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.0f")
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
# GRÁFICAS POR INTERVALOS
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

    divisiones = [minimo]

    divisiones += [
        x for x in cortes
        if minimo < x < maximo
    ]

    divisiones += [maximo]

    divisiones = sorted(
        set(divisiones)
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
            ys,
            xs.shape
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

    ax.set_xlim(minimo, maximo)
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
# OPERACIONES POR TRAMOS
# ==============================================

def expresion_en(tramos, muestra):

    for expr, a, b in tramos:

        if float(a) <= muestra < float(b):
            return expr

    return sp.Integer(0)


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
            and sp.simplify(
                expr - anterior
            ) == 0
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
    x, k, operacion
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

        fx = expresion_en(
            x, muestra
        )

        fk = expresion_en(
            k, muestra
        )

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

    resultado = combinar_iguales(
        resultado
    )

    return [
        tramo
        for tramo in resultado
        if tramo[0] != 0
    ]


# ==============================================
# VALIDACIÓN MATEMÁTICA
# ==============================================

def valor_finito(
    expr, punto, direccion
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
            f"real y finito en t = {punto}. "
            "Esta operación no admite esa singularidad."
        )

    return valor


def validar_intervalo(expr, a, b):

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
            "No se pudo verificar el dominio. "
            "Divide la señal en tramos continuos."
        )

    if valido is not True:

        raise ValueError(
            "La expresión contiene una discontinuidad "
            "interna, singularidad o zona no real. "
            "Divide la señal en esos puntos."
        )


# ==============================================
# DERIVACIÓN DE SEÑALES
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
                "Divide esta expresión en tramos "
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

        izquierda = sp.Integer(0)
        derecha = sp.Integer(0)

        for expr, a, b in tramos:

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
# INTEGRACIÓN DE SEÑALES
# ==============================================

def integrar_senal(tramos):

    limites = {
        -sp.oo,
        sp.oo,
        sp.Integer(0)
    }

    for expr, a, b in tramos:

        validar_intervalo(
            expr, a, b
        )

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

    acumulado = sp.Integer(0)

    for a, b, primitiva in celdas:

        if a < 0:
            continue

        referencia = valor_finito(
            primitiva,
            a,
            "+"
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
                expr,
                b,
                "-"
            )

    acumulado = sp.Integer(0)

    for a, b, primitiva in reversed(celdas):

        if b > 0:
            continue

        referencia = valor_finito(
            primitiva,
            b,
            "-"
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
                expr,
                a,
                "+"
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

        st.latex(
            r"A\delta(t-a)"
        )

        st.caption(
            "a: posición del impulso. "
            "A: peso del impulso. "
            "Para una señal formada únicamente por "
            "impulsos, selecciona 0 tramos."
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
                        "Posición y peso deben ser finitos."
                    )

                impulsos.append(
                    (p, w)
                )

            except Exception as e:

                st.error(str(e))
                return None

    return (
        tramos,
        normalizar_impulsos(impulsos)
    )


# ==============================================
# EXPRESIÓN ANALÍTICA DE SEÑALES MIXTAS
# ==============================================

def mostrar_formula(
    nombre,
    tramos,
    impulsos=()
):

    if tramos:

        st.latex(
            latex_tramos(
                nombre
                if not impulsos
                else nombre + r"_{reg}",
                tramos
            )
        )

    if impulsos:

        d = sum(
            (
                w * sp.DiracDelta(t - p)
                for p, w in impulsos
            ),
            sp.S.Zero
        )

        delta_latex = sp.latex(d)

        if tramos:

            signo = (
                ""
                if delta_latex.startswith("-")
                else "+"
            )

            st.latex(
                nombre
                + r"(t)="
                + nombre
                + r"_{reg}(t)"
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
# REPRESENTACIÓN GRÁFICA DE IMPULSOS
# ==============================================

def dibujar_impulsos(
    ax,
    impulsos,
    minimo,
    maximo,
    color="crimson"
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

        y = altura * float(w) / escala

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
                6 if y > 0 else -15
            ),
            textcoords="offset points",
            color=color,
            zorder=7
        )

    ax.set_ylim(
        min(
            lo,
            min(extremos) - altura * 0.25
        ),
        max(
            hi,
            max(extremos) + altura * 0.25
        )
    )

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
# GRÁFICAS MIXTAS
# PARTE REGULAR + IMPULSOS
# ==============================================

def graficar_mixta(
    tramos,
    impulsos,
    minimo,
    maximo,
    color,
    titulo,
    saltos=True
):

    fig, ax = plt.subplots(
        figsize=(11, 4)
    )

    try:

        dibujar(
            ax,
            lambda z: evaluar(tramos, z),
            fronteras(tramos),
            minimo,
            maximo,
            color,
            titulo,
            unir_saltos=saltos
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
# MANIPULACIÓN DE SEÑALES
# y(t) = A*x(B*t+C)
# ==============================================

def transformar(
    tramos,
    impulsos,
    A,
    B,
    C
):

    if B == 0:

        raise ValueError(
            "B debe ser distinto de cero."
        )

    if A == 0:

        return [], []

    salida = []

    for e, a, b in tramos:

        l = (a - C) / B
        r = (b - C) / B

        if B < 0:

            l, r = r, l

        expresion = sp.simplify(
            A * e.subs(
                t,
                B * t + C
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
            (p - C) / B,
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


def sumar_tramos(lista):

    salida = []

    for tramos in lista:

        salida = operacion_analitica(
            salida,
            tramos,
            "Suma"
        )

    return salida


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
            f"discontinuidad (t={p}): "
            "no definido sin una convención adicional."
        )

    return derecha


def operacion_mixta(
    x, k, operacion
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
        sumar_tramos(
            [reg] + extras
        ),
        []
    )


# ==============================================
# CONVOLUCIÓN NUMÉRICA
# ==============================================

def convolucion_numerica(
    x, k, minimo, maximo
):

    tiempos = np.linspace(
        minimo,
        maximo,
        151
    )

    valores = []

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

    for tiempo in tiempos:

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

        valores.append(total)

    return (
        tiempos,
        np.asarray(valores)
    )


# ==============================================
# CONVOLUCIÓN ANALÍTICA
# DETERMINACIÓN DE INTERVALOS
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

        muestra = punto_muestra(
            a, b
        )

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
# RESOLUCIÓN DE LAS INTEGRALES
# ==============================================

def resolver_plan(plan):

    tramos = []
    desarrollo = []

    completo = True

    for a, b, partes in plan:

        total = sp.S.Zero
        pasos = []

        for i, j, l, r, f in partes:

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
                ):

                    valor = None

            except Exception:

                F = None
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

        total = sp.simplify(total)
        total = sp.expand(total)

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
# CONVOLUCIÓN CON IMPULSOS DE DIRAC
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
# DESARROLLO MATEMÁTICO PASO A PASO
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
                r"t\in\{"
                + ", ".join(
                    map(sp.latex, cortes)
                )
                + r"\}"
            )

        else:

            st.write(
                "No existen puntos de cambio finitos."
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
            "Se integra únicamente cuando L < U. "
            "Los extremos aislados no cambian "
            "las integrales de las partes regulares."
        )

        filas = []

        for a, b, pasos, total in desarrollo:

            if not pasos:

                filas.append({
                    "Intervalo": f"{a} a {b}",
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
                    "No existe superposición."
                )

                st.latex(
                    r"c_{reg}(t)=0"
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

                    if (
                        l.is_finite
                        and r.is_finite
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
                        + sp.latex(
                            sp.expand(v)
                        )
                    )

                else:

                    st.warning(
                        "Integral no resuelta "
                        "o convergencia no establecida."
                    )

            if total is not None:

                st.latex(
                    r"c_{reg}(t)="
                    + sp.latex(
                        sp.expand(total)
                    )
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
                    + r"_{reg}(t-("
                    + sp.latex(p)
                    + "))"
                )

            for p, w in k[1]:

                st.latex(
                    sp.latex(w)
                    + fijo
                    + r"_{reg}(t-("
                    + sp.latex(p)
                    + "))"
                )

            st.latex(
                r"A\delta(t-p)*B\delta(t-q)"
                r"=AB\delta(t-p-q)"
            )


# ==============================================
# SIMULADOR INTERACTIVO DE CONVOLUCIÓN
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
    impulsos
):

    with st.expander(
        "Simulación interactiva de la convolución",
        expanded=False
    ):

        st.caption(
            "Desplaza la señal móvil para observar "
            "cómo se forma la convolución."
        )

        instante = st.slider(
            "Desplazamiento temporal t",
            float(minimo),
            float(maximo),
            float(minimo),
            step=0.1,
            key="sim_t"
        )

        def cuadro(z):

            zr = sp.Rational(str(z))

            xr, xi = x

            kr, ki = transformar(
                k[0],
                k[1],
                sp.S.One,
                -sp.S.One,
                zr
            )

            fig, axs = plt.subplots(
                3,
                1,
                figsize=(11, 9)
            )

            try:

                cortes = sorted(
                    set(
                        fronteras(xr)
                        + fronteras(kr)
                    )
                )

                # Primera gráfica:
                # Señal fija y señal móvil

                dibujar(
                    axs[0],
                    lambda q: evaluar(xr, q),
                    cortes,
                    minimo,
                    maximo,
                    "royalblue",
                    f"Superposición: t = {z:.2f}"
                )

                dibujar(
                    axs[0],
                    lambda q: evaluar(kr, q),
                    cortes,
                    minimo,
                    maximo,
                    "seagreen",
                    f"Superposición: t = {z:.2f}"
                )

                axs[0].plot(
                    [],
                    [],
                    color="royalblue",
                    label=f"{fijo}(τ): fija"
                )

                axs[0].plot(
                    [],
                    [],
                    color="seagreen",
                    label=f"{movil}(t−τ): móvil"
                )

                dibujar_impulsos(
                    axs[0],
                    xi,
                    minimo,
                    maximo,
                    "royalblue"
                )

                dibujar_impulsos(
                    axs[0],
                    ki,
                    minimo,
                    maximo,
                    "seagreen"
                )

                axs[0].legend(
                    fontsize=8
                )

                axs[0].set_xlabel(
                    "Variable τ"
                )

                # Segunda gráfica:
                # Producto de las señales

                def prod(q):

                    return (
                        evaluar(xr, q)
                        * evaluar(kr, q)
                    )

                dibujar(
                    axs[1],
                    prod,
                    cortes,
                    minimo,
                    maximo,
                    "purple",
                    "Producto de las señales"
                )

                q = np.linspace(
                    minimo,
                    maximo,
                    800
                )

                axs[1].fill_between(
                    q,
                    prod(q),
                    alpha=0.2,
                    color="purple"
                )

                axs[1].set_xlabel(
                    "Variable τ"
                )

                # Tercera gráfica:
                # Formación de la convolución

                axs[2].plot(
                    tiempos,
                    valores,
                    color="darkviolet",
                    alpha=0.25
                )

                mask = tiempos <= z

                axs[2].plot(
                    tiempos[mask],
                    valores[mask],
                    color="darkviolet",
                    lw=2
                )

                axs[2].axvline(
                    z,
                    color="gray",
                    ls="--"
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

                dibujar_impulsos(
                    axs[2],
                    impulsos,
                    minimo,
                    maximo
                )

                preparar_figura(fig)

                return fig

            except Exception:

                plt.close(fig)
                raise

        lugar = st.empty()

        def pintar(z):

            fig = cuadro(z)

            try:

                lugar.pyplot(fig)

            finally:

                plt.close(fig)

        pintar(instante)

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
                        "Límite inferior": inferior,
                        "Límite superior": superior
                    })

        if filas:

            st.table(filas)

        else:

            st.info(
                "No existe superposición "
                "entre las partes regulares."
            )

        st.caption(
            "La región sombreada representa el "
            "producto regular dentro de la ventana "
            "visible. Las regiones fuera de la ventana "
            "también pueden contribuir a la integral. "
            "Los impulsos se calculan por sus propiedades."
        )

        if st.button(
            "▶ Reproducir movimiento",
            key="play_conv"
        ):

            for z in np.linspace(
                minimo,
                maximo,
                24
            ):

                pintar(float(z))

                time.sleep(0.08)

        st.caption(
            "Utiliza el deslizador para examinar "
            "cualquier instante de la convolución."
        )


# ==============================================
# INTERFAZ PRINCIPAL
# GUÍA RÁPIDA
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
| Exponencial alternativa | `e^(-t)` | `e^-t` |
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

- **i**: i latina.
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
- Para una señal formada solo por impulsos,
  selecciona 0 tramos.
- Las integrales con límites infinitos deben converger.
""")

    st.info(
        "Ejemplo: para una exponencial decreciente "
        "desde cero hasta infinito, escribe: "
        "Expresión: exp(-t), Desde: 0, Hasta: inf."
    )


# ==============================================
# INGRESO DE LAS SEÑALES
# ==============================================

col1, col2 = st.columns(2)

with col1:

    x = editor_mixto("x")

with col2:

    k = editor_mixto("k")

if x is None or k is None:

    st.stop()


# ==============================================
# EXPRESIONES ANALÍTICAS
# ==============================================

st.divider()

st.header("Expresiones analíticas")

c1, c2 = st.columns(2)

with c1:

    mostrar_formula("x", *x)

with c2:

    mostrar_formula("k", *k)


# ==============================================
# REPRESENTACIÓN GRÁFICA
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
        x, k, op
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

except Exception as e:

    st.warning(str(e))


# ==============================================
# MANIPULACIÓN DE SEÑALES
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
        r"y(t)=Af(Bt+C)"
    )

    st.markdown("""
La transformación se realiza mediante tres parámetros.

### Parámetro A

Controla la amplitud.

- A > 1: aumenta la amplitud.
- 0 < A < 1: reduce la amplitud.
- A < 0: invierte la amplitud.
- A = 0: señal nula.

### Parámetro B

Controla la escala temporal.

- B > 1: compresión temporal.
- 0 < B < 1: expansión temporal.
- B < 0: reflexión temporal.
- B = 0: no admitido.

### Parámetro C

Determina el desplazamiento junto con B.

El desplazamiento efectivo se obtiene mediante -C/B.

### Ejemplos
""")

    st.table([
        {
            "Transformación": "x(t-2)",
            "A": "1",
            "B": "1",
            "C": "-2"
        },
        {
            "Transformación": "x(t+2)",
            "A": "1",
            "B": "1",
            "C": "2"
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
            "Transformación": "3x(t)",
            "A": "3",
            "B": "1",
            "C": "0"
        },
        {
            "Transformación": "2x(3t-6)",
            "A": "2",
            "B": "3",
            "C": "-6"
        }
    ])

    st.markdown(
        "### Transformación de los límites"
    )

    st.latex(
        r"t_1=\frac{a-C}{B}"
    )

    st.latex(
        r"t_2=\frac{b-C}{B}"
    )

    st.write(
        "Cuando B es negativo, se invierte "
        "el orden de los extremos."
    )

    st.caption(
        "Para señales reflejadas, el programa "
        "mantiene la convención de intervalos "
        "semiabiertos. Esto es equivalente casi "
        "en todas partes, pero puede cambiar "
        "el valor en puntos aislados."
    )

    st.latex(
        r"\delta(Bt+C-a)"
        r"=\frac{1}{|B|}"
        r"\delta\left(t-\frac{a-C}{B}\right)"
    )

nombre_transformar = st.selectbox(
    "Señal a manipular",
    ["x", "k"],
    format_func=lambda nombre: f"{nombre}(t)",
    key="senal_transformacion"
)

c1, c2, c3 = st.columns(3)

try:

    A = interpretar_limite(
        c1.text_input(
            "Factor de amplitud A",
            "1"
        )
    )

    B = interpretar_limite(
        c2.text_input(
            "Factor temporal B",
            "1"
        )
    )

    C = interpretar_limite(
        c3.text_input(
            "Término de desplazamiento C",
            "0"
        )
    )

    if not all(
        v.is_finite
        for v in (A, B, C)
    ):

        raise ValueError(
            "Los factores deben ser finitos."
        )

    senal = (
        x if nombre_transformar == "x"
        else k
    )

    transformada = transformar(
        *senal,
        A,
        B,
        C
    )

    prefijo = (
        ""
        if A == 1
        else "-"
        if A == -1
        else sp.latex(A)
    )

    st.latex(
        r"y(t)="
        + prefijo
        + nombre_transformar
        + r"\left("
        + sp.latex(B*t + C)
        + r"\right)"
    )

    st.markdown(
        "#### Resultado analítico"
    )

    mostrar_formula(
        "y",
        *transformada
    )

    st.markdown(
        "#### Representación gráfica"
    )

    graficar_mixta(
        *transformada,
        minimo,
        maximo,
        "teal",
        "Señal transformada y(t)"
    )

except Exception as e:

    st.warning(str(e))


# ==============================================
# DERIVACIÓN E INTEGRACIÓN
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
                    "La derivada de impulsos de entrada "
                    "requiere delta prima, δ′(t). "
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
                    "**Cálculo de impulsos**"
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
                False
            )

            st.caption(
                "La parte regular y los impulsos "
                "se representan en una sola gráfica. "
                "Las flechas indican pesos, "
                "no amplitudes finitas."
            )

        else:

            resultado = integrar_mixta(
                reg, imp
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
                "Las constantes de integración se ajustan "
                "entre los tramos. Para impulsos se usa "
                "la convención del escalón H(0)=1."
            )

    except Exception as e:

        st.error(str(e))


# ==============================================
# CONVOLUCIÓN
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
    "x" if movil == "k"
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

# Guardar el resultado para que no desaparezca
# al interactuar con el simulador.

clave = repr(
    (
        xf,
        km,
        minimo,
        maximo,
        movil
    )
)

if st.button(
    "Calcular convolución",
    type="primary"
):

    try:

        with st.spinner(
            "Calculando convolución..."
        ):

            for expr, a, b in (
                xf[0] + km[0]
            ):

                validar_intervalo(
                    expr, a, b
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

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "error",
                    IntegrationWarning
                )

                tiempos, valores = convolucion_numerica(
                    xf[0],
                    km[0],
                    minimo,
                    maximo
                )

            tiempos = np.unique(
                np.r_[
                    tiempos,
                    [
                        float(v)
                        for a, b, _ in plan
                        for v in (a, b)
                        if v.is_finite
                        and minimo <= float(v) <= maximo
                    ]
                ]
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

            else:

                with warnings.catch_warnings():

                    warnings.simplefilter(
                        "error",
                        IntegrationWarning
                    )

                    tiempos, valores = convolucion_numerica(
                        xf[0],
                        km[0],
                        minimo,
                        maximo
                    )

                valores += evaluar(
                    extra,
                    tiempos
                )

            if not np.all(
                np.isfinite(valores)
            ):

                raise ValueError(
                    "La integral no produjo "
                    "valores finitos."
                )

            st.session_state["conv_v06"] = (
                clave,
                plan,
                desarrollo,
                regular,
                imp,
                tiempos,
                valores
            )

    except Exception as e:

        st.session_state.pop(
            "conv_v06",
            None
        )

        st.error(
            "No se pudo calcular "
            f"la convolución: {e}"
        )


# ==============================================
# MOSTRAR RESULTADOS DE LA CONVOLUCIÓN
# ==============================================

if (
    "conv_v06" in st.session_state
    and st.session_state["conv_v06"][0] == clave
):

    (
        _,
        plan,
        desarrollo,
        regular,
        imp,
        tiempos,
        valores
    ) = st.session_state["conv_v06"]

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
            "No se obtuvo una solución "
            "simbólica completa."
        )

        st.info(
            "La gráfica de la parte regular "
            "corresponde a una solución numérica."
        )

        if imp:

            mostrar_formula(
                "c_{imp}",
                [],
                imp
            )

    # Desarrollo matemático desplegable.

    mostrar_desarrollo(
        plan,
        desarrollo,
        fijo,
        movil,
        xf,
        km
    )

    # Simulación interactiva desplegable.

    simulador(
        xf,
        km,
        fijo,
        movil,
        minimo,
        maximo,
        tiempos,
        valores,
        imp
    )


# ==============================================
# PIE DE PÁGINA
# ==============================================

st.divider()

st.caption(
    "CONVOLAB Web v0.6 | "
    "Proyecto académico de señales continuas | "
    "ig: vax._.07"
)
