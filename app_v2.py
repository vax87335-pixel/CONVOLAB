import ast
import warnings
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad, IntegrationWarning

# ==============================================
# CONVOLAB WEB v0.5
# ==============================================

st.set_page_config(
    page_title="CONVOLAB",
    page_icon="📈",
    layout="wide"
)

t = sp.Symbol("t", real=True)
u = sp.Symbol("u", real=True)

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
    "Operaciones · Convolución · Derivación · Integración"
)

# ==============================================
# INTERPRETACION
# ==============================================

def interpretar(texto):
    texto = (
        texto.strip()
        .replace("^", "**")
        .replace("∞", "oo")
        .replace("−", "-")
    )
    # Interpretar únicamente números, operaciones y funciones de la guía.
    # No ejecutar texto introducido en el formulario.
    if not texto or len(texto) > 500:
        raise ValueError("Escribe una expresión de hasta 500 caracteres.")
    arbol = ast.parse(texto, mode="eval")
    if sum(1 for _ in ast.walk(arbol)) > 120:
        raise ValueError("La expresión es demasiado extensa.")

    def convertir(nodo):
        if isinstance(nodo, ast.Constant) and type(nodo.value) in (int, float):
            return sp.Rational(str(nodo.value))
        if isinstance(nodo, ast.Name) and nodo.id in PERMITIDAS:
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
            izquierda, derecha = convertir(nodo.left), convertir(nodo.right)
            if isinstance(nodo.op, ast.Add):
                return izquierda + derecha
            if isinstance(nodo.op, ast.Sub):
                return izquierda - derecha
            if isinstance(nodo.op, ast.Mult):
                return izquierda * derecha
            if isinstance(nodo.op, ast.Div):
                return izquierda / derecha
            if isinstance(nodo.op, ast.Pow):
                if derecha.is_number and abs(float(derecha)) > 100:
                    raise ValueError("Usa exponentes entre -100 y 100.")
                return izquierda ** derecha
        if (isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name)
                and nodo.func.id in PERMITIDAS and not nodo.keywords
                and len(nodo.args) == 1):
            funcion = PERMITIDAS[nodo.func.id]
            if callable(funcion):
                return funcion(convertir(nodo.args[0]))
        raise ValueError("Usa solo t, números y las funciones de la guía.")

    expr = convertir(arbol.body)
    if expr.has(sp.nan, sp.zoo) or expr.is_real is False:
        raise ValueError("La expresión debe ser real y estar definida.")
    return expr


def interpretar_limite(texto):
    valor = interpretar(texto)

    if valor.free_symbols:
        raise ValueError("El límite debe ser constante.")

    if valor.is_extended_real is not True:
        raise ValueError("El límite debe ser real.")

    return valor

# ==============================================
# EDITOR DE SEÑALES
# ==============================================

def editor(nombre):
    st.markdown(f"### Señal {nombre}(t)")

    cantidad = st.number_input(
        "Número de tramos",
        min_value=1,
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
                value=str(i * (2 if nombre == "x" else 1)),
                key=f"{nombre}_ini_{i}"
            )

        with c3:
            final = st.text_input(
                "Hasta",
                value=str((i + 1) * (2 if nombre == "x" else 1)),
                key=f"{nombre}_fin_{i}"
            )

        try:
            expr = interpretar(expresion)
            if expr.has(sp.oo, -sp.oo):
                raise ValueError("El infinito solo se admite en los límites.")
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

    for anterior, actual in zip(tramos, tramos[1:]):
        if anterior[2] > actual[1]:
            st.error("Los tramos no deben superponerse.")
            return None

    return tramos

# ==============================================
# FORMATO MATEMATICO POR TRAMOS
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

    # Evitar repetir un caso cero cuando la
    # expresion cubre todos los numeros reales.
    if not (
        len(tramos) == 1
        and tramos[0][1] == -sp.oo
        and tramos[0][2] == sp.oo
    ):
        filas.append(r"0 & \text{otro caso}")

    cuerpo = r" \\ ".join(filas)

    return (
        rf"{nombre}(t)="
        rf"\begin{{cases}}"
        rf"{cuerpo}"
        rf"\end{{cases}}"
    )

# ==============================================
# EVALUACION DE SEÑALES
# ==============================================

def evaluar(tramos, puntos):
    puntos = np.asarray(puntos, dtype=float)
    salida = np.zeros_like(puntos)

    for expr, a, b in tramos:
        mask = (
            (puntos >= float(a))
            & (puntos < float(b))
        )

        if not np.any(mask):
            continue

        f = sp.lambdify(t, expr, "numpy")

        with np.errstate(all="ignore"):
            valores = f(puntos[mask])

        salida[mask] = np.broadcast_to(
            np.asarray(valores, dtype=float),
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
# MARCAS DE TIEMPO DE UNO EN UNO
# ==============================================

def cortes_convolucion(x, k):
    # También sirve cuando el cálculo simbólico no está disponible.
    return sorted({
        a + b for a in fronteras(x) for b in fronteras(k)
        if np.isfinite(a + b)
    })


def configurar_eje_tiempo(ax, cortes, minimo, maximo):
    from matplotlib.ticker import FixedLocator, FormatStrFormatter
    import math

    if minimo >= maximo:
        raise ValueError("El inicio de la ventana debe ser menor que el final.")

    ax.set_xlim(minimo, maximo)
    # Numeración uniforme: ..., -2, -1, 0, 1, 2, ...
    ticks = list(range(math.ceil(minimo), math.floor(maximo) + 1))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(axis="x", labelsize=9, rotation=0, pad=5)
    ax.minorticks_off()


def preparar_figura(fig):
    # Mantener las etiquetas en una sola fila.
    # Girarlas únicamente si la ventana contiene muchos enteros.
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        etiquetas = ax.get_xticklabels()
        cajas = [etiqueta.get_window_extent(renderer) for etiqueta in etiquetas]
        if any(a.x1 + 5 > b.x0 for a, b in zip(cajas, cajas[1:])):
            for etiqueta in etiquetas:
                etiqueta.set_rotation(90)
                etiqueta.set_horizontalalignment("center")
        ax.xaxis.labelpad = 8
    fig.tight_layout()


# ==============================================
# GRAFICAS POR INTERVALOS
# ==============================================

def dibujar(ax, funcion, cortes, minimo, maximo,
            color, titulo, unir_saltos=True):

    divisiones = [minimo]
    divisiones += [
        x for x in cortes
        if minimo < x < maximo
    ]
    divisiones += [maximo]
    divisiones = sorted(set(divisiones))

    for a, b in zip(
        divisiones[:-1], divisiones[1:]
    ):
        if a >= b:
            continue

        eps = (b - a) * 1e-8
        xs = np.linspace(a + eps, b - eps, 350)
        ys = np.asarray(funcion(xs), dtype=float)

        ys = np.broadcast_to(ys, xs.shape)
        ax.plot(xs, np.ma.masked_invalid(ys), color=color, lw=2)

    # Uniones verticales en las discontinuidades.
    for x in (cortes if unir_saltos else []):
        if not minimo < x < maximo:
            continue

        eps = min(1e-7, (maximo - minimo) * 1e-9)

        izquierda = float(
            funcion(np.array([x - eps]))[0]
        )
        derecha = float(
            funcion(np.array([x + eps]))[0]
        )

        if (
            np.isfinite(izquierda)
            and np.isfinite(derecha)
            and not np.isclose(
                izquierda, derecha,
                rtol=1e-6, atol=1e-7
            )
        ):
            ax.plot(
                [x, x],
                [izquierda, derecha],
                color=color,
                lw=1.7
            )

    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlim(minimo, maximo)
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo t")
    configurar_eje_tiempo(ax, cortes, minimo, maximo)
    ax.grid(alpha=0.25)

# ==============================================
# SIMPLIFICAR OPERACIONES POR TRAMOS
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
    return (float(a) + float(b)) / 2

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
            resultado[-1] = (anterior, inicio, b)
        else:
            resultado.append((expr, a, b))

    return resultado

def operacion_analitica(x, k, operacion):
    limites = {-sp.oo, sp.oo}

    for expr, a, b in x + k:
        limites.add(a)
        limites.add(b)

    limites = sorted(limites, key=float)

    resultado = []

    for a, b in zip(limites[:-1], limites[1:]):
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
        resultado.append((expr, a, b))

    resultado = combinar_iguales(resultado)

    # El cero se interpreta como otro caso.
    return [
        tramo for tramo in resultado
        if tramo[0] != 0
    ]

# ==============================================
# CONVOLUCION NUMERICA
# ==============================================

def convolucion_numerica(x, k, minimo, maximo):
    tiempos = np.linspace(minimo, maximo, 151)
    valores = []

    pares = []

    for fx, a, b in x:
        for fk, c, d in k:
            expresion = (
                fx.subs(t, u)
                * fk.subs(t, t - u)
            )

            funcion = sp.lambdify(
                (u, t), expresion, "numpy"
            )

            pares.append(
                (
                    funcion,
                    float(a), float(b),
                    float(c), float(d)
                )
            )

    for tiempo in tiempos:
        total = 0.0

        for f, a, b, c, d in pares:
            inferior = max(a, tiempo - d)
            superior = min(b, tiempo - c)

            if inferior >= superior:
                continue

            integral, error = quad(
                lambda z: float(f(z, tiempo)),
                inferior,
                superior,
                epsabs=1e-7,
                epsrel=1e-6,
                limit=150
            )

            total += integral

        valores.append(total)

    return tiempos, np.asarray(valores)

# ==============================================
# CONVOLUCION ANALITICA
# PARA SEÑALES DE SOPORTE FINITO
# ==============================================

def convolucion_analitica(x, k):

    # Primera version simbolica:
    # necesita extremos finitos.
    for expr, a, b in x + k:
        if not (a.is_finite and b.is_finite):
            return None, (
                "La solución analítica automática "
                "todavía está limitada a señales "
                "con intervalos finitos."
            )

    # Los puntos de cambio de la convolucion
    # surgen de la suma de extremos.
    cortes = set()

    for fx, a, b in x:
        for fk, c, d in k:
            cortes.update([
                a+c, a+d, b+c, b+d
            ])

    cortes = sorted(cortes, key=float)

    if len(cortes) > 16:
        return None, (
            "Hay demasiados intervalos para "
            "esta versión del cálculo simbólico."
        )

    tramos = []

    for inicio, fin in zip(
        cortes[:-1], cortes[1:]
    ):
        muestra = punto_muestra(inicio, fin)
        total = sp.Integer(0)

        for fx, a, b in x:
            for fk, c, d in k:

                # Interseccion:
                # [a,b] con [t-d,t-c]

                opciones_inf = [
                    (a, float(a)),
                    (t-d, muestra-float(d))
                ]

                opciones_sup = [
                    (b, float(b)),
                    (t-c, muestra-float(c))
                ]

                inferior, inferior_num = max(
                    opciones_inf,
                    key=lambda item: item[1]
                )

                superior, superior_num = min(
                    opciones_sup,
                    key=lambda item: item[1]
                )

                if inferior_num >= superior_num:
                    continue

                producto = (
                    fx.subs(t, u)
                    * fk.subs(t, t-u)
                )

                integral = sp.integrate(
                    producto,
                    (u, inferior, superior),
                    conds="none"
                )

                # No presentar como solucion cerrada
                # una integral no resuelta.
                if integral.has(sp.Integral):
                    return None, (
                        "SymPy no logró resolver "
                        "analíticamente todas las "
                        "integrales de esta convolución."
                    )

                total += integral

        total = sp.simplify(total)

        if total != 0:
            tramos.append((total, inicio, fin))

    tramos = combinar_iguales(tramos)

    return tramos, None


# ==============================================
# DERIVACION E INTEGRACION DE UNA SOLA SEÑAL
# ==============================================

def valor_finito(expr, punto, direccion):
    valor = sp.simplify(sp.limit(expr, t, punto, dir=direccion))
    if valor.is_finite is not True or valor.is_real is not True:
        raise ValueError(
            f"No existe un límite lateral real y finito en t = {punto}. "
            "Esta operación no admite esa singularidad."
        )
    return valor


def validar_intervalo(expr, a, b):
    # Evitar presentar resultados que atraviesen polos o zonas complejas.
    from sympy.calculus.util import continuous_domain
    try:
        dominio = continuous_domain(expr, t, sp.S.Reals)
        valido = sp.Interval.open(a, b).is_subset(dominio)
    except (NotImplementedError, ValueError):
        raise ValueError(
            "No se pudo verificar el dominio de esta expresión. "
            "Divide la señal en tramos continuos más simples."
        )
    if valido is not True:
        raise ValueError(
            "La expresión tiene una discontinuidad interna, una singularidad "
            "o una zona no real. Divide los tramos en esos puntos."
        )


def derivar_senal(tramos):
    regulares = []
    for expr, a, b in tramos:
        validar_intervalo(expr, a, b)
        derivada = sp.simplify(sp.diff(expr, t))
        if derivada.has(sp.Derivative, sp.DiracDelta):
            raise ValueError(
                "Divide esta expresión en tramos suaves antes de derivarla."
            )
        if derivada != 0:
            regulares.append((derivada, a, b))

    # Calcular los saltos incluyendo la señal cero fuera del soporte.
    limites = sorted({v for _, a, b in tramos for v in (a, b)
                      if v.is_finite}, key=float)
    impulsos = []
    for punto in limites:
        izquierda = sp.Integer(0)
        derecha = sp.Integer(0)
        for expr, a, b in tramos:
            if a < punto <= b:
                izquierda = valor_finito(expr, punto, "-")
            if a <= punto < b:
                derecha = valor_finito(expr, punto, "+")
        salto = sp.simplify(derecha - izquierda)
        if salto != 0:
            impulsos.append((punto, salto))
    return combinar_iguales(regulares), impulsos


def integrar_senal(tramos):
    # Dividir también en cero: de ahí se propagan las constantes hacia
    # ambos lados, incluidos los huecos y las colas constantes.
    limites = {-sp.oo, sp.oo, sp.Integer(0)}
    for expr, a, b in tramos:
        validar_intervalo(expr, a, b)
        limites.update((a, b))
    limites = sorted(limites, key=float)
    celdas = []
    for a, b in zip(limites[:-1], limites[1:]):
        expr = expresion_en(tramos, punto_muestra(a, b))
        primitiva = sp.integrate(expr, t)
        if primitiva.has(sp.Integral):
            raise ValueError(
                "No se obtuvo una primitiva cerrada para todos los tramos. "
                "No se mostrará una fórmula incompleta."
            )
        # En un intervalo real de continuidad, la parte real conserva
        # la primitiva y elimina constantes imaginarias de logaritmos.
        primitiva = sp.simplify(sp.re(primitiva))
        celdas.append((a, b, primitiva))

    resultados = []
    acumulado = sp.Integer(0)
    for a, b, primitiva in celdas:
        if a < 0:
            continue
        referencia = valor_finito(primitiva, a, "+")
        expr = sp.simplify(acumulado + primitiva - referencia)
        resultados.append((expr, a, b))
        if b != sp.oo:
            acumulado = valor_finito(expr, b, "-")

    acumulado = sp.Integer(0)
    for a, b, primitiva in reversed(celdas):
        if b > 0:
            continue
        referencia = valor_finito(primitiva, b, "-")
        expr = sp.simplify(acumulado + primitiva - referencia)
        resultados.append((expr, a, b))
        if a != -sp.oo:
            acumulado = valor_finito(expr, a, "+")

    resultados.sort(key=lambda tramo: float(tramo[1]))
    return [tramo for tramo in combinar_iguales(resultados) if tramo[0] != 0]


def mostrar_operacion_individual(tramos, nombre, operacion, minimo, maximo):
    if operacion == "Derivar":
        resultado, impulsos = derivar_senal(tramos)
        st.latex(rf"d(t)=\frac{{d}}{{dt}}{nombre}(t)")
        st.markdown("#### Resultado analítico")
        st.latex(latex_tramos(r"d_{\mathrm{reg}}", resultado)
                 if resultado else r"d_{\mathrm{reg}}(t)=0")
        distribucion = sum((peso * sp.DiracDelta(t - punto)
                            for punto, peso in impulsos), sp.Integer(0))
        st.latex(r"d(t)=d_{\mathrm{reg}}(t)" +
                 ("+" + sp.latex(distribucion) if impulsos else ""))
        st.caption(
            "La parte regular describe la derivada dentro de los tramos. "
            "En sus límites, la derivada clásica puede no existir. "
            "La fórmula completa se interpreta en sentido de distribuciones."
        )
        titulo = f"Derivada d(t) de {nombre}(t) · parte regular"
        color = "darkorange"
        cortes = fronteras(tramos)
    else:
        resultado = integrar_senal(tramos)
        impulsos = []
        st.latex(rf"i(t)=\int_0^t {nombre}(\tau)\,d\tau,\qquad i(0)=0")
        st.markdown("#### Resultado analítico")
        st.latex(latex_tramos("i", resultado) if resultado else r"i(t)=0")
        st.caption(
            "Integral acumulada desde cero. Las constantes se ajustan para "
            "mantener la continuidad entre tramos. Para t < 0 se invierte "
            "el sentido de integración."
        )
        titulo = f"Integral acumulada i(t) de {nombre}(t)"
        color = "teal"
        cortes = fronteras(resultado)

    fig_ind, ax_ind = plt.subplots(figsize=(11, 4))
    try:
        dibujar(ax_ind, lambda z: evaluar(resultado, z), cortes,
                minimo, maximo, color, titulo, unir_saltos=False)
        if operacion == "Derivar":
            # No asignar una altura finita a las deltas: usar un panel
            # independiente cuyo eje representa el peso del impulso.
            for punto in fronteras(tramos):
                if minimo <= punto <= maximo:
                    ax_ind.axvline(punto, color="gray", alpha=0.2, lw=0.8)
        preparar_figura(fig_ind)
        st.pyplot(fig_ind)
    finally:
        plt.close(fig_ind)

    if impulsos:
        st.markdown("#### Impulsos debidos a los saltos")
        st.caption("Las flechas indican el peso de cada impulso, no una amplitud finita.")
        st.table([{"Posición": str(punto), "Peso": str(peso)}
                  for punto, peso in impulsos])
        visibles = [(float(punto), float(peso)) for punto, peso in impulsos
                    if minimo <= float(punto) <= maximo]
        if visibles:
            fig_imp, ax_imp = plt.subplots(figsize=(11, 3))
            try:
                mayor = max(abs(peso) for _, peso in visibles)
                ax_imp.set_ylim(-1.45 * mayor, 1.45 * mayor)
                ax_imp.axhline(0, color="gray", lw=0.7)
                for punto, peso in visibles:
                    ax_imp.annotate("", xy=(punto, peso), xytext=(punto, 0),
                                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
                    ax_imp.annotate(f"{peso:g}", (punto, peso),
                                    xytext=(4, 7 if peso > 0 else -16),
                                    textcoords="offset points")
                configurar_eje_tiempo(ax_imp, [], minimo, maximo)
                ax_imp.set_title(f"Impulsos de d(t), al derivar {nombre}(t)")
                ax_imp.set_xlabel("Tiempo t")
                ax_imp.set_ylabel("Peso del impulso")
                ax_imp.grid(alpha=0.25)
                preparar_figura(fig_imp)
                st.pyplot(fig_imp)
            finally:
                plt.close(fig_imp)
        else:
            st.info("Los impulsos están fuera del intervalo de visualización.")


# ==============================================
# INTERFAZ: SEÑALES
# ==============================================


with st.expander("📖 Guía rápida · Cómo ingresar las señales", expanded=True):
    st.markdown("""
Escribe solo la expresión en función de **t**. Coloca sus límites en **Desde** y **Hasta**.

| Para ingresar | Escribe | Evita |
|---|---|---|
| Constante o rampa | `2` * `t+1` | `x(t)=2` |
| Multiplicación | `2*t` | `2t` |
| Potencia | `t**2` o `t^2` | `t²` |
| Exponencial | `exp(-t)` | `e-t` |
| Seno y coseno | `sin(pi*t)` · `cos(t)` | `sen(t)` |
| Raíz y valor absoluto | `sqrt(t)` · `Abs(t)` | `√t` · `|t|` |
| Logaritmo natural | `log(t)` | `ln(t)` |
| Decimales y fracciones | `3.5` · `1/2` | `3,5` |
| Límites infinitos | `-oo` · `oo` | `infinito` |

- Cada tramo cumple **Desde ≤ t < Hasta**. Desde debe ser menor que Hasta.
- Los tramos de una misma señal no deben superponerse. Fuera de ellos, la señal vale cero.
- Usa solo la variable `t`. Seno y coseno usan radianes.
- La expresión debe ser real y estar definida dentro del intervalo.
- Para representar saltos, crea tramos separados.
- La convolución analítica automática requiere límites finitos. Con límites infinitos,
  el cálculo numérico depende de que la integral converja.
""")
    st.info("Ejemplo: para x(t) = 2 entre 0 y 2, escribe Expresión: 2, Desde: 0, Hasta: 2.")

col1, col2 = st.columns(2)

with col1:
    x_tramos = editor("x")

with col2:
    k_tramos = editor("k")

if x_tramos is None or k_tramos is None:
    st.stop()

st.divider()
st.header("Expresiones analíticas")

c1, c2 = st.columns(2)

with c1:
    st.latex(latex_tramos("x", x_tramos))

with c2:
    st.latex(latex_tramos("k", k_tramos))

# ==============================================
# INTERFAZ: GRAFICAS
# ==============================================

st.divider()
st.header("Representación gráfica")

minimo, maximo = st.slider(
    "Intervalo de visualización",
    -30, 30, (-5, 8)
)

if minimo >= maximo:
    st.warning("Selecciona una ventana con inicio menor que el final.")
    st.stop()

fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 6),
    constrained_layout=False
)

dibujar(
    axes[0],
    lambda z: evaluar(x_tramos, z),
    fronteras(x_tramos),
    minimo, maximo,
    "royalblue",
    "Señal x(t)"
)

dibujar(
    axes[1],
    lambda z: evaluar(k_tramos, z),
    fronteras(k_tramos),
    minimo, maximo,
    "seagreen",
    "Señal k(t)"
)

preparar_figura(fig)
st.pyplot(fig)
plt.close(fig)

# ==============================================
# INTERFAZ: OPERACIONES
# ==============================================

st.divider()
st.header("Operaciones entre señales")

operacion = st.selectbox(
    "Selecciona una operación",
    ["Suma", "Resta", "Multiplicación"]
)

tramos_resultado = operacion_analitica(
    x_tramos,
    k_tramos,
    operacion
)

letra = {"Suma": "s", "Resta": "r", "Multiplicación": "p"}[operacion]
simbolo = {"Suma": "+", "Resta": "-", "Multiplicación": r"\cdot"}[operacion]
st.latex(rf"{letra}(t)=x(t){simbolo}k(t)")
st.markdown("#### Resultado analítico")

st.latex(
    latex_tramos(letra, tramos_resultado)
    if tramos_resultado
    else rf"{letra}(t)=0"
)

st.markdown("#### Representación gráfica")

fig2, ax2 = plt.subplots(figsize=(11, 4))

dibujar(
    ax2,
    lambda z: evaluar(tramos_resultado, z),
    fronteras(x_tramos + k_tramos),
    minimo, maximo,
    "purple",
    f"{operacion}: {letra}(t)"
)

preparar_figura(fig2)
st.pyplot(fig2)
plt.close(fig2)


# ==============================================
# INTERFAZ: OPERACIONES SOBRE UNA SOLA SEÑAL
# ==============================================

st.divider()
st.header("Operaciones sobre una señal")
sel1, sel2 = st.columns(2)
with sel1:
    nombre_elegido = st.selectbox("Señal que deseas utilizar", ["x", "k"],
                                 format_func=lambda nombre: f"{nombre}(t)")
with sel2:
    operacion_individual = st.selectbox("Operación sobre la señal", ["Derivar", "Integrar"])
st.caption("Se utiliza únicamente la señal seleccionada.")
if st.button("Calcular " + ("derivada" if operacion_individual == "Derivar" else "integral"),
             key="calcular_individual"):
    try:
        with st.spinner("Calculando el resultado y su gráfica..."):
            mostrar_operacion_individual(
                x_tramos if nombre_elegido == "x" else k_tramos,
                nombre_elegido, operacion_individual, minimo, maximo
            )
    except Exception as error:
        st.error("No se pudo completar esta operación: " + str(error))

# ==============================================
# INTERFAZ: CONVOLUCION
# ==============================================

st.divider()
st.header("Convolución")

st.latex(
    r"c(t)=x(t)*k(t)="
    r"\int_{-\infty}^{+\infty}"
    r"x(\tau)k(t-\tau)\,d\tau"
)

if st.button(
    "Calcular convolución",
    type="primary"
):

    st.markdown("### Resultado numérico")

    try:
        with st.spinner(
            "Calculando integral de convolución..."
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                tiempos, valores = convolucion_numerica(
                    x_tramos, k_tramos, minimo, maximo
                )

        if not np.all(np.isfinite(valores)):
            raise ValueError(
                "La integral produjo valores no finitos."
            )

        fig3, ax3 = plt.subplots(figsize=(11, 4))

        ax3.plot(
            tiempos,
            valores,
            color="darkviolet",
            linewidth=2
        )

        ax3.fill_between(
            tiempos,
            valores,
            color="violet",
            alpha=0.15
        )

        ax3.axhline(0, color="gray", lw=0.7)
        ax3.set_xlabel("Tiempo t")
        ax3.set_ylabel("Amplitud")
        ax3.set_title("Convolución c(t)")
        ax3.grid(alpha=0.3)
        configurar_eje_tiempo(
            ax3, cortes_convolucion(x_tramos, k_tramos), minimo, maximo
        )

        preparar_figura(fig3)
        st.pyplot(fig3)
        plt.close(fig3)

        st.markdown("### Resultado analítico")

        with st.spinner(
            "Resolviendo los intervalos..."
        ):
            tramos_conv, aviso = convolucion_analitica(
                x_tramos,
                k_tramos
            )

        if tramos_conv is not None:
            if tramos_conv:
                st.latex(
                    latex_tramos("c", tramos_conv)
                )
            else:
                st.latex(r"c(t)=0")

            st.success(
                "Se obtuvo la expresión analítica "
                "de la convolución."
            )

        else:
            st.warning(aviso)
            st.info(
                "La gráfica mostrada es numérica; "
                "no se ha obtenido una expresión "
                "analítica cerrada."
            )

    except Exception as error:
        st.error(
            "No se pudo calcular la convolución: "
            + str(error)
        )

st.divider()
st.caption(
    "CONVOLAB Web v0.5 | "
    "Proyecto académico de señales continuas"
)

