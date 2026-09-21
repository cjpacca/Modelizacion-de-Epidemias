# ==========================================
# Configuración Inicial y Parámetros
# ==========================================

# Parámetros poblacionales y condiciones iniciales
N = 30000000       # Población total de 3x10^7
I0 = 100           # Infectados iniciales
E0 = 0             # Expuestos iniciales
R0 = 0             # Recuperados iniciales
S0 = N - I0        # Susceptibles iniciales[cite: 1]

# Estado inicial agrupado en una tupla
y0 = (S0, E0, I0, R0)

# Parámetros biológicos base (Escenario A - Libre)
beta_A = 0.6       # Probabilidad de transmisión por contacto
sigma = 0.2        # Inverso del periodo medio de incubación
gamma = 0.1        # Inverso del periodo medio de infección

# Variables de simulación temporal (las usaremos en las siguientes fases)
t_inicial = 0
t_final = 150      # Días de simulación (ajustable)
h = 1.0            # Tamaño del paso temporal 

# ==========================================
# Definición del Sistema SEIR
# ==========================================

def sistema_seir(t, estado, beta, sigma, gamma, N):
    """
    Función que calcula las derivadas del modelo matemático SEIR en un instante t.
    """
    S, E, I, R = estado
    
    # Ecuaciones Diferenciales Ordinarias (EDO)
    
    # 1. Tasa de contagio por contacto
    dS_dt = -(beta * S * I) / N
    
    # 2. Transición de susceptible a expuesto e incubación
    dE_dt = ((beta * S * I) / N) - (sigma * E)
    
    # 3. Aparición de síntomas/contagiosidad y recuperación
    dI_dt = (sigma * E) - (gamma * I)
    
    # 4. Inmunidad o salida del sistema
    dR_dt = gamma * I
    
    return dS_dt, dE_dt, dI_dt, dR_dt

# ==========================================
# RK4 Manual
# ==========================================

def rk4_paso(f, t, y, h, beta, sigma, gamma, N):
    """
    Implementación manual de un paso temporal del algoritmo de Runge-Kutta de 4to Orden.
    """
    # k1 = f(t_n, y_n)
    k1 = f(t, y, beta, sigma, gamma, N)
    
    # k2 = f(t_n + h/2, y_n + h/2 * k1)
    y_k2 = tuple(y_i + 0.5 * h * k1_i for y_i, k1_i in zip(y, k1))
    k2 = f(t + 0.5 * h, y_k2, beta, sigma, gamma, N)
    
    # k3 = f(t_n + h/2, y_n + h/2 * k2)
    y_k3 = tuple(y_i + 0.5 * h * k2_i for y_i, k2_i in zip(y, k2))
    k3 = f(t + 0.5 * h, y_k3, beta, sigma, gamma, N)
    
    # k4 = f(t_n + h, y_n + h * k3)
    y_k4 = tuple(y_i + h * k3_i for y_i, k3_i in zip(y, k3))
    k4 = f(t + h, y_k4, beta, sigma, gamma, N)
    
    # y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)
    y_siguiente = tuple(
        y_i + (h / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i) 
        for y_i, k1_i, k2_i, k3_i, k4_i in zip(y, k1, k2, k3, k4)
    )
    
    return y_siguiente


# ==========================================
# Simulaciones
# ==========================================

def ejecutar_simulacion(escenario_intervencion=False):
    """
    Ciclo principal para simular la evolución poblacional a lo largo del tiempo t.
    Permite activar el Escenario B (Intervención) o mantener el Escenario A (Libre).
    """
    # Listas para almacenar el histórico de la evolución temporal
    t_hist = [t_inicial]
    S_hist, E_hist, I_hist, R_hist = [y0[0]], [y0[1]], [y0[2]], [y0[3]]
    
    # Variables de estado actual
    t = t_inicial
    y = y0
    
    while t < t_final:
        # Dinámica del Escenario B: Intervención en t = 30
        if escenario_intervencion and t >= 30:
            beta_actual = 0.25  # El parámetro de contagio se reduce a 0.25
        else:
            beta_actual = beta_A # Se mantiene la probabilidad de transmisión inicial (0.6)
            
        # Calcular el estado en el siguiente paso de tiempo usando RK4
        y = rk4_paso(sistema_seir, t, y, h, beta_actual, sigma, gamma, N)
        t += h
        
        # Registrar los datos
        t_hist.append(t)
        S_hist.append(y[0])
        E_hist.append(y[1])
        I_hist.append(y[2])
        R_hist.append(y[3])
        
    return t_hist, S_hist, E_hist, I_hist, R_hist

# Generación de datos numéricos para ambos escenarios
datos_A = ejecutar_simulacion(escenario_intervencion=False)
datos_B = ejecutar_simulacion(escenario_intervencion=True)

# ==========================================
# Análisis de Conservación
# ==========================================

def verificar_conservacion(S_hist, E_hist, I_hist, R_hist, N, escenario_nombre, tolerancia=1e-5):
    """
    Demuestra numéricamente que la suma poblacional S+E+I+R permanece constante.
    Se usa una pequeña tolerancia para evitar errores de precisión de punto flotante.
    """
    conservado = True
    for i in range(len(S_hist)):
        suma_poblacion = S_hist[i] + E_hist[i] + I_hist[i] + R_hist[i]
        
        # Validamos si la diferencia absoluta con N es mayor a la tolerancia permitida
        if abs(suma_poblacion - N) > tolerancia:
            print(f"[{escenario_nombre}] Error de conservación en el paso {i}: Suma = {suma_poblacion}")
            conservado = False
            break
            
    if conservado:
        print(f"[{escenario_nombre}] Verificacion exitosa: La poblacion total S+E+I+R se conserva en {N}.")

# Ejecutamos la prueba de conservación para los datos generados
verificar_conservacion(datos_A[1], datos_A[2], datos_A[3], datos_A[4], N, "Escenario A")
verificar_conservacion(datos_B[1], datos_B[2], datos_B[3], datos_B[4], N, "Escenario B")


# ==========================================
# Visualización y Análisis de Umbral
# ==========================================
import matplotlib.pyplot as plt

# 1. Gráficas de evolución temporal
def graficar_escenario(t, S, E, I, R, titulo):
    """
    Traza la evolución de los 4 compartimientos poblacionales.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptibles (S)', color='blue')
    plt.plot(t, E, label='Expuestos (E)', color='orange')
    plt.plot(t, I, label='Infectados (I)', color='red')
    plt.plot(t, R, label='Recuperados (R)', color='green')
    
    plt.title(titulo)
    plt.xlabel('Tiempo (Días)')
    plt.ylabel('Población')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Llamadas para graficar
graficar_escenario(datos_A[0], datos_A[1], datos_A[2], datos_A[3], datos_A[4], 'Escenario A (Libre)')
graficar_escenario(datos_B[0], datos_B[1], datos_B[2], datos_B[3], datos_B[4], 'Escenario B (Intervención en t=30)')

# 2. Análisis de Umbral (Número Básico de Reproducción R0)
# Cálculo según la fórmula R0 = beta / gamma
R0_A = beta_A / gamma
R0_B = 0.25 / gamma  # Usando el valor reducido por el confinamiento

print("\n--- ANALISIS DE UMBRAL (R0) ---")
print(f"R0 en Escenario A (Libre): {R0_A}")
print(f"R0 en Escenario B (Confinamiento): {R0_B}")