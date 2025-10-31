import multiprocessing
from simulacion import correr_simulaciones
from graficos import graficar_resultados

if __name__ == "__main__":
    # Configuraciones de la simulacion
    num_simulaciones = 20   # Numero de simulaciones independientes
    bins_por_simulacion = 10000  # Bines por cada simulacion

    # Configuraciones de barrido
    num_puntos_eficiencia = 10
    num_puntos_dark = 10

    tasa_pares = 0.01   # Promedio de pares por bin
    bin_duration = 1.0  # Duracion de cada bin (en unidades arbitrarias)
    ventana_coincidencia = 0.1  # Ventana de coincidencia (en las mismas unidades de tiempo)

    # Configuracion de uso de CPU y GPU
    num_cpus = multiprocessing.cpu_count()
    usar_gpu = True

    print(f"Iniciando {num_simulaciones} simulaciones...")

    correr_simulaciones(
        num_simulaciones=num_simulaciones,
        bins=bins_por_simulacion,
        tasa_pares=tasa_pares,
        bin_duration=bin_duration,
        ventana=ventana_coincidencia,
        num_puntos_eficiencia=num_puntos_eficiencia,
        num_puntos_dark=num_puntos_dark,
        usar_gpu=usar_gpu,
        num_cpus=num_cpus
    )

    print("Simulaciones completadas. Generando graficos...")

    graficar_resultados("resultados_simulacion", num_simulaciones)

    print("Proceso completo.")