# Proyecto de Estimación de Precios de Automóviles

## Integrantes del Proyecto
- **Matias Luciano Antezana**
- **Mateo Giacometti**

## Organización del Trabajo

Este trabajo está organizado de forma que se puedan especificar los archivos de configuración que se desean ejecutar mediante el script `go.sh`. 

Cada modelo, junto con los mejores parámetros obtenidos, los archivos utilizados del config, las métricas obtenidas y un gráfico de valor real vs estimación, se guarda en una carpeta independiente dentro de la carpeta `results`. Además, se guarda un registro (log) para asegurar que todos los pasos se ejecutaron correctamente.

### Estructura del Proyecto

- `go.sh`: Script principal para ejecutar los archivos de configuración.
- `config/`: Carpeta que contiene los archivos de configuración.
- `results/`: Carpeta donde se almacenan los resultados de cada ejecución.
  - Cada subcarpeta en `results/` incluye:
    - Modelo utilizado
    - Mejores parámetros obtenidos
    - Archivos de configuración utilizados
    - Métricas obtenidas
    - Gráfico de valor real vs estimación
    - Log de la ejecución

### Cómo Ejecutar

1. Coloca los archivos de configuración que desees correr en la carpeta `config`.
2. Ejecuta el script `go.sh` desde la terminal:
   ```bash
   ./go.sh
