# TP4-SIA 2021 1er C.

Implementaciones de Kohonen, Oja y Hopfield
```
Integrantes:
- Eugenia Sol Piñeiro
- Scott Lin
- Nicolás Comerci Wolcanyik
```

### Requerimientos previos
- Python 3


### Instalación

Ejecutar el siguiente comando en la terminal para instalar los módulos de python requeridos
```bash
$> pip install -r requirements.txt
```

### Guía de uso
En el archivo `config.json` del directorio raíz se encuentran los distintos parámetros para configurar la ejecución. Estos son:

- `file_path`: Directorio del archivo 'europe.csv'

- `oja`: Parámetros para los ejercicios de Oja
  - `epochs_amount`: Cantidad de épocas
  - `learning_rate`: Tasa de aprendizaje


- `kohonen`: Parámetros para los ejercicios de Kohonen
  - `epochs_amount`: Cantidad de épocas
  - `learning_rate`: Tasa de aprendizaje
  - `grid_dimension`: Dimensión de la matriz de neuronas
  - `radius_constant`: Booleano para la utilización de radio constante
  - `radius_value`: Valor inicial del radio
  - `use_input_as_weights`: Booleano para usar los datos de entrenamiento como pesos iniciales
- `hopfield`: Parámetros para los ejercicios de Hopfield
  - `max_iterations`: Cantidad máxima de iteraciones
  - `noise_probability`: Probabilidad de ruido
  - `conserve_pattern`: Booleano para agregar ruido conservando la silueta del patrón elegido
  - `pattern_to_add_noise`: Patrón a utilizar (Ej: "N")
  - `pattern_to_store`: Vector con patrones a almacenar

Para ejecutar el programa, correr el siguiente comando en consola:
```bash
$> python3 main.py
```

### Ejemplo de configuración

`config.json`:
```json
{
    "ej1": {
        "file_path": "files/europe.csv",
        "oja":{
            "epochs_amount": 1000, 
            "learning_rate": 0.01
        },
        "kohonen": {
            "epochs_amount": 100, 
            "learning_rate": 0.01,
            "grid_dimension": 4, 
            "radius_constant": false,
            "radius_value": 10,
            "use_input_as_weights": true
        }
    },
    "ej2":{
        "hopfield": {
            "max_iterations": 20,
            "noise_probability": 0.2,
            "conserve_pattern": false, 
            "pattern_to_add_noise": "N",
            "pattern_to_store": ["J","E","K","N"]
        }
    }
}
```