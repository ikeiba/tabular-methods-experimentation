# Chrome Dino RL — Tabular Reinforcement Learning Environment

Implementación completa del popular juego **Chrome Dino** como un entorno compatible con la API estándar de **Gymnasium**, adaptado para la experimentación e investigación con **algoritmos de Aprendizaje por Refuerzo Tabular** (Q-Learning, SARSA y Expected SARSA).

## Estructura del Proyecto

```text
.
├── dino_game/
│   ├── __init__.py
│   ├── dino_env.py             # Definición del entorno Gymnasium DinoEnv
│   └── renderers/
│       ├── __init__.py
│       ├── pygame_renderer.py  # Renderizador gráfico en PyGame (humano)
│       └── ansi_renderer.py    # Renderizador de texto ASCII (terminal)
│
├── deustorl/
│   ├── common.py               # Tablas Q, políticas (EpsilonGreedy, Max), evaluación
│   ├── helpers.py              # TensorboardLogger y wrappers de discretización
│   ├── qlearning_fixed.py      # Q-Learning con corrección de Bellman en terminales
│   ├── sarsa_fixed.py          # SARSA con corrección de Bellman en terminales
│   └── expected_sarsa_fixed.py # Expected SARSA con corrección de Bellman en terminales
│
├── examples/
│   ├── play_human.py           # Control manual con teclado
│   ├── random_agent.py         # Demostración con acciones aleatorias
│   ├── train_qlearning.py      # Entrenamiento y evaluación de Q-Learning
│   └── full_experiment.py      # Optimización bayesiana con Optuna y comparativa de entornos
│
└── optuna/                     # Bases de datos SQLite y reportes HTML generados por Optuna
```

## Guía de Scripts y Ejemplos (`examples/`)

### 1. Jugar en Modo Manual: `play_human.py`
Permite probar la física y comprobar la jugabilidad en tiempo real con el teclado:

```bash
python examples/play_human.py
```
* **Controles:**
  * **Espacio** o **Flecha Arriba:** Saltar.
  * **Flecha Abajo:** Agacharse.
  * **Sin pulsar:** Correr erguido normal.

---

### 2. Agente Aleatorio: `random_agent.py`
Ejecuta un agente que selecciona acciones aleatorias uniformes (`action_space.sample()`) para verificar el renderizado:

```bash
# Modo gráfico con PyGame
python examples/random_agent.py --human

# Modo texto ANSI en la terminal
python examples/random_agent.py
```

---

### 3. Entrenamiento Individual Q-Learning: `train_qlearning.py`
Entrena un agente de Q-Learning durante 2 millones de pasos con velocidad incremental por defecto y luego evalúa la política aprendida mostrando el juego en PyGame (con el flag `--human`) o en la consola ANSI:

```bash
# Entrenar y visualizar en ventana gráfica
python examples/train_qlearning.py --human

# Entrenar y evaluar en consola ANSI
python examples/train_qlearning.py
```

---

### 4. Experimento Completo y Optimización Optuna: `full_experiment.py`
Script integral de experimentación basado en Optuna (con muestreador TPE). Permite comparar **SARSA**, **Expected SARSA** y **Q-Learning**, contrastando dos entornos con distinta dificultad:

1. **Estudio `dino_constant_speed`:** Entorno estacionario fácil ($1.620$ estados).
2. **Estudio `dino_variable_speed`:** Entorno dinámico difícil con aceleración ($4.860$ estados).

#### Comandos de ejecución:

* **Ejecutar ambos estudios y generar la comparativa final:**
  ```bash
  python examples/full_experiment.py --mode both
  ```

* **Ejecutar únicamente el estudio de velocidad constante:**
  ```bash
  python examples/full_experiment.py --mode constant
  ```

* **Ejecutar únicamente el estudio de velocidad variable:**
  ```bash
  python examples/full_experiment.py --mode variable
  ```

* **Ajustar el número de trials y pasos de entrenamiento:**
  ```bash
  python examples/full_experiment.py --mode both --n-trials 20 --n-steps 500000
  ```

#### Archivos y visualizaciones generadas por Optuna:
Los resultados se almacenan en una base de datos SQLite común (`optuna/optuna.db`). Para cada estudio ejecutado (`optuna/<nombre_estudio>/`) se exportan:
* `best_trial.json`: Hiperparámetros óptimos y recompensas alcanzadas.
* `optimization_history.html`: Evolución de la recompensa a lo largo de los trials.
* `contour.html`: Relaciones bidimensionales entre hiperparámetros.
* `slice.html`: Gráfico de corte de la función objetivo por parámetro.
* `param_importances.html`: Gráfico de importancia e influencia de cada hiperparámetro.

Si se ejecutan ambos estudios (`--mode both`), se genera además en la raíz:
* `optuna/comparison_summary.json`: Resumen comparativo directo entre ambos entornos.

---

## Instalación y Requisitos

El entorno virtual utilizado es el de conda que hemos utilizado en clase.