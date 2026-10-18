# Chrome Dino RL — Tabular Reinforcement Learning Environment

Implementación completa del popular juego **Chrome Dino** como un entorno compatible con la API estándar de **Gymnasium**, adaptado para la experimentación e investigación con **algoritmos de Aprendizaje por Refuerzo Tabular** (Q-Learning, SARSA y Expected SARSA).

---

## Índice

- [Estructura del Proyecto](#estructura-del-proyecto)
- [El Entorno: `DinoEnv`](#el-entorno-dinoenv)
  - [Espacio de Acciones](#espacio-de-acciones)
  - [Espacio de Observaciones y Discretización](#espacio-de-observaciones-y-discretización)
  - [Física del Dinosaurio y Obstáculos](#física-del-dinosaurio-y-obstáculos)
  - [Función de Recompensas](#función-de-recompensas)
  - [Renderizadores](#renderizadores)
- [Algoritmos Tabulares (`deustorl`)](#algoritmos-tabulares-deustorl)
- [Guía de Scripts y Ejemplos (`examples/`)](#guía-de-scripts-y-ejemplos-examples)
  - [1. Jugar en Modo Manual: `play_human.py`](#1-jugar-en-modo-manual-play_humanpy)
  - [2. Agente Aleatorio: `random_agent.py`](#2-agente-aleatorio-random_agentpy)
  - [3. Entrenamiento Individual Q-Learning: `train_qlearning.py`](#3-entrenamiento-individual-q-learning-train_qlearningpy)
  - [4. Experimento Completo y Optimización Optuna: `full_experiment.py`](#4-experimento-completo-y-optimización-optuna-full_experimentpy)
- [Instalación y Requisitos](#instalación-y-requisitos)

---

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

---

## El Entorno: `DinoEnv`

El entorno simula la cinemática en tiempo continuo del juego Chrome Dino mediante una discretización en frames con frecuencia de actualización a 30 FPS.

### Espacio de Acciones

El espacio de acciones es discreto (`gymnasium.spaces.Discrete(3)`):

| Acción | Nombre | Descripción |
| :---: | :---: | :--- |
| `0` | **`NOOP`** | No hacer nada. El dinosaurio corre erguido a su altura normal ($50\text{ px}$). |
| `1` | **`JUMP`** | Salta con velocidad inicial vertical de $14.0\text{ px/frame}$. No permite doble salto. |
| `2` | **`DUCK`** | Se agacha reduciendo su altura a $25\text{ px}$ para esquivar obstáculos aéreos. |

---

### Espacio de Observaciones y Discretización

El entorno continuo calcula 7 variables cinemáticas internas que son mapeadas a una tupla discreta mediante `_discretize_obs()` y posteriormente indexadas con `np.ravel_multi_index`:

$$\text{Observación} = (\text{dino\_y}, \text{dino\_vel\_y}, \text{obs1\_dist\_x}, \text{obs1\_y\_min}, \text{obs1\_width}, \text{obs2\_gap}, \text{game\_speed})$$

| Variable | Bins | Descripción e Intervalos |
| :--- | :---: | :--- |
| **`dino_y`** | **3** | Posición vertical: En el suelo ($= 0$), altura media ($< 60$), altura máxima ($\ge 60$). |
| **`dino_vel_y`** | **3** | Velocidad vertical: Cayendo ($< -0.5$), estable ($[-0.5, 0.5]$), subiendo ($> 0.5$). |
| **`obs1_dist_x`** | **10** | **Zona crítica de aproximación al primer obstáculo:**<br>• $1$ bin para ya superado ($< 0\text{ px}$)<br>• **$8$ bins finos de $20\text{ px}$** entre $[0, 160\text{ px})$<br>• $1$ bin para zona segura / lejana ($\ge 160\text{ px}$) |
| **`obs1_y_min`** | **3** | Altura del obstáculo: Suelo/cactus ($< 15$), pájaro a media altura ($< 55$), pájaro alto ($\ge 55$). |
| **`obs1_width`** | **3** | Ancho del obstáculo: Estrecho ($< 35$), mediano ($< 60$), ancho ($\ge 60$). |
| **`obs2_gap`** | **2** | Distancia al segundo obstáculo: Próximo ($< 250\text{ px}$), lejano ($\ge 250\text{ px}$). |
| **`game_speed`** | **1 o 3** | Velocidad horizontal del juego: Con `constant_speed=True` usa **1 bin**; con `constant_speed=False` usa **3 bins** ($< 9.5$, $< 12.5$, $\ge 12.5$). |

#### Tamaño total de la Tabla Q:
* **Modo Velocidad Constante (`constant_speed=True`):**
  $$3 \times 3 \times 10 \times 3 \times 3 \times 2 \times 1 = \mathbf{1.620\text{ estados}} \quad (4.860\text{ valores Q})$$
* **Modo Velocidad Variable (`constant_speed=False`):**
  $$3 \times 3 \times 10 \times 3 \times 3 \times 2 \times 3 = \mathbf{4.860\text{ estados}} \quad (14.580\text{ valores Q})$$

---

### Física del Dinosaurio y Obstáculos

* **Parábola de Salto:** Velocidad vertical inicial $v_0 = 14.0\text{ px/frame}$, gravedad $g = 0.8\text{ px/frame}^2$. La trayectoria dura exactamente **35 frames**, alcanzando una altura máxima de $122.5\text{ px}$.
* **Control de Salto Seguro:** Se desactiva el multiplicador de *fast-fall* accidental en el aire durante la exploración (`enable_fast_fall=False`), permitiendo que el salto complete su arco balístico sin interrupciones involuntarias.
* **Spawneo de Obstáculos:**
  * **Primer obstáculo:** Aparece a una distancia inmediata de $\approx 200\text{ px}$ por delante del dinosaurio tras el `reset()`, eliminando los 80 pasos muertos iniciales y triplicando la densidad útil de entrenamiento.
  * **Tipos:** $70\%$ de probabilidad de cactus en el suelo (anchuras de 1, 2 o 3 bloques) y $30\%$ de probabilidad de aves voladoras a tres alturas distintas ($0.0$, $35.0$, $75.0\text{ px}$).

---

### Función de Recompensas

* **Supervivencia:** **`+1.0`** ([`STEP_REWARD`](file:///home/rl/assignments/1_assignment/tabular-methods-experimentation/dino_game/dino_env.py#L38)) por cada frame sobrevivido sin colisionar.
* **Colisión:** **`-100.0`** ([`DEATH_PENALTY`](file:///home/rl/assignments/1_assignment/tabular-methods-experimentation/dino_game/dino_env.py#L39)) al entrar en contacto con cualquier obstáculo (AABB collision box).

---

### Renderizadores

* **PyGame (`render_mode="human"`):** Ventana gráfica a 30 FPS con renderizado de texturas poligonales, ojos reactivos del dinosaurio, separación visual de grupos de cactus, pico de aves, HUD con marcador de puntuación y velocidad, y alternancia automática de **ciclo día/noche** cada 700 pasos.
* **ANSI (`render_mode="ansi"`):** Renderizado ligero en consola basado en una matriz de caracteres de $80 \times 10$ (`#` para cactus, `V` para aves y `d`/`D` para el dinosaurio).

---

## Algoritmos Tabulares (`deustorl`)

El proyecto utiliza implementaciones adaptadas de tres métodos fundamentales de Diferencias Temporales (TD):

* **Q-Learning (`qlearning_fixed.py`):** Algoritmo *off-policy* que actualiza el valor mediante la acción codiciosa máxima del siguiente estado:
  $$\text{Target} = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) \cdot (1 - \text{terminated})$$
* **SARSA (`sarsa_fixed.py`):** Algoritmo *on-policy* que actualiza mediante la acción efectivamente seleccionada por la política de exploración en el siguiente paso:
  $$\text{Target} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) \cdot (1 - \text{terminated})$$
* **Expected SARSA (`expected_sarsa_fixed.py`):** Algoritmo que calcula la esperanza matemática ponderada sobre todas las acciones posibles según la distribución de la política en el siguiente estado:
  $$\text{Target} = R_{t+1} + \gamma \sum_{a} \pi(a \mid S_{t+1}) Q(S_{t+1}, a) \cdot (1 - \text{terminated})$$

> **Nota sobre los módulos `_fixed`:** Todos ellos incorporan la corrección del valor futuro nulo en estados absorbentes/terminales (`(1 - terminated)`), garantizando que la penalización de muerte (`-100`) no se diluya con valores del estado posterior al choque.

---

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
Entrena un agente de Q-Learning durante 2 millones de pasos con velocidad constante y luego evalúa la política aprendida mostrando el juego en PyGame:

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
Los resultados se almacenan en una base de datos SQLite común (`optuna/optuna.db`) y exportan para cada estudio:
* `best_trial.json`: Hiperparámetros óptimos y recompensas alcanzadas.
* `optimization_history.html`: Evolución de la recompensa a lo largo de los trials.
* `contour.html`: Relaciones bidimensionales entre hiperparámetros.
* `slice.html`: Gráfico de corte de la función objetivo por parámetro.
* `param_importances.html`: Gráfico de importancia e influencia de cada hiperparámetro.
* `comparison_summary.json`: Resumen comparativo directo entre ambos entornos.

---

## Instalación y Requisitos

Se recomienda utilizar un entorno virtual (Conda o Python `venv` con Python 3.10 o superior):

```bash
pip install gymnasium pygame optuna torch numpy matplotlib plotly
```