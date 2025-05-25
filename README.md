# ADV-DQN-SNAKE 🐍

**Agente RL avanzado jugando Snake en PyGame + Gym + PyTorch, con DQN, stacking y tracking MLflow**

---

## 🚀 Descripción

Este proyecto implementa un entorno Snake en PyGame, compatible con OpenAI Gym, y agentes de *Reinforcement Learning* (RL) utilizando DQN y variantes.  
Incluye herramientas para análisis experimental (MLflow), render visual rápido, y estructura lista para escalar a técnicas más avanzadas como Dueling DQN, Double DQN y experiencia priorizada.

---

## 🗂️ Estructura de carpetas

snake_rl/
env/ # Entorno Gym (SnakeEnv)
game/ # Lógica Snake (SnakeGameEnv, Snake, Food)
agents/ # DQN, random agent, utilidades
requirements.txt
.gitignore
README.md
---

## ⚡ Requisitos

- Python 3.11+
- Conda o venv (recomendado)
- PyTorch
- Gym
- pygame
- numpy
- mlflow

Instala dependencias:

pip install -r requirements.txt
🎮 Ejecución rápida
1. Entrenamiento DQN + visualización + tracking
Desde la raíz del proyecto:

Render visual de Snake en vivo (ajustable con FPS).
Logging de reward, epsilon y modelo con MLflow.

2. Seguimiento experimental con MLflow
En otra terminal, lanza el UI de MLflow:
Visualiza la curva de reward, los hiperparámetros y descarga modelos entrenados.

🧠 ¿Qué incluye este repo?
Snake Gym Env: 100% compatible, modular, reproducible.

DQN minimal funcional: replay buffer, epsilon-greedy, target network, MLflow.
Stacking de estados: memoria temporal para mejor aprendizaje.
Render PyGame acelerado: control de FPS, visualización en tiempo real.
Tracking profesional: métrica por episodio, historial de experimentos y modelos.

Listo para:

Ajuste de reward/exploración
Dueling/Double DQN
Replay buffer priorizado
Experimentación incremental

📊 Resultados esperados
DQN minimal + stacking: ~30–40 puntos de media tras 1500 episodios (benchmark público).

Tuning avanzado y mejoras de arquitectura.

🔬 Siguientes pasos propuestos
Reward shaping y exploración avanzada.
Dueling DQN, Double DQN, experiencia priorizada (PER).
Grid visual (CNN), curiosity, meta-RL.

