# ¡Limpia el Espejo! - Juego Serio de Rehabilitación

## Descripción

**¡Limpia el Espejo!** es un juego serio diseñado específicamente para la rehabilitación física de las extremidades superiores, con énfasis en el hombro. El juego se centra en mejorar tres aspectos fundamentales:

- **Rango de Movimiento (ROM)**: Amplitud del movimiento articular
- **Coordinación**: Control y precisión del movimiento
- **Fuerza Isométrica**: Mantenimiento de posiciones estáticas

La característica principal del juego es su sistema de detección de movimiento basado en **visión por computador**, que utiliza la webcam para rastrear el movimiento de la mano del usuario (específicamente el nudillo del dedo corazón) sin necesidad de sensores físicos adicionales, haciendo que la rehabilitación sea accesible y no invasiva.

---

## Requisitos Previos

Antes de comenzar, asegúrate de contar con lo siguiente:

- **Ordenador** compatible (Windows, Mac o Linux)
- **Webcam funcional** (integrada o externa) y libre para utilizar (sin conexión activa a apps como Google Meet o Skype)
- **Python 3.8 o superior** instalado en el sistema
- **Buena iluminación** en el área de juego para una detección óptima

---

## Instalación y Configuración

### Paso 1: Preparar los archivos

Asegúrate de tener los siguientes archivos en una carpeta del proyecto:
- `app.py` (script principal del juego)
- `requirements.txt` (lista de dependencias)

### Paso 2 (Opcional): Crear un entorno virtual

Se recomienda usar un entorno para aislar las dependencias del proyecto:

- Opción 1: Entorno virtual
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate
```

- Opción 2: Entorno Conda
```bash
conda create -n IPM python=3.12.12
conda activate IPM
```

### Paso 3: Instalar dependencias

Ejecuta el siguiente comando para instalar todas las bibliotecas necesarias:

```bash
pip install -r requirements.txt
```

---

## Cómo Iniciar

Para ejecutar el juego, utiliza el siguiente comando:

```bash
python app.py
```

---

## Guía de Juego

### Control del Cursor

El cursor del juego está controlado por el **nudillo base del dedo corazón** de tu mano, detectado mediante la webcam. Asegúrate de mantener tu mano visible y bien iluminada durante el juego.

### Fases del Juego

#### 1. Calibración
- Al inicio, aparecerán **4 círculos rojos** en las esquinas de la pantalla
- Toca cada círculo con el cursor (tu nudillo) para calibrar el área de juego
- Los círculos se volverán **verdes** al ser tocados exitosamente
- Esta fase establece tu rango de movimiento personal

#### 2. Nivel 1 - Limpieza General
- Objetivo: Limpiar la suciedad marrón del espejo (exitoso a partir del 90%)
- Mecánica: Realiza movimientos libres tipo **"limpieza de espejo"** con tu mano
- Trabaja el rango de movimiento completo del hombro
- No hay límite de tiempo, explora toda el área

#### 3. Nivel 2 - Precisión y Resistencia
Este nivel presenta dos tipos de manchas:

- **Manchas Claras**: Requieren una pasada rápida para limpiarlas (coordinación)
- **Manchas Oscuras**: Debes mantener tu mano quieta sobre ellas durante **1.5 segundos** (fuerza isométrica y estabilidad)

Este nivel combina movimientos dinámicos con trabajo estático del hombro.

### Controles

- **ENTER**: Avanzar a la siguiente fase o reiniciar el juego
- **ESC**: Salir del juego en cualquier momento
