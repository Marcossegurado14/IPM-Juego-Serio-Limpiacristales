import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURACIÓN INICIAL ---
# Inicializamos MediaPipe Pose (detecta cuerpo entero para ver compensaciones)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configuración de la "suciedad"
RADIO_PINCEL = 40  # Tamaño del borrador
COLOR_SUCIEDAD = (200, 200, 200) # Gris claro (niebla)
OPACIDAD_SUCIEDAD = 0.8

# Inicializamos cámara
cap = cv2.VideoCapture(0)
# Ajustar resolución (opcional, mejora rendimiento)
cap.set(3, 1280)
cap.set(4, 720)

# Variables para la lógica del juego
mask_suciedad = None # Se creará en el primer frame

print("Juego Iniciado. Usa tu MUÑECA DERECHA para limpiar.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Espejo (flip) para que sea intuitivo
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # --- 1. INICIALIZAR LA CAPA DE SUCIEDAD (Solo una vez) ---
    if mask_suciedad is None:
        # Creamos una máscara blanca (255 = sucio, 0 = limpio)
        mask_suciedad = np.ones((h, w), dtype=np.uint8) * 255

    # --- 2. DETECCIÓN CON MEDIAPIPE ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # --- 3. LÓGICA DEL JUEGO (LIMPIEZA) ---
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Obtenemos la muñeca derecha (Índice 16 en MP Pose)
        # Usamos Pose en vez de Hands para poder medir "trampas" con la espalda luego
        muneca = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Comprobamos visibilidad (confidence)
        if muneca.visibility > 0.5:
            # Convertir coordenadas normalizadas (0-1) a píxeles
            cx, cy = int(muneca.x * w), int(muneca.y * h)
            
            # "Borrar" suciedad: Dibujamos un círculo NEGRO (0) en la máscara
            cv2.circle(mask_suciedad, (cx, cy), RADIO_PINCEL, 0, -1)
            
            # Feedback visual: Dibujar el esqueleto para referencia
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Dibujar el "pincel" actual
            cv2.circle(frame, (cx, cy), RADIO_PINCEL, (0, 255, 0), 2)

    # --- 4. COMPOSICIÓN DE IMAGEN (MEZCLAR CAPAS) ---
    # Paso A: Crear la imagen de suciedad real
    # Creamos una imagen gris sólida
    capa_gris = np.zeros_like(frame)
    capa_gris[:] = COLOR_SUCIEDAD
    
    # Paso B: Aplicar la máscara a la capa gris
    # Donde la máscara es negra (0), la capa gris se vuelve negra
    capa_suciedad_final = cv2.bitwise_and(capa_gris, capa_gris, mask=mask_suciedad)
    
    # Paso C: Mezclar con el video original
    # Solo mezclamos donde todavía hay suciedad
    # Esto es un truco visual: Frame + Suciedad
    frame_final = cv2.addWeighted(frame, 1, capa_suciedad_final, OPACIDAD_SUCIEDAD, 0)
    
    # --- 5. CALCULO DE PROGRESO (SERIOUS GAME METRIC) ---
    # Contamos cuántos píxeles quedan sucios (valor 255)
    pixeles_totales = w * h
    pixeles_sucios = cv2.countNonZero(mask_suciedad)
    porcentaje_limpio = 100 - (pixeles_sucios / pixeles_totales * 100)
    
    # Mostrar texto en pantalla
    cv2.putText(frame_final, f'Limpiado: {int(porcentaje_limpio)}%', (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    if porcentaje_limpio > 95:
        cv2.putText(frame_final, '¡NIVEL COMPLETADO!', (w//4, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    cv2.imshow('Limpiacristales - Prototipo V1', frame_final)

    if cv2.waitKey(1) & 0xFF == 27: # Salir con ESC
        break

cap.release()
cv2.destroyAllWindows()