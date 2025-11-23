import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --- CONFIGURACIÓN INICIAL ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Configuración Nivel 1 (Suciedad General)
RADIO_PINCEL = 40  
COLOR_SUCIEDAD = (200, 200, 200) # Gris claro
OPACIDAD_SUCIEDAD = 0.8
NIVEL_1_UMBRAL = 25 # Porcentaje para completar el Nivel 1

# Configuración Nivel 2 (Rápido - Reacción)
LEVEL_2_TIME_LIMIT = 3.0 # Segundos antes de que la mancha desaparezca
LEVEL_2_SPOTS_TOTAL = 5 # Cuántas manchas hay que limpiar/intentar para terminar el Nivel 2
MANCHA_RADIO = 30 # Tamaño de la mancha pequeña
MANCHA_COLOR = (0, 165, 255) # Color Naranja/Ámbar para diferenciación

# Configuración Nivel 3 (Mantener la Mano)
LEVEL_3_HOVER_TIME = 3.0 # Segundos que debe mantenerse la mano sobre la mancha
LEVEL_3_SPOTS_TOTAL = 5 # Cuántas manchas hay que limpiar para terminar el Nivel 3
MANCHA_COLOR_HOVER = (0, 0, 255) # Color Rojo

# Nombre de la ventana unificado para evitar múltiples ventanas
WINDOW_NAME = 'Limpiacristales - Juego'

# Inicializamos cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variables de control de estado del juego
game_state = 'START' # 'START', 'PLAYING', 'COMPLETED_L1', 'LEVEL_TWO', 'COMPLETED_L2', 'LEVEL_THREE', 'FINISHED'
mask_suciedad = None # Máscara para Nivel 1

# Variables de Tiempos y Récords (Nivel 1)
level_start_time = 0.0 
current_attempt_time = 0.0 
best_time = float('inf') 

# Variables de Nivel 2 (Rápido)
level_2_spot = None # Posición de la mancha
level_2_spawn_time = 0.0 # Momento en que apareció la mancha
level_2_spots_done = 0 # Contador de manchas procesadas (aparecidas)
level_2_spots_hit = 0 # Contador de manchas acertadas (puntos)

# Variables de Nivel 3 (Hover)
level_3_spot = None # (cx, cy) de la mancha actual
time_on_spot = 0.0 # Tiempo acumulado sobre la mancha
hover_start_time = None # Hora de inicio del hover
level_3_spots_cleaned = 0 # Contador de manchas limpiadas


def create_new_spot(w, h, margin_factor=100):
    """Genera una nueva mancha en una posición aleatoria de la pantalla."""
    margin = MANCHA_RADIO + margin_factor 
    cx = random.randint(margin, w - margin)
    cy = random.randint(margin, h - margin)
    return (cx, cy)


def start_new_spot_level_2(w, h):
    """Inicializa una nueva mancha para el Nivel 2."""
    global level_2_spot, level_2_spawn_time
    level_2_spot = create_new_spot(w, h)
    level_2_spawn_time = time.time()


print("Juego Listo. Esperando 'Enter' para empezar.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    current_time = time.time() # Almacena el tiempo actual una vez por bucle

    # MANEJO DE TECLAS
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27: # Salir con ESC
        break
    
    if key == 13: # Iniciar/Reiniciar/Siguiente Nivel con ENTER
        if game_state == 'START':
            game_state = 'PLAYING'
            mask_suciedad = None # Inicializa Nivel 1
            level_start_time = current_time # INICIA EL TIEMPO DEL NIVEL 1
            current_attempt_time = 0.0
        elif game_state == 'COMPLETED_L1':
            game_state = 'LEVEL_TWO'
            level_2_spots_done = 0
            level_2_spots_hit = 0
            start_new_spot_level_2(w, h) # Comienza Nivel 2
        elif game_state == 'COMPLETED_L2':
            game_state = 'LEVEL_THREE'
            level_3_spots_cleaned = 0
            level_3_spot = create_new_spot(w, h) # Comienza Nivel 3
            time_on_spot = 0.0
            hover_start_time = None
        elif game_state == 'FINISHED':
             game_state = 'START' # Vuelve al inicio


    # --- 1. MANEJO DE LA CÁMARA EN RGB ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hand_cx, hand_cy = -1, -1 
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        muneca = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        hand_cx = int(muneca.x * w)
        hand_cy = int(muneca.y * h)
        
        # Dibujar esqueleto de la mano (solo en el juego)
        if game_state in ['PLAYING', 'LEVEL_TWO', 'LEVEL_THREE']:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )


    # --- 2. MANEJO DE LA MÁQUINA DE ESTADOS ---

    if game_state == 'START':
        # Pantalla de Inicio
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.6 
        frame_final = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        text_start = "Pulsa ENTER para comenzar" 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        
        (tw, th), baseline = cv2.getTextSize(text_start, font, font_scale, font_thickness)
        cx = (w - tw) // 2
        cy = (h + th) // 2 
        
        cv2.putText(frame_final, text_start, (cx, cy), 
                    font, font_scale, (255, 255, 255), font_thickness)
        
        cv2.imshow(WINDOW_NAME, frame_final) 
        
    elif game_state == 'PLAYING':
        # Lógica del Juego - Nivel 1 (Limpiar niebla general)

        if mask_suciedad is None:
            mask_suciedad = np.ones((h, w), dtype=np.uint8) * 255

        current_attempt_time = current_time - level_start_time

        if hand_cx != -1:
            cv2.circle(mask_suciedad, (hand_cx, hand_cy), RADIO_PINCEL, 0, -1)
            cv2.circle(frame, (hand_cx, hand_cy), RADIO_PINCEL, (0, 255, 0), 2)

        # COMPOSICIÓN DE IMAGEN
        capa_gris = np.zeros_like(frame)
        capa_gris[:] = COLOR_SUCIEDAD
        capa_suciedad_final = cv2.bitwise_and(capa_gris, capa_gris, mask=mask_suciedad)
        frame_final = cv2.addWeighted(frame, 1, capa_suciedad_final, OPACIDAD_SUCIEDAD, 0)
        
        # CALCULO DE PROGRESO
        pixeles_totales = w * h
        pixeles_sucios = cv2.countNonZero(mask_suciedad)
        porcentaje_limpio = 100 - (pixeles_sucios / pixeles_totales * 100)
        
        # --- MOSTRAR HUD Nivel 1 ---
        cv2.putText(frame_final, f'Limpiado: {int(porcentaje_limpio)}%', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        record_text = f'RECORD: {best_time:.2f}s' if best_time != float('inf') else 'RECORD: --'
        cv2.putText(frame_final, record_text, (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        cv2.putText(frame_final, f'Tiempo: {current_attempt_time:.2f}s', (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        

        if porcentaje_limpio > NIVEL_1_UMBRAL:
            if current_attempt_time < best_time:
                best_time = current_attempt_time
            
            game_state = 'COMPLETED_L1' # Transición al estado de finalización del nivel 1
            
        cv2.imshow(WINDOW_NAME, frame_final)
        
    elif game_state == 'COMPLETED_L1':
        # Pantalla de Transición entre Nivel 1 y Nivel 2
        
        text1 = "Evento Completado!!"
        text2 = "Pulsa ENTER para el siguiente juego"
        text_last_time = f"Tu Tiempo: {current_attempt_time:.2f} segundos"
        
        if best_time != float('inf'):
            text_record = f"Record: {best_time:.2f} segundos"
        else:
            text_record = "Record: -- segundos"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 0) # Verde

        # 1. MENSAJE PRINCIPAL
        (tw1, th1), baseline1 = cv2.getTextSize(text1, font, 1.5, 4)
        cx1 = (w - tw1) // 2
        cy1 = h // 2 - 150 

        # 2. TIEMPO ACTUAL
        (tw_lt, th_lt), _ = cv2.getTextSize(text_last_time, font, 1.2, 2)
        cx_lt = (w - tw_lt) // 2
        cy_lt = h // 2 - 50 

        # 3. RECORD
        (tw_r, th_r), _ = cv2.getTextSize(text_record, font, 1.2, 2)
        cx_r = (w - tw_r) // 2
        cy_r = h // 2 

        # 4. PULSA ENTER
        (tw2, th2), baseline2 = cv2.getTextSize(text2, font, 1.0, 2)
        cx2 = (w - tw2) // 2
        cy2 = h // 2 + 100 

        cv2.putText(frame, text1, (cx1, cy1), font, 1.5, text_color, 4)
        cv2.putText(frame, text_last_time, (cx_lt, cy_lt), font, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, text_record, (cx_r, cy_r), font, 1.2, (255, 255, 0), 2)
        cv2.putText(frame, text2, (cx2, cy2), font, 1.0, text_color, 2)
        
        cv2.imshow(WINDOW_NAME, frame)

    elif game_state == 'LEVEL_TWO':
        # Lógica del Juego - Nivel 2 (Rápido - Reacción)
        
        if level_2_spots_done >= LEVEL_2_SPOTS_TOTAL:
            game_state = 'COMPLETED_L2' # Pasa a la transición al Nivel 3
            level_2_spot = None # Limpia la mancha
        
        is_on_spot = False
        
        # --- Lógica de Detección de Mancha ---
        if hand_cx != -1 and level_2_spot:
            spot_x, spot_y = level_2_spot
            
            # Dibujar la mancha
            cv2.circle(frame, level_2_spot, MANCHA_RADIO, MANCHA_COLOR, -1)

            # Calcular la distancia de la muñeca a la mancha
            distance = np.sqrt((hand_cx - spot_x)**2 + (hand_cy - spot_y)**2)
            
            if distance < MANCHA_RADIO:
                is_on_spot = True
        
        # --- Lógica del Temporizador y Acierto ---
        time_elapsed = current_time - level_2_spawn_time
        
        if is_on_spot:
            # 1. ACIERTO: Pone la mano a tiempo
            level_2_spots_hit += 1
            level_2_spots_done += 1
            start_new_spot_level_2(w, h) # Genera nueva mancha inmediatamente
        
        elif time_elapsed >= LEVEL_2_TIME_LIMIT:
            # 2. FALLO: Se acaba el tiempo
            level_2_spots_done += 1
            start_new_spot_level_2(w, h) # Genera nueva mancha
        
        # Dibujar progreso visual (solo tiempo restante)
        if level_2_spot:
            # Tiempo restante para la mancha actual
            time_left = max(0, LEVEL_2_TIME_LIMIT - time_elapsed)
            cv2.putText(frame, f'Tiempo: {time_left:.1f}s', (w - 250, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
        
        # --- Mostrar HUD de Nivel 2 ---
        cv2.putText(frame, f'Aciertos: {level_2_spots_hit}/{LEVEL_2_SPOTS_TOTAL}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (MANCHA_COLOR), 3) # Usamos el color de la mancha

        cv2.imshow(WINDOW_NAME, frame)

    elif game_state == 'COMPLETED_L2':
        # Pantalla de Transición entre Nivel 2 y Nivel 3
        
        text1 = "Nivel 2 Completado!!"
        text2 = "Pulsa ENTER para el Nivel 3"
        text_score = f"Acertaste: {level_2_spots_hit} de {LEVEL_2_SPOTS_TOTAL}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 0) # Verde

        # 1. MENSAJE PRINCIPAL
        (tw1, th1), _ = cv2.getTextSize(text1, font, 1.5, 4)
        cx1 = (w - tw1) // 2
        cy1 = h // 2 - 100

        # 2. PUNTUACIÓN
        (tw_s, th_s), _ = cv2.getTextSize(text_score, font, 1.2, 2)
        cx_s = (w - tw_s) // 2
        cy_s = h // 2 

        # 3. PULSA ENTER
        (tw2, th2), _ = cv2.getTextSize(text2, font, 1.0, 2)
        cx2 = (w - tw2) // 2
        cy2 = h // 2 + 100 

        cv2.putText(frame, text1, (cx1, cy1), font, 1.5, text_color, 4)
        cv2.putText(frame, text_score, (cx_s, cy_s), font, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, text2, (cx2, cy2), font, 1.0, text_color, 2)
        
        cv2.imshow(WINDOW_NAME, frame)

    elif game_state == 'LEVEL_THREE':
        # Lógica del Juego - Nivel 3 (Mantener la mano, el antiguo Nivel 2)
        
        if level_3_spots_cleaned >= LEVEL_3_SPOTS_TOTAL:
            game_state = 'FINISHED'
            level_3_spot = None # Limpia la mancha
        
        is_on_spot = False
        
        if hand_cx != -1 and level_3_spot:
            spot_x, spot_y = level_3_spot
            
            cv2.circle(frame, level_3_spot, MANCHA_RADIO, MANCHA_COLOR_HOVER, -1)

            distance = np.sqrt((hand_cx - spot_x)**2 + (hand_cy - spot_y)**2)
            
            if distance < MANCHA_RADIO:
                is_on_spot = True
        
        # --- Lógica del Temporizador de Hover ---
        if is_on_spot:
            if hover_start_time is None:
                hover_start_time = current_time
            
            time_on_spot = current_time - hover_start_time
            
            # Dibujar el progreso visual (círculo)
            angle = int((time_on_spot / LEVEL_3_HOVER_TIME) * 360)
            if level_3_spot:
                # Dibuja el arco de progreso amarillo
                cv2.ellipse(frame, level_3_spot, (MANCHA_RADIO + 10, MANCHA_RADIO + 10), 
                            0, 0, angle, (255, 255, 0), 5) # Amarillo
                
            # Si el tiempo requerido se cumple
            if time_on_spot >= LEVEL_3_HOVER_TIME:
                level_3_spots_cleaned += 1
                hover_start_time = None
                time_on_spot = 0.0
                
                if level_3_spots_cleaned < LEVEL_3_SPOTS_TOTAL:
                    level_3_spot = create_new_spot(w, h) # Nueva mancha
        else:
            # Si se retira la mano, reiniciamos el temporizador
            hover_start_time = None
            time_on_spot = 0.0


        # --- Mostrar HUD de Nivel 3 ---
        cv2.putText(frame, f'Limpiadas: {level_3_spots_cleaned}/{LEVEL_3_SPOTS_TOTAL}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, MANCHA_COLOR_HOVER, 3)
        
        cv2.imshow(WINDOW_NAME, frame)

    elif game_state == 'FINISHED':
        # Pantalla de Juego Terminado
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.6 
        frame_final = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        text1 = "¡JUEGO TERMINADO!"
        text2 = "Gracias por participar. Pulsa ENTER para volver a empezar."
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (tw1, th1), _ = cv2.getTextSize(text1, font, 2.0, 5)
        cx1 = (w - tw1) // 2
        cy1 = h // 2 - 50

        (tw2, th2), _ = cv2.getTextSize(text2, font, 1.0, 2)
        cx2 = (w - tw2) // 2
        cy2 = h // 2 + 50

        cv2.putText(frame_final, text1, (cx1, cy1), 
                    font, 2.0, (255, 255, 0), 5)
        cv2.putText(frame_final, text2, (cx2, cy2), 
                    font, 1.0, (255, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, frame_final)


cap.release()
cv2.destroyAllWindows()