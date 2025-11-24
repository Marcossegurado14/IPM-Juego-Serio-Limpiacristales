import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os 

# --- CONFIGURACIÓN DE MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURACIÓN ESTÉTICA (COLORES BGR) ---
# Interfaz Gráfica (Azul y Blanco)
COLOR_UI_AZUL = (88, 40, 1)      # Azul corporativo oscuro
COLOR_UI_BLANCO = (255, 255, 255) # Blanco

# Colores de Juego 
COLOR_SUCIEDAD_BASE = (100, 140, 180) 
COLOR_MANCHA_LIGERA = (120, 170, 210) 
COLOR_MANCHA_PERSISTENTE = (40, 80, 139) 

# Colores Calibración
COLOR_CALIB_PENDIENTE = (0, 0, 255)   # Rojo
COLOR_CALIB_OK = (0, 255, 0)          # Verde

# --- CONFIGURACIÓN DE JUEGO ---
# Nivel 1
RADIO_PINCEL = 50  
OPACIDAD_SUCIEDAD = 0.95
NIVEL_1_UMBRAL =  10

# Nivel 2
TOTAL_MANCHAS_A_GENERAR = 10 
INTERVALO_APARICION = 2.0    
RADIO_MANCHA = 35
TIEMPO_HOVER_REQUERIDO = 1.5 

# Ventana
WINDOW_NAME = 'Juego Serio: Rehabilitacion Hombro'

# --- VARIABLES GLOBALES ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# --- CARGA DE IMÁGENES ---
def cargar_imagen(nombre):
    # Construye la ruta: carpeta_del_script/imagenes/nombre_archivo.jpg
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenes', nombre)
    
    if os.path.exists(ruta):
        return cv2.imread(ruta)
    else:
        print(f"ERROR: No se encontró la imagen en: {ruta}")
        return None

img_inicio = cargar_imagen('inicio.png')
img_calib = cargar_imagen('calibracion.png')
img_inst_n1 = cargar_imagen('instrucciones_n1.png')
img_inst_n2 = cargar_imagen('instrucciones_n2.png')
img_fin = cargar_imagen('fin.png')

# ESTADOS DEL JUEGO:
# Flujo: START -> INSTRUCT_CALIB -> CALIBRATION -> INSTRUCT_L1 -> LEVEL_1 -> INSTRUCT_L2 -> LEVEL_2 -> FINISHED
game_state = 'START' 
mask_suciedad = None 

# Tiempos
start_time_total = 0.0
time_level_1 = 0.0
time_level_2 = 0.0
start_time_l2 = 0.0
last_spawn_time = 0.0

# Objetos
manchas_nivel_2 = []
manchas_generadas_count = 0
esquinas_calibracion = [] 

# --- FUNCIONES AUXILIARES ---

def mostrar_imagen_pantalla_completa(frame_camara, imagen_fondo):
    """Ajusta la imagen al tamaño de la cámara y la devuelve"""
    h, w = frame_camara.shape[:2]
    if imagen_fondo is not None:
        return cv2.resize(imagen_fondo, (w, h))
    else:
        # Fallback si no hay imagen: Fondo blanco con texto genérico
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        blank[:] = COLOR_UI_BLANCO
        cv2.putText(blank, "Imagen no encontrada", (50, h//2), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
        return blank

def draw_centered_text(img, text, font_scale, y_offset, color=COLOR_UI_AZUL, thickness=2, x_offset=0):
    """
    Centra texto con offsets.
    y_offset: +Abajo, -Arriba
    x_offset: +Derecha, -Izquierda
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_TRIPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cx = ((w - tw) // 2) + x_offset
    cy = (h // 2) + y_offset
    cv2.putText(img, text, (cx, cy), font, font_scale, color, thickness)

def spawn_single_spot(w, h):
    margen = 100
    tipo = random.choice(['light', 'heavy'])
    return {
        'x': random.randint(margen, w - margen),
        'y': random.randint(margen, h - margen),
        'tipo': tipo,
        'active': True,
        'hover_start': None,
        'progress': 0.0
    }

def init_calibration(w, h):
    margen = 80
    return [
        {'x': margen, 'y': margen, 'reached': False},          
        {'x': w - margen, 'y': margen, 'reached': False},      
        {'x': margen, 'y': h - margen, 'reached': False},      
        {'x': w - margen, 'y': h - margen, 'reached': False}   
    ]

def check_all_calibrated(esquinas):
    for esq in esquinas:
        if not esq['reached']: return False
    return True

def check_level_2_finished(manchas, total_generadas):
    if total_generadas < TOTAL_MANCHAS_A_GENERAR: return False
    for m in manchas:
        if m['active']: return False
    return True

# --- BUCLE PRINCIPAL ---
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Preparar imagen
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    current_time = time.time()
    
    # Detección de Manos
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    puntero_x, puntero_y = -1, -1
    
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        landmark_punto = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        puntero_x = int(landmark_punto.x * w)
        puntero_y = int(landmark_punto.y * h)
        
        # Dibujar mano solo si estamos en fases interactivas
        if game_state in ['CALIBRATION', 'LEVEL_1', 'LEVEL_2']:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (puntero_x, puntero_y), 10, (0, 255, 0), -1)

    # --- MÁQUINA DE ESTADOS ---

    if game_state == 'START':
        # Mostrar imagen de inicio sin texto extra (ya está en la imagen)
        frame = mostrar_imagen_pantalla_completa(frame, img_inicio)
        
    elif game_state == 'INSTRUCT_CALIB':
        # Mostrar instrucciones de calibración
        frame = mostrar_imagen_pantalla_completa(frame, img_calib)

    elif game_state == 'CALIBRATION':
        
        if not esquinas_calibracion:
            esquinas_calibracion = init_calibration(w, h)
            
        # HUD
        cv2.rectangle(frame, (w//2 - 200, h - 100), (w//2 + 170, h - 40), (255,255,255), -1)
        cv2.putText(frame, "Toca las esquinas rojas", (w//2 - 180, h - 60), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, COLOR_UI_AZUL, 2)
            
        for esq in esquinas_calibracion:
            color = COLOR_CALIB_OK if esq['reached'] else COLOR_CALIB_PENDIENTE
            cv2.circle(frame, (esq['x'], esq['y']), 40, color, -1)
            cv2.circle(frame, (esq['x'], esq['y']), 40, (255,255,255), 2)
            
            if puntero_x != -1:
                dist = np.sqrt((puntero_x - esq['x'])**2 + (puntero_y - esq['y'])**2)
                if dist < 40:
                    esq['reached'] = True
        
        if check_all_calibrated(esquinas_calibracion):
            cv2.putText(frame, "LISTO", (w//2 - 100, h//2), 
                cv2.FONT_HERSHEY_TRIPLEX, 2, COLOR_UI_AZUL, 4)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(500) 
            game_state = 'INSTRUCT_L1' 
            mask_suciedad = None 

    elif game_state == 'INSTRUCT_L1':
        # Mostrar instrucciones Nivel 1
        frame = mostrar_imagen_pantalla_completa(frame, img_inst_n1)

    elif game_state == 'LEVEL_1':
        if mask_suciedad is None:
            mask_suciedad = np.ones((h, w), dtype=np.uint8) * 255
            start_time_total = current_time

        if puntero_x != -1:
            cv2.circle(mask_suciedad, (puntero_x, puntero_y), RADIO_PINCEL, 0, -1)

        capa_marron = np.zeros_like(frame)
        capa_marron[:] = COLOR_SUCIEDAD_BASE
        suciedad_visible = cv2.bitwise_and(capa_marron, capa_marron, mask=mask_suciedad)
        frame = cv2.addWeighted(frame, 1, suciedad_visible, OPACIDAD_SUCIEDAD, 0)
        
        dirty_pixels = cv2.countNonZero(mask_suciedad)
        percent_dirty = (dirty_pixels / (w * h)) * 100
        time_level_1 = current_time - start_time_total
        
        cv2.rectangle(frame, (30, 40), (300, 140), COLOR_UI_BLANCO, -1)
        cv2.putText(frame, f"Suciedad: {int(percent_dirty)}%", (40, 80), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, COLOR_UI_AZUL, 2)
        cv2.putText(frame, f"Tiempo: {time_level_1:.1f}s", (40, 120), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, COLOR_UI_AZUL, 2)
        
        if percent_dirty < NIVEL_1_UMBRAL:
            game_state = 'INSTRUCT_L2'

    elif game_state == 'INSTRUCT_L2':
        # Mostrar instrucciones Nivel 2
        frame = mostrar_imagen_pantalla_completa(frame, img_inst_n2)

    elif game_state == 'LEVEL_2':
        if start_time_l2 == 0:
            start_time_l2 = current_time
            last_spawn_time = current_time - INTERVALO_APARICION 
            manchas_nivel_2 = []
            manchas_generadas_count = 0
            
        time_level_2 = current_time - start_time_l2
        
        if (manchas_generadas_count < TOTAL_MANCHAS_A_GENERAR and 
            current_time - last_spawn_time > INTERVALO_APARICION):
            manchas_nivel_2.append(spawn_single_spot(w, h))
            manchas_generadas_count += 1
            last_spawn_time = current_time
        
        for mancha in manchas_nivel_2:
            if not mancha['active']: continue
            
            mx, my = mancha['x'], mancha['y']
            colision = False
            
            color_m = COLOR_MANCHA_LIGERA if mancha['tipo'] == 'light' else COLOR_MANCHA_PERSISTENTE
            cv2.circle(frame, (mx, my), RADIO_MANCHA, color_m, -1)
            cv2.circle(frame, (mx, my), RADIO_MANCHA, (255,255,255), 2)

            if puntero_x != -1:
                dist = np.sqrt((puntero_x - mx)**2 + (puntero_y - my)**2)
                if dist < RADIO_MANCHA: colision = True
            
            if colision:
                if mancha['tipo'] == 'light':
                    mancha['active'] = False 
                elif mancha['tipo'] == 'heavy':
                    if mancha['hover_start'] is None: mancha['hover_start'] = current_time
                    elapsed = current_time - mancha['hover_start']
                    mancha['progress'] = min(elapsed / TIEMPO_HOVER_REQUERIDO, 1.0)
                    angulo = int(mancha['progress'] * 360)
                    cv2.ellipse(frame, (mx, my), (RADIO_MANCHA+5, RADIO_MANCHA+5), 0, 0, angulo, (0, 255, 255), 4)
                    if mancha['progress'] >= 1.0: mancha['active'] = False
            else:
                if mancha['tipo'] == 'heavy':
                    mancha['hover_start'] = None
                    mancha['progress'] = 0.0

        active_count = sum(1 for m in manchas_nivel_2 if m['active'])
        pendientes = active_count + (TOTAL_MANCHAS_A_GENERAR - manchas_generadas_count)
        
        cv2.rectangle(frame, (30, 40), (300, 140), COLOR_UI_BLANCO, -1)
        cv2.putText(frame, f"Restantes: {pendientes}", (40, 80), 
            cv2.FONT_HERSHEY_TRIPLEX, 0.8, COLOR_UI_AZUL, 2)
        cv2.putText(frame, f"Tiempo: {time_level_2:.1f}s", (40, 120), 
            cv2.FONT_HERSHEY_TRIPLEX, 0.8, COLOR_UI_AZUL, 2)
        
        if check_level_2_finished(manchas_nivel_2, manchas_generadas_count):
            game_state = 'FINISHED'

    elif game_state == 'FINISHED':
        # Fondo de imagen FIN
        frame = mostrar_imagen_pantalla_completa(frame, img_fin)
        total_score = time_level_1 + time_level_2
        
        # Ajuste de posiciones
        pos_y_n1, pos_x_n1 = -100, 75
        pos_y_n2, pos_x_n2 = -50, 75
        pos_y_tot, pos_x_tot = 30, 75
        
        draw_centered_text(frame, f"Nivel 1: {time_level_1:.1f}s", 1.0, pos_y_n1, COLOR_UI_AZUL, 2, x_offset=pos_x_n1)
        draw_centered_text(frame, f"Nivel 2: {time_level_2:.1f}s", 1.0, pos_y_n2, COLOR_UI_AZUL, 2, x_offset=pos_x_n2)
        draw_centered_text(frame, f"TOTAL: {total_score:.2f}s", 1.5, pos_y_tot, COLOR_UI_BLANCO, 3, x_offset=pos_x_tot)
        

    # --- INPUT ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == 13: # ENTER
        if game_state == 'START':
            game_state = 'INSTRUCT_CALIB'
        elif game_state == 'INSTRUCT_CALIB':
            game_state = 'CALIBRATION'
        elif game_state == 'INSTRUCT_L1':
            game_state = 'LEVEL_1'
        elif game_state == 'INSTRUCT_L2':
            game_state = 'LEVEL_2'
        elif game_state == 'FINISHED':
            # Reiniciar variables
            mask_suciedad = None
            manchas_nivel_2 = []
            esquinas_calibracion = []
            start_time_l2 = 0
            game_state = 'START'

    cv2.imshow(WINDOW_NAME, frame)

cap.release()
cv2.destroyAllWindows()