import pygame
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from typing import List
import csv

ANCHO = 800
ALTO = 600
FPS = 60
COLOR_FONDO = (30, 30, 40)
COLOR_SUELO = (60, 60, 80)
COLOR_JUGADOR = (100, 200, 255)
COLOR_PELOTA = (255, 80, 80)
COLOR_TEXTO = (220, 220, 220)
GRAVEDAD = 0.8
VEL_SALTO_INICIAL = -13
VEL_PELOTA_MIN = -8
VEL_PELOTA_MAX = -4
INTERVALO_PELOTA = 1800
ALTURA_SUELO = 500
TAM_JUGADOR = 32
RADIO_PELOTA = 12
MAX_PROFUNDIDAD_ARBOL = 5
CAPAS_MLP = (12,)

@dataclass
class Muestra:
    velocidad_bala: float
    distancia_jugador: float
    altura_pelota: float
    salto: int

class Juego:
    def __init__(self):
        pygame.init()
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Juego Bala y Salto - Arbol vs MLP")
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.Font(None, 28)
        self.fuente_peq = pygame.font.Font(None, 20)
        self.reset_juego()
        self.modelo_arbol = None
        self.modelo_mlp = None
        self.datos: List[Muestra] = []
        self.modo_auto = None
        self.mensaje = ""
        self.tiempo_mensaje = 0
        self.mostrando_menu = True
        self.ejecutando = True
        self.muestras_totales = 0
        self.acierto_arbol = 0
        self.acierto_mlp = 0

    def reset_juego(self):
        self.jugador_x = 100
        self.jugador_y = ALTURA_SUELO - TAM_JUGADOR
        self.jugador_vy = 0
        self.en_suelo = True
        self.pelota_x = -100
        self.pelota_y = ALTURA_SUELO - TAM_JUGADOR
        self.pelota_vx = 0
        self.pelota_activa = False
        self.tiempo_ultima_pelota = 0
        self.puntuacion = 0
        self.perdio = False

    def disparar_pelota(self):
        self.pelota_activa = True
        self.pelota_x = ANCHO + 20
        self.pelota_y = np.random.randint(ALTURA_SUELO - TAM_JUGADOR - 80, ALTURA_SUELO - TAM_JUGADOR + 10)
        self.pelota_vx = np.random.uniform(VEL_PELOTA_MIN, VEL_PELOTA_MAX)

    def iniciar_salto(self):
        if self.en_suelo:
            self.jugador_vy = VEL_SALTO_INICIAL
            self.en_suelo = False

    def manejar_salto(self):
        if not self.en_suelo:
            self.jugador_vy += GRAVEDAD
            self.jugador_y += self.jugador_vy
            if self.jugador_y >= ALTURA_SUELO - TAM_JUGADOR:
                self.jugador_y = ALTURA_SUELO - TAM_JUGADOR
                self.jugador_vy = 0
                self.en_suelo = True

    def registrar_decision(self, salto_decision):
        if self.pelota_activa:
            distancia = self.pelota_x - self.jugador_x
            muestra = Muestra(
                velocidad_bala=self.pelota_vx,
                distancia_jugador=distancia,
                altura_pelota=self.pelota_y,
                salto=salto_decision
            )
            self.datos.append(muestra)

    def preparar_datos(self):
        if len(self.datos) < 5:
            return None, None, None, None
        X = np.array([[m.velocidad_bala, m.distancia_jugador, m.altura_pelota] for m in self.datos])
        y = np.array([m.salto for m in self.datos])
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def entrenar_arbol(self):
        if len(self.datos) < 5:
            self.mensaje = "Pocos datos. Recolecta mas en modo manual."
            self.tiempo_mensaje = pygame.time.get_ticks()
            return
        X_train, X_test, y_train, y_test = self.preparar_datos()
        self.modelo_arbol = DecisionTreeClassifier(max_depth=MAX_PROFUNDIDAD_ARBOL, random_state=42)
        self.modelo_arbol.fit(X_train, y_train)
        y_pred = self.modelo_arbol.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.mensaje = f"Arbol entrenado. Accuracy: {acc:.3f}. Muestras: {len(self.datos)}"
        self.tiempo_mensaje = pygame.time.get_ticks()

    def entrenar_mlp(self):
        if len(self.datos) < 5:
            self.mensaje = "Pocos datos. Recolecta mas en modo manual."
            self.tiempo_mensaje = pygame.time.get_ticks()
            return
        X_train, X_test, y_train, y_test = self.preparar_datos()
        self.modelo_mlp = MLPClassifier(
            hidden_layer_sizes=CAPAS_MLP,
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42
        )
        self.modelo_mlp.fit(X_train, y_train)
        y_pred = self.modelo_mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.mensaje = f"MLP entrenado. Accuracy: {acc:.3f}. Muestras: {len(self.datos)}"
        self.tiempo_mensaje = pygame.time.get_ticks()

    def decision_auto_arbol(self):
        if self.modelo_arbol is None or not self.pelota_activa:
            return False
        distancia = self.pelota_x - self.jugador_x
        features = np.array([[self.pelota_vx, distancia, self.pelota_y]])
        return self.modelo_arbol.predict(features)[0] == 1

    def decision_auto_mlp(self):
        if self.modelo_mlp is None or not self.pelota_activa:
            return False
        distancia = self.pelota_x - self.jugador_x
        features = np.array([[self.pelota_vx, distancia, self.pelota_y]])
        return self.modelo_mlp.predict(features)[0] == 1

    def exportar_csv(self):
        if not self.datos:
            self.mensaje = "No hay datos para exportar."
            self.tiempo_mensaje = pygame.time.get_ticks()
            return
        archivo = "datos_juego.csv"
        with open(archivo, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["velocidad_bala", "distancia_jugador", "altura_pelota", "salto"])
            for m in self.datos:
                writer.writerow([m.velocidad_bala, m.distancia_jugador, m.altura_pelota, m.salto])
        self.mensaje = f"Datos exportados a {archivo} ({len(self.datos)} muestras)"
        self.tiempo_mensaje = pygame.time.get_ticks()

    def dibujar_suelo(self):
        pygame.draw.rect(self.pantalla, COLOR_SUELO, (0, ALTURA_SUELO, ANCHO, ALTO - ALTURA_SUELO))
        for i in range(0, ANCHO, 40):
            pygame.draw.line(self.pantalla, (80, 80, 100), (i, ALTURA_SUELO), (i + 20, ALTURA_SUELO), 2)

    def dibujar_jugador(self):
        y = self.jugador_y if self.en_suelo or self.jugador_y < ALTURA_SUELO - TAM_JUGADOR else ALTURA_SUELO - TAM_JUGADOR
        pygame.draw.rect(self.pantalla, COLOR_JUGADOR, (self.jugador_x, y, TAM_JUGADOR, TAM_JUGADOR))
        pygame.draw.circle(self.pantalla, (150, 220, 255), (self.jugador_x + TAM_JUGADOR // 2, y - 4), 10)

    def dibujar_pelota(self):
        if self.pelota_activa:
            pygame.draw.circle(self.pantalla, COLOR_PELOTA, (int(self.pelota_x), int(self.pelota_y)), RADIO_PELOTA)
            pygame.draw.circle(self.pantalla, (255, 150, 150), (int(self.pelota_x), int(self.pelota_y)), RADIO_PELOTA - 4)

    def dibujar_menu(self):
        self.pantalla.fill(COLOR_FONDO)
        titulo = self.fuente.render("JUEGO BALA Y SALTO", True, COLOR_TEXTO)
        self.pantalla.blit(titulo, (ANCHO // 2 - titulo.get_width() // 2, 50))

        opciones = [
            ("M - Modo Manual", "Juega con ESPACIO; se recolectan datos"),
            ("A - Modo Auto (Arbol)", "Juega el arbol de decision"),
            ("N - Modo Auto (MLP)", "Juega la red neuronal"),
            ("T - Entrenar Arbol", f"DecisionTree (profundidad {MAX_PROFUNDIDAD_ARBOL})"),
            ("P - Entrenar MLP", f"MLP {CAPAS_MLP[0]} neuronas"),
            ("C - Exportar CSV", "Guarda datos_juego.csv"),
            ("Q - Salir", "Cierra el juego"),
        ]

        y = 120
        for tecla, desc in opciones:
            txt = self.fuente.render(tecla, True, COLOR_TEXTO)
            self.pantalla.blit(txt, (150, y))
            txt2 = self.fuente_peq.render(desc, True, (160, 160, 180))
            self.pantalla.blit(txt2, (400, y + 4))
            y += 45

        info_y = y + 30
        info = f"Muestras: {len(self.datos)}  |  Arbol: {'listo' if self.modelo_arbol else 'no'}  |  MLP: {'listo' if self.modelo_mlp else 'no'}"
        txt_info = self.fuente.render(info, True, (180, 220, 180))
        self.pantalla.blit(txt_info, (ANCHO // 2 - txt_info.get_width() // 2, info_y))

        if self.mensaje and pygame.time.get_ticks() - self.tiempo_mensaje < 3000:
            txt_msg = self.fuente_peq.render(self.mensaje, True, (255, 220, 100))
            self.pantalla.blit(txt_msg, (ANCHO // 2 - txt_msg.get_width() // 2, info_y + 40))

        pygame.display.flip()

    def dibujar_hud(self):
        modo_txt = "MANUAL" if self.modo_auto is None else ("AUTO (Arbol)" if self.modo_auto == 'arbol' else "AUTO (MLP)")
        color_modo = (100, 200, 100) if self.modo_auto else (200, 200, 100)
        txt = self.fuente_peq.render(f"Modo: {modo_txt}", True, color_modo)
        self.pantalla.blit(txt, (10, 10))
        txt2 = self.fuente_peq.render(f"Puntaje: {self.puntuacion}  Muestras: {len(self.datos)}", True, COLOR_TEXTO)
        self.pantalla.blit(txt2, (10, 35))
        if self.mensaje and pygame.time.get_ticks() - self.tiempo_mensaje < 2000:
            txt_msg = self.fuente_peq.render(self.mensaje, True, (255, 220, 100))
            self.pantalla.blit(txt_msg, (10, 60))
        if self.pelota_activa and self.modo_auto == 'mlp' and self.modelo_mlp:
            distancia = self.pelota_x - self.jugador_x
            features = np.array([[self.pelota_vx, distancia, self.pelota_y]])
            proba = self.modelo_mlp.predict_proba(features)[0]
            proba_salto = proba[1] if len(proba) > 1 else 0
            txt_p = self.fuente_peq.render(f"proba_salto={proba_salto:.2f}", True, (200, 200, 255))
            self.pantalla.blit(txt_p, (10, 85))

    def manejar_eventos_menu(self):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.ejecutando = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_q:
                    self.ejecutando = False
                elif evento.key == pygame.K_m:
                    self.reset_juego()
                    self.modo_auto = None
                    self.datos = []
                    self.modelo_arbol = None
                    self.modelo_mlp = None
                    self.mostrando_menu = False
                    self.mensaje = "Modo manual. ESPACIO para saltar."
                    self.tiempo_mensaje = pygame.time.get_ticks()
                elif evento.key == pygame.K_a:
                    if self.modelo_arbol is None:
                        self.mensaje = "Primero entrena el arbol con T"
                        self.tiempo_mensaje = pygame.time.get_ticks()
                    else:
                        self.reset_juego()
                        self.modo_auto = 'arbol'
                        self.mostrando_menu = False
                elif evento.key == pygame.K_n:
                    if self.modelo_mlp is None:
                        self.mensaje = "Primero entrena el MLP con P"
                        self.tiempo_mensaje = pygame.time.get_ticks()
                    else:
                        self.reset_juego()
                        self.modo_auto = 'mlp'
                        self.mostrando_menu = False
                elif evento.key == pygame.K_t:
                    self.entrenar_arbol()
                elif evento.key == pygame.K_p:
                    self.entrenar_mlp()
                elif evento.key == pygame.K_c:
                    self.exportar_csv()

    def manejar_eventos_juego(self):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.ejecutando = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE or evento.key == pygame.K_p:
                    if self.perdio:
                        self.reset_juego()
                    self.mostrando_menu = True
                elif evento.key == pygame.K_q:
                    self.ejecutando = False
                elif evento.key == pygame.K_SPACE and self.modo_auto is None and not self.perdio:
                    self.registrar_decision(1)
                    self.iniciar_salto()

    def actualizar_pelota(self):
        if not self.pelota_activa:
            ahora = pygame.time.get_ticks()
            if ahora - self.tiempo_ultima_pelota > INTERVALO_PELOTA:
                self.disparar_pelota()
                self.tiempo_ultima_pelota = ahora
        else:
            self.pelota_x += self.pelota_vx
            if self.pelota_x < -50:
                self.pelota_activa = False
                self.puntuacion += 1

    def verificar_colision(self):
        if not self.pelota_activa:
            return False
        distancia_x = abs(self.pelota_x - (self.jugador_x + TAM_JUGADOR // 2))
        distancia_y = abs(self.pelota_y - (self.jugador_y + TAM_JUGADOR // 2))
        return distancia_x < TAM_JUGADOR // 2 + RADIO_PELOTA and distancia_y < TAM_JUGADOR // 2 + RADIO_PELOTA

    def ejecutar_modo_auto(self):
        if self.modo_auto == 'arbol':
            if self.decision_auto_arbol():
                self.iniciar_salto()
        elif self.modo_auto == 'mlp':
            if self.decision_auto_mlp():
                self.iniciar_salto()

    def loop(self):
        while self.ejecutando:
            self.reloj.tick(FPS)
            if self.mostrando_menu:
                self.manejar_eventos_menu()
                self.dibujar_menu()
            else:
                self.manejar_eventos_juego()
                if self.perdio:
                    self.dibujar_juego()
                    continue
                self.actualizar_pelota()
                self.manejar_salto()
                if self.modo_auto is not None:
                    self.ejecutar_modo_auto()
                elif self.pelota_activa and self.en_suelo:
                    self.registrar_decision(0)
                if self.verificar_colision():
                    self.perdio = True
                    self.mensaje = f"Colision! Puntuacion: {self.puntuacion}"
                    self.tiempo_mensaje = pygame.time.get_ticks()
                self.dibujar_juego()
        pygame.quit()

    def dibujar_juego(self):
        self.pantalla.fill(COLOR_FONDO)
        self.dibujar_suelo()
        self.dibujar_jugador()
        self.dibujar_pelota()
        self.dibujar_hud()

        if self.perdio:
            s = pygame.Surface((ANCHO, ALTO), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.pantalla.blit(s, (0, 0))
            txt = self.fuente.render(f"GAME OVER - Puntaje: {self.puntuacion}", True, (255, 100, 100))
            self.pantalla.blit(txt, (ANCHO // 2 - txt.get_width() // 2, ALTO // 2 - 20))
            txt2 = self.fuente_peq.render("Presiona ESC para volver al menu", True, COLOR_TEXTO)
            self.pantalla.blit(txt2, (ANCHO // 2 - txt2.get_width() // 2, ALTO // 2 + 20))

        pygame.display.flip()

if __name__ == "__main__":
    juego = Juego()
    juego.loop()
