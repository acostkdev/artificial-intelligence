import os
import csv
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Opcional: para graficar los datos en 2D y 3D
import matplotlib
# Configuramos backend para ventanas interactivas (TkAgg funciona en la mayoría de sistemas)
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass  # Usa el backend por defecto
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, necesario para activar 3D en matplotlib

# Activamos modo interactivo para que las ventanas no bloqueen el juego
plt.ion()


# Ventana base y factor de escala
BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1


_FRAMES_AGACHADO = 18
_FRAMES_AGACHADO_AUTO = 45
_FRAMES_CONSECUTIVOS = 3
_FRAMES_QUIETO_MUESTREO = 30
_UMBRAL = 0.35


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    altura_bala: int
    accion: int


class Juego:
    def __init__(self) -> None:
        pygame.init()

        # Ventana fija (sin redimensionamiento automático) para evitar
        # problemas en pantallas muy grandes / 2K / 4K.
        self._flags = 0
        self._fullscreen = False

        # Tamaño fijo de ventana
        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + MLP (solo memoria)")

        # Colores
        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        # Estado global
        self.corriendo = True
        self.modo_auto = False

        # Datos / modelo
        self.datos_modelo: List[Sample] = []
        self.modelo: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        self.clase_unica: Optional[int] = None
        self.ultima_proba: Optional[list] = None
        self.tipo_modelo: Optional[str] = None

        # Geometría / física (se rellenan en _apply_resolution)
        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (32, 48)
        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)
        self.fondo_speed = 3

        self.salto = False
        self.en_suelo = True
        self.agachado = False
        self.timer_agachado = 0
        self.cooldown_agachado = 0
        self.buffer_decision: deque = deque(maxlen=_FRAMES_CONSECUTIVOS)
        self.frames_subsampling = 0
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        self.velocidad_bala = -12
        self.altura_bala = 0
        self.bala_disparada = False
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- resolución / assets -----------------
    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin = int(50 * self.scale)
        ground_offset = int(100 * self.scale)
        self.ground_y = self.h - ground_offset

        self.player_size = (int(32 * self.scale), int(48 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial

        self.decision_window = int(500 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(self.margin, self.ground_y, self.player_size[0], self.player_size[1])
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0],
                self.bullet_size[1],
            )
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0],
                self.ship_size[1],
            )

    def _cargar_assets(self) -> None:
        def safe_load(path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, "assets/sprites/mono_frame_1.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_2.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_3.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_4.png"), self.player_size),
        ]
        self.jugador_agachado_frames = []
        for frame in self.jugador_frames:
            try:
                ag = pygame.transform.scale(frame, (self.player_size[0], self.player_size[1] // 2))
            except Exception:
                ag = pygame.Surface((self.player_size[0], self.player_size[1] // 2), pygame.SRCALPHA)
                ag.fill((200, 100, 100, 255))
            self.jugador_agachado_frames.append(ag)

        self.bala_img = safe_load(
            os.path.join(base, "assets/sprites/purple_ball.png"),
            self.bullet_size,
            (160, 120, 255, 255),
        )
        self.fondo_img = safe_load(
            os.path.join(base, "assets/game/fondo2.png"),
            (self.w, self.h),
            (40, 40, 40, 255),
        )
        self.nave_img = safe_load(
            os.path.join(base, "assets/game/ufo.png"),
            self.ship_size,
            (140, 255, 200, 255),
        )

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            # Volver a ventana fija BASE_W x BASE_H
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- estado juego / modelo -----------------
    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)
        self.bala_disparada = False
        self.velocidad_bala = int(-10 * self.scale)
        self.altura_bala = 0
        self.salto = False
        self.en_suelo = True
        self.agachado = False
        self.timer_agachado = 0
        self.salto_vel = self.salto_vel_inicial
        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None

    # ----------------- export / gráficas -----------------

    def exportar_datos_csv(self) -> str:
        """
        Exporta el contenido de self.datos_modelo a un CSV sencillo.
        Devuelve un mensaje con la ruta del archivo o el motivo del fallo.
        """
        if not self.datos_modelo:
            return "No hay datos para exportar."

        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")

        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "altura_bala", "accion"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala, s.accion])
        except Exception as e:
            return f"Error al guardar CSV: {e}"

        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        """
        Grafica velocidad_bala vs distancia en 2D,
        coloreando por salto (0 / 1).
        Abre una ventana interactiva (desde el hilo principal, no bloqueante).
        """
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        cs = ["red" if s.salto == 1 else "blue" for s in self.datos_modelo]

        # Cerrar figura anterior si existe para evitar acumulación
        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()
        
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos entrenamiento MLP (rojo=salto, azul=no salto)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        # Mostrar sin bloquear (modo interactivo ya está activado con plt.ion())
        plt.show(block=False)
        plt.draw()  # Forzar actualización de la ventana

        return "Mostrando gráfica 2D interactiva (puedes rotar/zoom)."

    def graficar_datos_3d(self) -> str:
        """
        Grafica velocidad_bala vs distancia vs índice de tiempo (frame) en 3D,
        coloreando por salto (0 / 1).
        Abre una ventana interactiva (desde el hilo principal, no bloqueante).
        """
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = list(range(len(self.datos_modelo)))  # eje "tiempo" aproximado
        cs = ["red" if s.salto == 1 else "blue" for s in self.datos_modelo]

        # Cerrar figura anterior si existe para evitar acumulación
        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
        plt.clf()

        # Crear eje 3D correctamente desde la figura
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Índice (tiempo aproximado)")
        ax.set_title("Datos entrenamiento MLP 3D (rojo=salto, azul=no salto)")
        plt.tight_layout()
        # Mostrar sin bloquear (modo interactivo ya está activado con plt.ion())
        plt.show(block=False)
        plt.draw()  # Forzar actualización de la ventana

        return "Mostrando gráfica 3D interactiva (puedes rotar/zoom)."

    # ----------------- bala / salto -----------------
    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-18, -4) * self.scale)
            self.altura_bala = random.randint(0, 1)
            if self.altura_bala == 0:
                self.bala.y = int(self.ground_y + 30 * self.scale)
            else:
                self.bala.y = int(self.ground_y + 5 * self.scale)
            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    def iniciar_agacharse(self, frames: int = _FRAMES_AGACHADO) -> None:
        if self.en_suelo and not self.salto:
            self.agachado = True
            self.timer_agachado = frames

    def manejar_agachado(self) -> None:
        if self.agachado and self.timer_agachado > 0:
            self.timer_agachado -= 1
            if self.timer_agachado == 0:
                self.agachado = False
                self.cooldown_agachado = 8
        if self.cooldown_agachado > 0:
            self.cooldown_agachado -= 1

    def _colisiona(self) -> bool:
        if not self.bala_disparada:
            return False
        if self.agachado:
            player_check = pygame.Rect(
                self.jugador.x,
                self.jugador.y + self.player_size[1] // 2,
                self.player_size[0],
                self.player_size[1] // 2
            )
            return player_check.colliderect(self.bala)
        if self.altura_bala == 1:
            return (self.jugador.x < self.bala.x + self.bala.width and
                    self.jugador.x + self.jugador.width > self.bala.x)
        return self.jugador.colliderect(self.bala)

    # ----------------- datos / ML -----------------
    def registrar_decision_manual(self, accion: Optional[int] = None) -> None:
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)

        if accion is not None:
            # Registro por tecla presionada
            self.datos_modelo.append(
                Sample(
                    velocidad_bala=float(self.velocidad_bala),
                    distancia=float(distancia),
                    altura_bala=self.altura_bala,
                    accion=accion,
                )
            )
            return

        # Registro automático de "quieto" con submuestreo
        if self.en_suelo and not self.agachado:
            self.frames_subsampling += 1
            if self.frames_subsampling >= _FRAMES_QUIETO_MUESTREO:
                self.frames_subsampling = 0
                self.datos_modelo.append(
                    Sample(
                        velocidad_bala=float(self.velocidad_bala),
                        distancia=float(distancia),
                        altura_bala=self.altura_bala,
                        accion=0,
                    )
                )

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas mas datos (>= 80). Juega en MANUAL."
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            tipo = {0: "SIEMPRE QUIETO", 1: "SIEMPRE SALTA", 2: "SIEMPRE AGACHADO"}
            return True, f"Modelo trivial: {tipo.get(self.clase_unica, str(self.clase_unica))}. Junta datos de las 3 clases."

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = MLPClassifier(
            hidden_layer_sizes=(8, 4),
            activation="relu",
            solver="adam",
            max_iter=300000,
            random_state=42,
        )
        clf.fit(X_train_s, y_train)
        acc = clf.score(X_test_s, y_test)

        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = clf.predict(X_test_s)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True
        self.tipo_modelo = "mlp"

        return True, (f"MLP entrenado. Acc={acc:.3f} Prec={precision:.3f} "
                      f"Rec={recall:.3f} F1={f1:.3f}")

    def entrenar_arbol(self) -> Tuple[bool, str]:
        from sklearn.tree import DecisionTreeClassifier
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas mas datos (>= 80). Juega en MANUAL."
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            self.tipo_modelo = "arbol"
            tipo = {0: "SIEMPRE QUIETO", 1: "SIEMPRE SALTA", 2: "SIEMPRE AGACHADO"}
            return True, f"Arbol trivial: {tipo.get(self.clase_unica, str(self.clase_unica))}. Junta datos de las 3 clases."

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = clf.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        self._reset_modelo()
        self.modelo = clf
        self.modelo_entrenado = True
        self.tipo_modelo = "arbol"

        return True, (f"Arbol entrenado. Acc={acc:.3f} Prec={precision:.3f} "
                      f"Rec={recall:.3f} F1={f1:.3f}")

    def decision_auto(self) -> int:
        if not self.modelo_entrenado:
            return 0

        if not self.bala_disparada:
            return 0

        distancia = abs(self.jugador.x - self.bala.x)

        if self.clase_unica is not None and self.modelo is None:
            self.ultima_proba = [1.0 if i == self.clase_unica else 0.0 for i in range(3)]
            return self.clase_unica

        if self.modelo is None or (self.tipo_modelo == "mlp" and self.scaler is None):
            return 0

        if self.tipo_modelo == "arbol":
            Xs = [[float(self.velocidad_bala), float(distancia), float(self.altura_bala)]]
        else:
            Xs = self.scaler.transform([[float(self.velocidad_bala), float(distancia), float(self.altura_bala)]])

        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            self.ultima_proba = list(probas)
            best_idx = int(probas.argmax())
            if probas[best_idx] < _UMBRAL:
                return -1
            return int(self.modelo.classes_[best_idx])
        else:
            decision = int(self.modelo.predict(Xs)[0])
            self.ultima_proba = [1.0 if i == decision else 0.0 for i in range(3)]
            return decision

    # ----------------- menú -----------------
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa modelo; sin modelo NO juega solo)",
            "T - Entrenar MLP",
            "R - Entrenar Arbol de Decision",
            "C - Exportar datos a CSV",
            "F - Fullscreen (toggle)",
            "Q - Salir",
        ]
        x0 = int(80 * self.scale)
        y = int(140 * self.scale)
        line_h = self.fuente.get_linesize()
        pad = max(6, int(6 * self.scale))
        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad

        y += int(8 * self.scale)
        tipo_txt = self.tipo_modelo.upper() if self.tipo_modelo else "ninguno"
        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelo: {'si' if self.modelo_entrenado else 'no'} [{tipo_txt}]",
            f"Resolucion: {self.w}x{self.h} | scale={self.scale:.2f}",
            "Controles: FLECHA ARRIBA=salto, FLECHA ABAJO=agacharse",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()

        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))

        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado:
                            msg = "Primero entrena el MLP (T) en esta sesion."
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_r:
                        ok, info = self.entrenar_arbol()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    # ----------------- render / loop -----------------
    def _update_frame(self) -> None:
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w:
            self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w:
            self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count = 0

        if self.agachado:
            self.pantalla.blit(
                self.jugador_agachado_frames[self.current_frame],
                (self.jugador.x, self.jugador.y + self.player_size[1] // 2)
            )
        else:
            self.pantalla.blit(self.jugador_frames[self.current_frame], (self.jugador.x, self.jugador.y))

        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        if self._colisiona():
            self._reset_estado_juego()

        if self.modelo_entrenado and self.modo_auto and self.ultima_proba is not None:
            nombres = {0: "quieto", 1: "salto", 2: "agachado"}
            partes = []
            for i, p in enumerate(self.ultima_proba):
                if hasattr(self.modelo, "classes_") and i < len(self.modelo.classes_):
                    cls = int(self.modelo.classes_[i])
                else:
                    cls = i
                partes.append(f"{nombres.get(cls, str(cls))}:{p:.2f}")
            txt = self.fuente_chica.render(" | ".join(partes), True, self.AMARILLO)
            self.pantalla.blit(txt, (10, 10))

        if self.modo_auto:
            estado_txt = f"auto | {'saltando' if self.salto else 'agachado' if self.agachado else 'quieto'}"
            lbl = self.fuente_chica.render(estado_txt, True, self.GRIS)
            self.pantalla.blit(lbl, (10, self.h - int(30 * self.scale)))

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif e.key == pygame.K_UP and (not self.modo_auto) and self.en_suelo and not self.agachado:
                        self.iniciar_salto()
                        self.registrar_decision_manual(1)
                    elif e.key == pygame.K_DOWN and (not self.modo_auto) and self.en_suelo and not self.agachado:
                        self.iniciar_agacharse()
                        self.registrar_decision_manual(2)

            if not self.corriendo:
                break

            if self.salto:
                self.manejar_salto()

            if self.modo_auto:
                decision = self.decision_auto()
                self.buffer_decision.append(decision)
                if len(self.buffer_decision) == _FRAMES_CONSECUTIVOS and all(d == self.buffer_decision[0] for d in self.buffer_decision):
                    decision = self.buffer_decision[0]
                    self.buffer_decision.clear()
                else:
                    decision = -1
                if decision == 1 and self.en_suelo and not self.agachado and self.cooldown_agachado == 0:
                    self.iniciar_salto()
                elif decision == 2 and self.en_suelo and not self.salto and not self.agachado and self.cooldown_agachado == 0:
                    self.iniciar_agacharse(_FRAMES_AGACHADO_AUTO)
                elif decision == 0:
                    self.agachado = False
                    self.timer_agachado = 0
            else:
                self.registrar_decision_manual()

            self.manejar_agachado()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()


def main() -> None:
    Juego().loop()


if __name__ == "__main__":
    main()

