import os
import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
plt.ion()

from tensorflow import keras
from tensorflow.keras import layers


BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1

_FRAMES_AGACHADO = 18
_FRAMES_AGACHADO_AUTO = 45
_FRAMES_CONSECUTIVOS = 3
_FRAMES_QUIETO_MUESTREO = 30
_UMBRAL = 0.35
_EPOCHS_RNN = 60
_BATCH_SIZE_RNN = 32

LONGITUDES_SEC = [5, 10, 20]
TIPOS_RNN = ["SimpleRNN", "GRU", "LSTM"]


@dataclass
class FrameSecuencia:
    velocidad_bala: float
    distancia: float
    salto: int


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    altura_bala: int
    accion: int


class JuegoRNN:
    def __init__(self):
        pygame.init()
        self._flags = 0
        self._fullscreen = False
        start_w, start_h = BASE_W, BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + Salto + RNN")

        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        self.corriendo = True
        self.modo_auto = False
        self.modo_rnn = False

        self.datos_mlp: List[Sample] = []
        self.modelo_mlp: Optional[MLPClassifier] = None
        self.scaler_mlp: Optional[StandardScaler] = None
        self.mlp_entrenado = False
        self.clase_unica_mlp: Optional[int] = None
        self.ultima_proba_mlp: Optional[list] = None
        self.mlp_metricas: Optional[dict] = None

        self.longitud_secuencia = 10
        self.indice_longitud = 1
        self.tipo_rnn = "GRU"
        self.indice_tipo_rnn = 1
        self.buffer_secuencia: deque = deque(maxlen=self.longitud_secuencia)
        self.secuencias_modelo: List[dict] = []
        self.modelo_rnn = None
        self.scaler_rnn: Optional[StandardScaler] = None
        self.rnn_entrenado = False
        self.ultima_proba_rnn: Optional[float] = None
        self.rnn_metricas: Optional[dict] = None

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

    def _apply_resolution(self, w, h, reset_positions):
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
            self.jugador = pygame.Rect(
                self.margin, self.ground_y,
                self.player_size[0], self.player_size[1]
            )
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0], self.bullet_size[1],
            )
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0], self.ship_size[1],
            )

    def _cargar_assets(self):
        def safe_load(path, size, fallback_color=(200, 200, 200, 255)):
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "project", "project1", "pygames"
        )
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
            self.bullet_size, (160, 120, 255, 255),
        )
        self.fondo_img = safe_load(
            os.path.join(base, "assets/game/fondo2.png"),
            (self.w, self.h), (40, 40, 40, 255),
        )
        self.nave_img = safe_load(
            os.path.join(base, "assets/game/ufo.png"),
            self.ship_size, (140, 255, 200, 255),
        )

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    def _reset_estado_juego(self):
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

    def _reset_modelos(self):
        self.modelo_mlp = None
        self.scaler_mlp = None
        self.mlp_entrenado = False
        self.clase_unica_mlp = None
        self.mlp_metricas = None
        self.modelo_rnn = None
        self.scaler_rnn = None
        self.rnn_entrenado = False
        self.rnn_metricas = None

    def disparar_bala(self):
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-18, -4) * self.scale)
            self.altura_bala = random.randint(0, 1)
            if self.altura_bala == 0:
                self.bala.y = int(self.ground_y + 30 * self.scale)
            else:
                self.bala.y = int(self.ground_y + 5 * self.scale)
            self.bala_disparada = True

    def reset_bala(self):
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self):
        if self.en_suelo and not self.agachado:
            self.salto = True
            self.en_suelo = False

    def manejar_salto(self):
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    def iniciar_agacharse(self, frames=_FRAMES_AGACHADO):
        if self.en_suelo and not self.salto:
            self.agachado = True
            self.timer_agachado = frames

    def manejar_agachado(self):
        if self.agachado and self.timer_agachado > 0:
            self.timer_agachado -= 1
            if self.timer_agachado == 0:
                self.agachado = False
                self.cooldown_agachado = 8
        if self.cooldown_agachado > 0:
            self.cooldown_agachado -= 1

    def _colisiona(self):
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

    def registrar_decision_manual(self, accion: Optional[int] = None):
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)
        if accion is not None:
            self.datos_mlp.append(
                Sample(
                    velocidad_bala=float(self.velocidad_bala),
                    distancia=float(distancia),
                    altura_bala=self.altura_bala,
                    accion=accion,
                )
            )
            return
        if self.en_suelo and not self.agachado:
            self.frames_subsampling += 1
            if self.frames_subsampling >= _FRAMES_QUIETO_MUESTREO:
                self.frames_subsampling = 0
                self.datos_mlp.append(
                    Sample(
                        velocidad_bala=float(self.velocidad_bala),
                        distancia=float(distancia),
                        altura_bala=self.altura_bala,
                        accion=0,
                    )
                )

    def registrar_frame_secuencia(self):
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)
        frame = FrameSecuencia(
            velocidad_bala=float(self.velocidad_bala),
            distancia=float(distancia),
            salto=0 if self.en_suelo else 1,
        )
        self.buffer_secuencia.append(frame)
        if len(self.buffer_secuencia) == self.longitud_secuencia:
            self.secuencias_modelo.append({
                'frames': list(self.buffer_secuencia),
                'decision': frame.salto,
            })

    def preparar_datos_rnn(self):
        if len(self.secuencias_modelo) < 50:
            return None, None, None
        X_list = []
        y_list = []
        for seq in self.secuencias_modelo:
            features = [[f.velocidad_bala, f.distancia] for f in seq['frames']]
            X_list.append(features)
            y_list.append(seq['decision'])
        X = np.array(X_list)
        y = np.array(y_list)
        X_reshaped = X.reshape(-1, 2)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_reshaped)
        X_norm = X_norm.reshape(X.shape)
        return X_norm, y, scaler

    def crear_modelo_rnn(self, tipo=None):
        if tipo is None:
            tipo = self.tipo_rnn
        if tipo == "LSTM":
            capa = layers.LSTM(32, return_sequences=False)
        elif tipo == "GRU":
            capa = layers.GRU(32, return_sequences=False)
        else:
            capa = layers.SimpleRNN(32, return_sequences=False)
        modelo = keras.Sequential([
            layers.Input(shape=(self.longitud_secuencia, 2)),
            capa,
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid'),
        ])
        modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return modelo

    def entrenar_mlp(self):
        samples = list(self.datos_mlp)
        if len(samples) < 80:
            return False, "Necesitas mas datos (>= 80). Juega en MANUAL."

        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelos()
            self.clase_unica_mlp = int(clases[0])
            self.mlp_entrenado = True
            tipo = {0: "SIEMPRE QUIETO", 1: "SIEMPRE SALTA", 2: "SIEMPRE AGACHADO"}
            return True, f"MLP trivial: {tipo.get(self.clase_unica_mlp, str(self.clase_unica_mlp))}. Junta datos de las 3 clases."

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
        y_pred = clf.predict(X_test_s)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        self._reset_modelos()
        self.scaler_mlp = scaler
        self.modelo_mlp = clf
        self.mlp_entrenado = True
        self.mlp_metricas = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tipo': 'MLP (3 clases)',
        }

        return True, (f"MLP entrenado. Acc={acc:.3f} Prec={precision:.3f} "
                      f"Rec={recall:.3f} F1={f1:.3f}")

    def entrenar_rnn(self, tipo=None):
        if tipo is None:
            tipo = self.tipo_rnn
        X, y, scaler = self.preparar_datos_rnn()
        if X is None:
            return False, f"Necesitas mas secuencias (>= 50). Llevas {len(self.secuencias_modelo)}."

        if len(set(y)) < 2:
            self.rnn_entrenado = True
            self.modelo_rnn = None
            self.scaler_rnn = None
            self.rnn_metricas = None
            return True, f"RNN trivial: {'siempre salta' if y[0] == 1 else 'nunca salta'}. Junta datos de ambas clases."

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        modelo = self.crear_modelo_rnn(tipo)

        t_inicio = time.time()
        historia = modelo.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=_EPOCHS_RNN,
            batch_size=_BATCH_SIZE_RNN,
            verbose=0,
        )
        t_fin = time.time()

        y_pred_prob = modelo.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        self._reset_modelos()
        self.scaler_rnn = scaler
        self.modelo_rnn = modelo
        self.rnn_entrenado = True
        self.rnn_metricas = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tiempo': t_fin - t_inicio,
            'tipo': f'RNN ({tipo})',
        }

        return True, (f"RNN ({tipo}) entrenada. Acc={acc:.3f} Prec={precision:.3f} "
                      f"Rec={recall:.3f} F1={f1:.3f} T={t_fin-t_inicio:.1f}s")

    def decision_mlp(self):
        if not self.mlp_entrenado:
            return 0
        if not self.bala_disparada:
            return 0

        distancia = abs(self.jugador.x - self.bala.x)

        if self.clase_unica_mlp is not None and self.modelo_mlp is None:
            self.ultima_proba_mlp = [1.0 if i == self.clase_unica_mlp else 0.0 for i in range(3)]
            return self.clase_unica_mlp

        if self.modelo_mlp is None or self.scaler_mlp is None:
            return 0

        Xs = self.scaler_mlp.transform(
            [[float(self.velocidad_bala), float(distancia), float(self.altura_bala)]]
        )
        if hasattr(self.modelo_mlp, "predict_proba"):
            probas = self.modelo_mlp.predict_proba(Xs)[0]
            self.ultima_proba_mlp = list(probas)
            best_idx = int(probas.argmax())
            if probas[best_idx] < _UMBRAL:
                return -1
            return int(self.modelo_mlp.classes_[best_idx])
        else:
            decision = int(self.modelo_mlp.predict(Xs)[0])
            self.ultima_proba_mlp = [1.0 if i == decision else 0.0 for i in range(3)]
            return decision

    def decision_auto_rnn(self):
        if not self.rnn_entrenado or len(self.buffer_secuencia) < self.longitud_secuencia:
            return 0
        if not self.bala_disparada or not self.en_suelo:
            return 0

        features = [[f.velocidad_bala, f.distancia] for f in self.buffer_secuencia]
        X = np.array([features])
        X_reshaped = X.reshape(-1, 2)
        X_norm = self.scaler_rnn.transform(X_reshaped)
        X_norm = X_norm.reshape(X.shape)

        proba = self.modelo_rnn.predict(X_norm, verbose=0)[0][0]
        self.ultima_proba_rnn = proba

        return 1 if proba >= 0.5 else 0

    def comparar_modelos(self):
        mensajes = []
        if self.mlp_entrenado and self.mlp_metricas:
            m = self.mlp_metricas
            mensajes.append(
                f"MLP: Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} "
                f"Rec={m['recall']:.3f} F1={m['f1']:.3f}"
            )
        if self.rnn_entrenado and self.rnn_metricas:
            m = self.rnn_metricas
            mensajes.append(
                f"{m['tipo']}: Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} "
                f"Rec={m['recall']:.3f} F1={m['f1']:.3f} T={m['tiempo']:.1f}s"
            )
        return "\n".join(mensajes) if mensajes else "No hay modelos entrenados para comparar."

    def exportar_secuencias_csv(self):
        if not self.secuencias_modelo:
            return "No hay secuencias para exportar."
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "secuencias_rnn.csv")
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["secuencia_id", "frame_index", "velocidad_bala", "distancia", "decision"])
                for sid, seq in enumerate(self.secuencias_modelo):
                    for fidx, frame in enumerate(seq['frames']):
                        writer.writerow([sid, fidx, frame.velocidad_bala, frame.distancia, seq['decision']])
        except Exception as e:
            return f"Error al guardar CSV: {e}"
        return f"CSV guardado ({len(self.secuencias_modelo)} secuencias)."

    def graficar_secuencias(self):
        if len(self.secuencias_modelo) < 1:
            return "No hay secuencias para graficar."
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle("Secuencias de juego - distancia y velocidad en el tiempo")
        for seq in self.secuencias_modelo[:20]:
            frames = seq['frames']
            tiempos = list(range(len(frames)))
            distancias = [f.distancia for f in frames]
            velocidades = [abs(f.velocidad_bala) for f in frames]
            label = "salta" if seq['decision'] == 1 else "quieto"
            color = "red" if seq['decision'] == 1 else "blue"
            axes[0].plot(tiempos, distancias, color=color, alpha=0.5, linewidth=0.8)
            axes[1].plot(tiempos, velocidades, color=color, alpha=0.5, linewidth=0.8)
        axes[0].set_ylabel("Distancia")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel("Velocidad (abs)")
        axes[1].set_xlabel("Frame en secuencia")
        axes[1].grid(True, alpha=0.3)
        from matplotlib.patches import Patch
        axes[0].legend(handles=[
            Patch(color='red', label='salta'),
            Patch(color='blue', label='quieto'),
        ], loc='upper right')
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando graficas de secuencias."

    def graficar_comparativa(self):
        if not self.mlp_metricas and not self.rnn_metricas:
            return "No hay metricas para graficar."
        nombres = []
        accs = []
        precs = []
        recs = []
        f1s = []
        if self.mlp_metricas:
            nombres.append("MLP")
            accs.append(self.mlp_metricas['accuracy'])
            precs.append(self.mlp_metricas['precision'])
            recs.append(self.mlp_metricas['recall'])
            f1s.append(self.mlp_metricas['f1'])
        if self.rnn_metricas:
            nombres.append(self.rnn_metricas['tipo'])
            accs.append(self.rnn_metricas['accuracy'])
            precs.append(self.rnn_metricas['precision'])
            recs.append(self.rnn_metricas['recall'])
            f1s.append(self.rnn_metricas['f1'])
        x = np.arange(len(nombres))
        ancho = 0.2
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - 1.5*ancho, accs, ancho, label='Accuracy')
        ax.bar(x - 0.5*ancho, precs, ancho, label='Precision')
        ax.bar(x + 0.5*ancho, recs, ancho, label='Recall')
        ax.bar(x + 1.5*ancho, f1s, ancho, label='F1')
        ax.set_xticks(x)
        ax.set_xticklabels(nombres)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Comparativa MLP vs RNN")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando grafica comparativa."

    def _dibujar_menu(self, msg=""):
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENU - Juego Bala y Salto RNN", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(40 * self.scale)))
        opciones = [
            "M - Manual (recolectar datos)",
            "A - Auto MLP (usa MLP entrenado)",
            "R - Auto RNN (usa RNN entrenado)",
            "T - Entrenar MLP",
            "Y - Entrenar RNN (tipo actual)",
            "",
            "1/2/3 - Tipo RNN: 1=SimpleRNN  2=GRU  3=LSTM",
            "5/6/7 - Long secuencia: 5=5  6=10  7=20",
            "",
            "C - Exportar secuencias a CSV",
            "V - Visualizar secuencias",
            "G - Grafica comparativa MLP vs RNN",
            "F - Fullscreen",
            "Q - Salir",
        ]
        x0 = int(80 * self.scale)
        y = int(100 * self.scale)
        line_h = self.fuente.get_linesize()
        pad = max(4, int(4 * self.scale))
        for op in opciones:
            if op == "":
                y += int(6 * self.scale)
                continue
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad
        y += int(10 * self.scale)
        tipo_txt_rnn = f"{self.tipo_rnn} (seq={self.longitud_secuencia})"
        tipo_txt_mlp = self.mlp_metricas['tipo'] if self.mlp_metricas else "no entrenado"
        estado = [
            f"MLP datos: {len(self.datos_mlp)} | MLP: {'si' if self.mlp_entrenado else 'no'}",
            f"RNN secuencias: {len(self.secuencias_modelo)} | RNN: {'si' if self.rnn_entrenado else 'no'}",
            f"Tipo RNN: {tipo_txt_rnn}",
            f"Buffer: {len(self.buffer_secuencia)}/{self.longitud_secuencia} | {len(self.secuencias_modelo)} secs guardadas",
            f"Resolucion: {self.w}x{self.h} | scale={self.scale:.2f}",
            "Controles: FLECHA ARRIBA=salto, FLECHA ABAJO=agacharse",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()
        if self.rnn_metricas:
            m = self.rnn_metricas
            t_rnn = self.fuente_chica.render(
                f"RNN: Acc={m['accuracy']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} T={m['tiempo']:.0f}s",
                True, self.AMARILLO,
            )
            self.pantalla.blit(t_rnn, (x0, y))
            y += self.fuente_chica.get_linesize()
        if self.mlp_metricas:
            m = self.mlp_metricas
            t_mlp = self.fuente_chica.render(
                f"MLP: Acc={m['accuracy']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}",
                True, self.AMARILLO,
            )
            self.pantalla.blit(t_mlp, (x0, y))
            y += self.fuente_chica.get_linesize()
        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))
        pygame.display.flip()

    def mostrar_menu(self):
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
                        self.modo_rnn = False
                        self.datos_mlp.clear()
                        self.secuencias_modelo.clear()
                        self.buffer_secuencia.clear()
                        self._reset_modelos()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.mlp_entrenado:
                            msg = "Primero entrena el MLP (T)."
                        else:
                            self.modo_auto = True
                            self.modo_rnn = False
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_r:
                        if not self.rnn_entrenado:
                            msg = "Primero entrena la RNN (Y)."
                        else:
                            self.modo_auto = True
                            self.modo_rnn = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_mlp()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_y:
                        ok, info = self.entrenar_rnn()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_1:
                        self.indice_tipo_rnn = 0
                        self.tipo_rnn = TIPOS_RNN[0]
                        msg = f"Tipo RNN cambiado a {self.tipo_rnn}"
                    if e.key == pygame.K_2:
                        self.indice_tipo_rnn = 1
                        self.tipo_rnn = TIPOS_RNN[1]
                        msg = f"Tipo RNN cambiado a {self.tipo_rnn}"
                    if e.key == pygame.K_3:
                        self.indice_tipo_rnn = 2
                        self.tipo_rnn = TIPOS_RNN[2]
                        msg = f"Tipo RNN cambiado a {self.tipo_rnn}"
                    if e.key == pygame.K_5:
                        self.indice_longitud = 0
                        self.longitud_secuencia = LONGITUDES_SEC[0]
                        self.buffer_secuencia = deque(maxlen=self.longitud_secuencia)
                        msg = f"Longitud secuencia cambiada a {self.longitud_secuencia}"
                    if e.key == pygame.K_6:
                        self.indice_longitud = 1
                        self.longitud_secuencia = LONGITUDES_SEC[1]
                        self.buffer_secuencia = deque(maxlen=self.longitud_secuencia)
                        msg = f"Longitud secuencia cambiada a {self.longitud_secuencia}"
                    if e.key == pygame.K_7:
                        self.indice_longitud = 2
                        self.longitud_secuencia = LONGITUDES_SEC[2]
                        self.buffer_secuencia = deque(maxlen=self.longitud_secuencia)
                        msg = f"Longitud secuencia cambiada a {self.longitud_secuencia}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_secuencias_csv()
                    if e.key == pygame.K_v:
                        msg = self.graficar_secuencias()
                    if e.key == pygame.K_g:
                        msg = self.graficar_comparativa()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    def _update_frame(self):
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

        y_info = 10
        if self.modo_auto and not self.modo_rnn and self.mlp_entrenado and self.ultima_proba_mlp is not None:
            nombres = {0: "quieto", 1: "salto", 2: "agachado"}
            partes = []
            for i, p in enumerate(self.ultima_proba_mlp):
                if hasattr(self.modelo_mlp, "classes_") and i < len(self.modelo_mlp.classes_):
                    cls = int(self.modelo_mlp.classes_[i])
                else:
                    cls = i
                partes.append(f"{nombres.get(cls, str(cls))}:{p:.2f}")
            txt = self.fuente_chica.render(" | ".join(partes), True, self.AMARILLO)
            self.pantalla.blit(txt, (10, y_info))
            y_info += self.fuente_chica.get_linesize() + 4

        if self.modo_auto and self.modo_rnn and self.rnn_entrenado and self.ultima_proba_rnn is not None:
            txt = self.fuente_chica.render(
                f"RNN proba salto: {self.ultima_proba_rnn:.3f}",
                True, self.AMARILLO,
            )
            self.pantalla.blit(txt, (10, y_info))
            y_info += self.fuente_chica.get_linesize() + 4

        if self.modo_auto:
            label = "auto RNN" if self.modo_rnn else "auto MLP"
            estado_txt = f"{label} | {'saltando' if self.salto else 'agachado' if self.agachado else 'quieto'}"
            lbl = self.fuente_chica.render(estado_txt, True, self.GRIS)
            self.pantalla.blit(lbl, (10, self.h - int(30 * self.scale)))

    def loop(self):
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

            if not self.modo_auto:
                self.registrar_frame_secuencia()
                self.registrar_decision_manual()

            if self.modo_auto:
                if self.modo_rnn:
                    self.registrar_frame_secuencia()
                    decision = self.decision_auto_rnn()
                    if decision == 1 and self.en_suelo and not self.agachado and self.cooldown_agachado == 0:
                        self.iniciar_salto()
                else:
                    decision = self.decision_mlp()
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

            self.manejar_agachado()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()


def main():
    JuegoRNN().loop()


if __name__ == "__main__":
    main()
