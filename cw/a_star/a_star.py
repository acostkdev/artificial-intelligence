import pygame
import heapq

ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Algoritmo A* - Busqueda de Caminos")

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_camino(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        if self.fila > 0:
            self.vecinos.append(grid[self.fila - 1][self.col])
        if self.fila < self.total_filas - 1:
            self.vecinos.append(grid[self.fila + 1][self.col])
        if self.col > 0:
            self.vecinos.append(grid[self.fila][self.col - 1])
        if self.col < self.total_filas - 1:
            self.vecinos.append(grid[self.fila][self.col + 1])

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        if actual.es_inicio():
            continue
        actual.hacer_camino()
        dibujar()

def algoritmo_a_star(dibujar, grid, inicio, fin):
    contador = 0
    open_set = []
    heapq.heappush(open_set, (0, contador, inicio))

    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0

    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        _, _, actual = heapq.heappop(open_set)

        if actual != inicio and actual != fin:
            if actual.color == ROJO:
                continue
            actual.hacer_cerrado()

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            print("=== LISTA CERRADA (nodos explorados) ===")
            cerrados = []
            for fila in grid:
                for nodo in fila:
                    if nodo.color == ROJO:
                        cerrados.append(nodo.get_pos())
            print(cerrados)
            print(f"Total nodos explorados: {len(cerrados)}")
            return True


        for vecino in actual.vecinos:
            if vecino.es_pared() or vecino.color == ROJO:
                continue

            temp_g_score = g_score[actual] + 1
            if temp_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + h(vecino.get_pos(), fin.get_pos())
                contador += 1
                heapq.heappush(open_set, (f_score[vecino], contador, vecino))
                if vecino != fin:
                    vecino.hacer_abierto()

        dibujar()

    return False

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def main(ventana, ancho):
    FILAS = 10
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    if algoritmo_a_star(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin):
                        print("Camino encontrado!")
                    else:
                        print("No hay camino posible")

                if event.key == pygame.K_r:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)
