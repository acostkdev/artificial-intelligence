import { GameState } from './types.js';

export const UI = {
    boardElement: document.querySelector('#board') as HTMLElement,
    statusElement: document.querySelector('#status strong') as HTMLElement,
    resetButton: document.querySelector('#reset-btn') as HTMLButtonElement,

    // Inicializa el tablero en el DOM
    initBoard(onCellClick: (index: number) => void) {
        this.boardElement.innerHTML = '';
        for (let i = 0; i < 9; i++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.index = i.toString();
            cell.addEventListener('click', () => onCellClick(i));
            this.boardElement.appendChild(cell);
        }
    },

    // Sincroniza el DOM con el estado actual
    render(state: GameState) {
        const cells = this.boardElement.querySelectorAll('.cell');
        
        state.board.forEach((value, i) => {
            const cell = cells[i] as HTMLElement;
            cell.textContent = value;
            cell.className = `cell ${value ? value.toLowerCase() : ''}`;
        });

        if (state.isGameOver) {
            this.statusElement.textContent = state.winner === 'Draw' 
                ? '¡Empate!' 
                : `¡Ganó ${state.winner}!`;
        } else {
            this.statusElement.textContent = state.currentPlayer;
        }
    }
};