export const UI = {
    boardElement: document.querySelector('#board'),
    statusElement: document.querySelector('#status strong'),
    resetButton: document.querySelector('#reset-btn'),
    // Inicializa el tablero en el DOM
    initBoard(onCellClick) {
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
    render(state) {
        const cells = this.boardElement.querySelectorAll('.cell');
        state.board.forEach((value, i) => {
            const cell = cells[i];
            cell.textContent = value;
            cell.className = `cell ${value ? value.toLowerCase() : ''}`;
        });
        if (state.isGameOver) {
            this.statusElement.textContent = state.winner === 'Draw'
                ? '¡Empate!'
                : `¡Ganó ${state.winner}!`;
        }
        else {
            this.statusElement.textContent = state.currentPlayer;
        }
    }
};
//# sourceMappingURL=ui.js.map