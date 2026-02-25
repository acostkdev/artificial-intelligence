import { UI } from './ui.js';
import { checkWinner } from './engine.js';
let state = createInitialState();
UI.resetButton.addEventListener('click', () => {
    state = createInitialState();
    UI.render(state);
    console.log("Epale epale");
});
UI.initBoard(handleTurn);
UI.render(state);
function createInitialState() {
    return { board: Array(9).fill(null), currentPlayer: 'X', isGameOver: false, winner: null };
}
function handleTurn(index) {
    if (state.board[index] || state.isGameOver)
        return;
    state.board[index] = state.currentPlayer;
    const winner = checkWinner(state.board);
    if (winner) {
        state.isGameOver = true;
        state.winner = winner;
    }
    else {
        state.currentPlayer = state.currentPlayer === 'X' ? 'O' : 'X';
    }
    UI.render(state);
}
//# sourceMappingURL=main.js.map