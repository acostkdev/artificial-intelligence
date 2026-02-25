export type Player = 'X' | 'O';
export type CellValue = Player | null;
export type GameState = {
    board: CellValue[];
    currentPlayer: Player;
    isGameOver: boolean;
    winner: Player | 'Draw' | null;
};