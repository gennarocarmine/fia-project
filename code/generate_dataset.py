import numpy as np
import pandas as pd
import random

ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER_PIECE = 1 
AI_PIECE = -1    

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    # Controllo Orizzontale
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece: return True
    # Controllo Verticale
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece: return True
    # Diagonale Positiva
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece: return True
    # Diagonale Negativa
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece: return True
    return False

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE: opp_piece = AI_PIECE

    if window.count(piece) == 4: score += 100
    elif window.count(piece) == 3 and window.count(0) == 1: score += 5
    elif window.count(piece) == 2 and window.count(0) == 2: score += 2
    if window.count(opp_piece) == 3 and window.count(0) == 1: score -= 4
    return score

def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)
    return score

def minimax_score(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_locations) == 0
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER_PIECE): return 1000000
            elif winning_move(board, AI_PIECE): return -1000000
            else: return 0 
        else:
            return score_position(board, PLAYER_PIECE)

    if maximizingPlayer:
        value = -float('inf')
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax_score(b_copy, depth-1, alpha, beta, False)
            value = max(value, new_score)
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value
    else:
        value = float('inf')
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax_score(b_copy, depth-1, alpha, beta, True)
            value = min(value, new_score)
            beta = min(beta, value)
            if alpha >= beta: break
        return value


def generate_high_quality_data(num_samples=2000):
    print(f"Inizio generazione di {num_samples} campioni sintetici.")
    print("Utilizzo di Minimax per l'etichettatura automatica (Labeling)...")
    
    data = []
    
    for i in range(num_samples):
        if i > 0 and i % 100 == 0: print(f"Generati {i}/{num_samples}...")
            
        board = create_board()
        # Simulazione stato di metà partita (State Sampling)
        moves_made = random.randint(4, 24) 
        
        game_over_early = False
        piece_to_move = PLAYER_PIECE 

        for _ in range(moves_made):
            valid_cols = get_valid_locations(board)
            if not valid_cols: break 
            
            col = random.choice(valid_cols)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, piece_to_move)
            
            if winning_move(board, piece_to_move):
                game_over_early = True
                break
            piece_to_move *= -1 

        label = 0
        if game_over_early:
            if winning_move(board, PLAYER_PIECE): label = 1
            elif winning_move(board, AI_PIECE): label = -1
        else:
            is_maximizing = (piece_to_move == PLAYER_PIECE)
            score = minimax_score(board, 3, -float('inf'), float('inf'), is_maximizing)
            
            if score > 50: label = 1      # Classe 1: Vince Player
            elif score < -50: label = -1  # Classe -1: Vince AI
            else: label = 0               # Classe 0: Pareggio/Incerto

        # Flattening (da Matrice a Vettore)
        flat_board = board.flatten().tolist()
        flat_board.append(label)
        data.append(flat_board)

    # Salvataggio CSV
    cols = [f"pos_{i}" for i in range(42)] + ["winner"]
    df = pd.DataFrame(data, columns=cols)
    
    df.to_csv("connect4_dataset_hq.csv", index=False)
    print("\n--- COMPLETATO ---")
    print("Dataset salvato come 'connect4_dataset_hq.csv'")

if __name__ == "__main__":
    generate_high_quality_data()