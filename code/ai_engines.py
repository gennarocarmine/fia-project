import random
import numpy as np
import game_logic as gl 

# MiniMax con Alpha-Beta Pruning
def evaluate_window(window, piece):
    score = 0
    opp_piece = gl.PLAYER_PIECE if piece == gl.AI_PIECE else gl.AI_PIECE
    
    if window.count(piece) == 4: score += 100
    elif window.count(piece) == 3 and window.count(gl.EMPTY) == 1: score += 5
    elif window.count(piece) == 2 and window.count(gl.EMPTY) == 2: score += 2
    
    if window.count(opp_piece) == 3 and window.count(gl.EMPTY) == 1: score -= 4
    return score

def score_position(board, piece):
    score = 0
    # Preferenza Centro
    center_array = [int(i) for i in list(board[:, gl.COLUMN_COUNT//2])]
    score += center_array.count(piece) * 3
    
    # Analisi Finestre
    for r in range(gl.ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(gl.COLUMN_COUNT-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)
            
    return score

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = gl.get_valid_locations(board)
    is_terminal = gl.winning_move(board, gl.PLAYER_PIECE) or gl.winning_move(board, gl.AI_PIECE) or len(valid_locations) == 0
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if gl.winning_move(board, gl.AI_PIECE): return (None, 1000000)
            elif gl.winning_move(board, gl.PLAYER_PIECE): return (None, -1000000)
            else: return (None, 0)
        else:
            return (None, score_position(board, gl.AI_PIECE))

    if maximizingPlayer:
        value = -float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = gl.get_next_open_row(board, col)
            b_copy = board.copy()
            gl.drop_piece(b_copy, row, col, gl.AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta: break
        return column, value
    else:
        value = float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = gl.get_next_open_row(board, col)
            b_copy = board.copy()
            gl.drop_piece(b_copy, row, col, gl.PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta: break
        return column, value


# Rete Neurale (MLP)
def get_neural_move(model, board):
    valid_locations = gl.get_valid_locations(board)
    best_score = -100
    best_col = random.choice(valid_locations)
    
    for col in valid_locations:
        row = gl.get_next_open_row(board, col)
        temp_board = board.copy()
        gl.drop_piece(temp_board, row, col, gl.AI_PIECE)
        
        # Prepara input per la rete
        flat_board = temp_board.flatten().reshape(1, -1)
        prediction = model.predict(flat_board)[0]
        
        score = 0
        if prediction == gl.AI_PIECE: score = 10    # La rete prevede vittoria AI
        elif prediction == 0: score = 0             # Pareggio
        else: score = -10                           # Vittoria Player (da evitare)
        
        if score > best_score:
            best_score = score
            best_col = col
            
    return best_col