import random
import numpy as np
import game_logic as gl 

def evaluate_window(window, piece):
    score = 0
    opp_piece = gl.PLAYER_PIECE if piece == gl.AI_PIECE else gl.AI_PIECE
    
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(gl.EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(gl.EMPTY) == 2:
        score += 2
    
    if window.count(opp_piece) == 3 and window.count(gl.EMPTY) == 1:
        score -= 4 
        
    return score

def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, gl.COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    for r in range(gl.ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(gl.COLUMN_COUNT-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    for c in range(gl.COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(gl.ROW_COUNT-3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    for r in range(gl.ROW_COUNT-3):
        for c in range(gl.COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    for r in range(gl.ROW_COUNT-3):
        for c in range(gl.COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
            
    return score

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = gl.get_valid_locations(board)
    
    center_preferred_order = [3, 2, 4, 1, 5, 0, 6]
    
    valid_locations = [col for col in center_preferred_order if col in valid_locations]

    is_terminal = gl.winning_move(board, gl.PLAYER_PIECE) or gl.winning_move(board, gl.AI_PIECE) or len(valid_locations) == 0
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if gl.winning_move(board, gl.AI_PIECE):
                return (None, 100000000000000 + depth) # Vittoria certa
            elif gl.winning_move(board, gl.PLAYER_PIECE):
                return (None, -10000000000000 - depth) # Sconfitta certa
            else: # Pareggio
                return (None, 0)
        else: # Profondità 0, usa l'euristica
            return (None, score_position(board, gl.AI_PIECE))

    if maximizingPlayer: # Turno AI
        value = -float('inf')
        column = valid_locations[0] 
        
        for col in valid_locations:
            row = gl.get_next_open_row(board, col)
            b_copy = board.copy()
            gl.drop_piece(b_copy, row, col, gl.AI_PIECE)
            
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            
            if new_score > value:
                value = new_score
                column = col
            
            # Alpha-Beta Pruning
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Taglio del ramo (Beta Cutoff)
                
        return column, value

    else: # Turno Giocatore (Minimizing)
        value = float('inf')
        column = valid_locations[0]
        
        for col in valid_locations:
            row = gl.get_next_open_row(board, col)
            b_copy = board.copy()
            gl.drop_piece(b_copy, row, col, gl.PLAYER_PIECE)
            
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            
            if new_score < value:
                value = new_score
                column = col
            
            # Alpha-Beta Pruning
            beta = min(beta, value)
            if alpha >= beta:
                break # Taglio del ramo (Alpha Cutoff)
                
        return column, value

def get_neural_move(model, board):
    valid_locations = gl.get_valid_locations(board)

    for col in valid_locations:
        row = gl.get_next_open_row(board, col)
        temp_board = board.copy()
        gl.drop_piece(temp_board, row, col, gl.AI_PIECE)
        if gl.winning_move(temp_board, gl.AI_PIECE):
            return col 
            
    for col in valid_locations:
        row = gl.get_next_open_row(board, col)
        temp_board = board.copy()
        gl.drop_piece(temp_board, row, col, gl.PLAYER_PIECE)
        if gl.winning_move(temp_board, gl.PLAYER_PIECE):
            return col 

    best_score = -100
    best_col = random.choice(valid_locations)
    
    random.shuffle(valid_locations) 
    
    for col in valid_locations:
        row = gl.get_next_open_row(board, col)
        temp_board = board.copy()
        gl.drop_piece(temp_board, row, col, gl.AI_PIECE)
        
        flat_board = temp_board.flatten().reshape(1, -1)
        prediction = model.predict(flat_board)[0]
        
        score = 0
        if prediction == 1: score = -50    # Vince P1 (Male per AI)
        elif prediction == 0: score = 0    # Pareggio (Neutro)
        elif prediction == -1: score = 100 # Vince AI (Ottimo)
        
        if col == 3: score += 5
        elif col == 2 or col == 4: score += 2

        if score > best_score:
            best_score = score
            best_col = col
            
    return best_col