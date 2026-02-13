import time
import pandas as pd
import game_logic as gl
import ai_engines as ai

def run_benchmark():
    board = gl.create_board()
    
    gl.drop_piece(board, 0, 3, gl.PLAYER_PIECE) # Player gioca al centro
    gl.drop_piece(board, 1, 3, gl.AI_PIECE)     # AI risponde sopra
    
    depths = [1, 2, 3, 4, 5, 6] # Fermati a 6, il 7 potrebbe impiegare minuti
    results = []

    print(f"{'Depth':<10} | {'Time (sec)':<15} | {'Nodi stimati'}")
    print("-" * 45)

    for d in depths:
        start_time = time.time()
    
        _ = ai.minimax(board, d, -float('inf'), float('inf'), True)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        nodes_est = 7**d 
        
        print(f"{d:<10} | {elapsed:.4f}          | ~{nodes_est}")
        
        results.append({
            "Profondità": d,
            "Tempo (s)": round(elapsed, 4),
            "Note": "Ok" if elapsed < 1.0 else ("Lento" if elapsed > 5.0 else "Accettabile")
        })

if __name__ == "__main__":
    run_benchmark()