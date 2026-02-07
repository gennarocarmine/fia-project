import streamlit as st
import numpy as np
import pandas as pd
import time
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import game_logic as gl
import ai_engines as ai

st.set_page_config(page_title="T.W.A.I. - Connect4", page_icon="🔴", layout="centered")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles/style.css")

# Carichiamo il modello MLP una sola volta e lo teniamo in cache per tutta la sessione
@st.cache_resource
def load_mlp_model():
    try:
        df = pd.read_csv("connect4_dataset_hq.csv")
        X = df.iloc[:, 0:42].values 
        y = df.iloc[:, 42].values   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        acc = accuracy_score(y_test, mlp.predict(X_test))
        return mlp, acc
    except FileNotFoundError:
        return None, 0.0

if 'board' not in st.session_state:
    st.session_state.board = gl.create_board()
    st.session_state.game_over = False
    st.session_state.turn = 0 # 0: Player, 1: AI
    st.session_state.winner = None
    st.session_state.winning_cells = [] 
    st.session_state.last_move = None # Per evidenziare l'ultima mossa AI

def handle_click(col):
    """Gestisce il click sui pulsanti superiori"""
    if not st.session_state.game_over and st.session_state.turn == 0:
        if gl.is_valid_location(st.session_state.board, col):
            row = gl.get_next_open_row(st.session_state.board, col)
            gl.drop_piece(st.session_state.board, row, col, gl.PLAYER_PIECE)
            
            if gl.winning_move(st.session_state.board, gl.PLAYER_PIECE):
                st.session_state.game_over = True
                st.session_state.winner = "PLAYER"
                st.session_state.winning_cells = gl.get_winning_coordinates(st.session_state.board, gl.PLAYER_PIECE)
            else:
                st.session_state.turn = 1 # Passa turno all'AI

def reset_game():
    st.session_state.board = gl.create_board()
    st.session_state.game_over = False
    st.session_state.turn = 0
    st.session_state.winner = None
    st.session_state.winning_cells = []
    st.session_state.last_move = None

with st.sidebar:
    st.title("⚙️ Impostazioni")
    
    st.markdown("### 🤖 Scegli il Cervello")
    algo_choice = st.radio(
        "", 
        ["Minimax (Alpha-Beta)", "Rete Neurale (MLP)"],
        captions=["Logica Pura", "Intuito Statistico (Veloce)"]
    )
    
    mlp_model, mlp_acc = load_mlp_model()
    
    if algo_choice == "Rete Neurale (MLP)":
        if mlp_model:
            st.success(f"🧠 Rete Neurale Attiva\nAccuratezza Test: **{mlp_acc:.1%}**")
        else:
            st.error("Dataset non trovato. Esegui prima 'generate_dataset.py'")
     
    st.markdown("---")
    if st.button("🔄 Nuova Partita", use_container_width=True):
        reset_game()
        st.rerun()

st.title("🔴 T.W.A.I. 🟡")
st.caption("Tris Was Already Invented")

status_placeholder = st.empty()
if st.session_state.game_over:
    if st.session_state.winner == "PLAYER":
        status_placeholder.markdown('<div class="status-box status-win">🏆 VITTORIA! L\'umanità prevale.</div>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown('<div class="status-box status-lose">💀 SCONFITTA. L\'algoritmo è superiore.</div>', unsafe_allow_html=True)
else:
    if st.session_state.turn == 0:
        status_placeholder.markdown('<div class="status-box status-player">🔴 TOCCA A TE</div>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown(f'<div class="status-box status-ai">🟡 L\'IA ({algo_choice.split()[0]}) STA CALCOLANDO...</div>', unsafe_allow_html=True)

btns_disabled = st.session_state.game_over or st.session_state.turn == 1

cols = st.columns(gl.COLUMN_COUNT)
for c in range(gl.COLUMN_COUNT):
    # Controlla se la colonna è piena per disabilitare il singolo bottone
    col_full = not gl.is_valid_location(st.session_state.board, c)
    
    # Pulsante "Freccia giù"
    if cols[c].button("⬇️", key=f"drop_{c}", disabled=btns_disabled or col_full, use_container_width=True):
        handle_click(c)
        st.rerun()

game_container = st.container()

with game_container:
    for r in range(gl.ROW_COUNT-1, -1, -1):
        cols = st.columns(gl.COLUMN_COUNT)
        for c in range(gl.COLUMN_COUNT):
            val = st.session_state.board[r][c]
            
            css_class = "empty-cell"
            label = "" # Vuoto di default
            
            if val == gl.PLAYER_PIECE:
                css_class = "player-cell"
            elif val == gl.AI_PIECE:
                css_class = "ai-cell"
            
            # Evidenzia celle vincenti
            if (r, c) in st.session_state.winning_cells:
                css_class += " winning-cell"
            

            cols[c].markdown(f"""
                <button class="{css_class}" style="
                    width:100%; 
                    aspect-ratio:1/1; 
                    border-radius:50%; 
                    border:none; 
                    cursor:default;" disabled>
                </button>
                """, unsafe_allow_html=True)

if not st.session_state.game_over and st.session_state.turn == 1:
    time.sleep(0.3) # Piccola pausa per UX
    
    col = None
    start_time = time.time()
    
    if algo_choice.startswith("Minimax"):
        # Profondità 5 per bilanciare velocità e intelligenza
        col, _ = ai.minimax(st.session_state.board, 5, -float('inf'), float('inf'), True)
    else:
        # MLP
        if mlp_model:
            col = ai.get_neural_move(mlp_model, st.session_state.board)
        else:
            col = random.choice(gl.get_valid_locations(st.session_state.board))
    
    end_time = time.time()
    
    # Applicazione Mossa AI
    if col is not None and gl.is_valid_location(st.session_state.board, col):
        row = gl.get_next_open_row(st.session_state.board, col)
        gl.drop_piece(st.session_state.board, row, col, gl.AI_PIECE)
        st.session_state.last_move = (row, col)
        
        if gl.winning_move(st.session_state.board, gl.AI_PIECE):
            st.session_state.game_over = True
            st.session_state.winner = "AI"
            st.session_state.winning_cells = gl.get_winning_coordinates(st.session_state.board, gl.AI_PIECE)
        else:
            st.session_state.turn = 0
            
    st.rerun()