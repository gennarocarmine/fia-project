import streamlit as st  # Importa la libreria Streamlit per creare l'interfaccia web interattiva
import numpy as np      
import random         
import pandas as pd     
import matplotlib.pyplot as plt 

st.set_page_config(page_title="(IA)'M FINE", layout="wide") 

class SetteMezzoEnv: # Definisce la classe che rappresenta l'Ambiente (il gioco)
    def __init__(self): 
        self.reset() # Chiama il metodo reset per preparare il mazzo e le mani all'avvio

    def get_card_value(self, card_code): # Metodo per convertire il codice carta nel suo valore di gioco
        if card_code == 'M': return 'M' # Se è la Matta, restituisce 'M' (valore dinamico)
        if card_code >= 8: return 0.5 # Se è una figura (8, 9, 10), vale mezzo punto
        return card_code # Altrimenti (1-7) vale il suo numero

    def calculate_score(self, hand): # Metodo per calcolare il punteggio totale di una mano
        if 'M' not in hand: # Caso semplice: non c'è la Matta
            return sum(self.get_card_value(c) for c in hand) # Somma i valori delle carte
        
        # Caso complesso: c'è la Matta. Bisogna trovare il valore migliore.
        base_sum = sum(self.get_card_value(c) for c in hand if c != 'M') # Somma tutto tranne la Matta
        possible_values = [0.5, 1, 2, 3, 4, 5, 6, 7] # Tutti i possibili valori che la Matta può assumere
        best_score = -1 # Inizializza il miglior punteggio a un valore basso
        
        for v in possible_values: # Prova ogni possibile valore della Matta
            current = base_sum + v # Calcola il totale ipotetico
            if current <= 7.5 and current > best_score: # Se non sballa ed è migliore del precedente
                best_score = current # Aggiorna il miglior punteggio trovato
        
        # Se best_score è ancora -1, significa che si sballa con qualsiasi valore.
        # In tal caso, la Matta vale il minimo possibile (0.5).
        return base_sum + 0.5 if best_score == -1 else best_score # Ritorna il punteggio calcolato

    def reset(self): # Metodo per iniziare una nuova partita
        # Crea il mazzo: 4 semi per le carte da 1 a 10
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4 
        self.deck.remove(10)  # Rimuove un 10 (il Re di Denari)
        self.deck.append('M') # Aggiunge la Matta al posto del Re di Denari
        random.shuffle(self.deck) # Mescola casualmente il mazzo
        
        self.player_hand = [self.deck.pop()] # Distribuisce la prima carta al giocatore (toglie dal mazzo)
        self.dealer_hand = [self.deck.pop()] # Distribuisce la prima carta al banco
        return self.get_state() # Restituisce lo stato iniziale del gioco

    def get_state(self): # Metodo che definisce cosa "vede" l'IA (lo Stato)
        p_score = self.calculate_score(self.player_hand) # Calcola punti giocatore
        d_vis_raw = self.dealer_hand[0] # Guarda la carta visibile del banco
        # Semplificazione per l'IA: se il banco ha figura o Matta, lo vede come 0.5 generico
        d_vis = 0.5 if d_vis_raw == 'M' or d_vis_raw >= 8 else d_vis_raw 
        return (float(p_score), float(d_vis)) # Restituisce la tupla (Punti Giocatore, Carta Banco)

    def step(self, action): # Esegue un'azione e restituisce il risultato (Step)
        # Azione 1 = Chiedere Carta
        if action == 1: 
            self.player_hand.append(self.deck.pop()) # Pesca una carta
            score = self.calculate_score(self.player_hand) # Ricalcola punteggio
            if score > 7.5: return self.get_state(), -1, True # Se sballa: reward -1, gioco finito
            return self.get_state(), 0, False # Se non sballa: reward 0, gioco continua
        
        # Azione 0 = Sto Bene (Turno al Banco)
        else: 
            d_score = self.calculate_score(self.dealer_hand) # Calcola punti banco
            while d_score < 5.0: # Strategia fissa del banco: tira se ha meno di 5
                self.dealer_hand.append(self.deck.pop()) # Banco pesca
                d_score = self.calculate_score(self.dealer_hand) # Ricalcola punti banco
            
            p_score = self.calculate_score(self.player_hand) # Punti finali giocatore
            
            # Determina vincitore
            if d_score > 7.5: return self.get_state(), 1, True # Banco sballa -> Vinci (+1)
            elif p_score > d_score: return self.get_state(), 1, True # Punteggio più alto -> Vinci (+1)
            else: return self.get_state(), -1, True # Pareggio o punteggio più basso -> Perdi (-1)


class QLearningAgent: 
    def __init__(self): 
        self.q_table = {} # Dizionario per la Q-Table (Memoria dell'agente)
        self.epsilon = 1.0 # Tasso di esplorazione iniziale (100% casuale all'inizio)
        self.alpha = 0.1   # Learning Rate (velocità di apprendimento)
        self.gamma = 0.9   # Discount Factor (importanza del futuro)
        
    def get_action(self, state, force_greedy=False): # Sceglie l'azione da compiere
        if state not in self.q_table: self.q_table[state] = [0.0, 0.0] # Se stato nuovo, inizializza a 0
        
        # Strategia Epsilon-Greedy
        if not force_greedy and random.random() < self.epsilon: # Se numero casuale < epsilon
            return random.choice([0, 1]) # Sceglie azione casuale (Esplorazione)
        return np.argmax(self.q_table[state]) # Altrimenti sceglie l'azione migliore nota (Sfruttamento)

    def update(self, state, action, reward, next_state): # Aggiorna la conoscenza (Q-Table)
        if state not in self.q_table: self.q_table[state] = [0.0, 0.0] # Inizializza stato corrente
        if next_state not in self.q_table: self.q_table[next_state] = [0.0, 0.0] # Inizializza stato futuro
        
        old_val = self.q_table[state][action] # Valore attuale nella tabella
        next_max = np.max(self.q_table[next_state]) # Miglior valore ottenibile dal prossimo stato
        
        # Formula di Bellman: Aggiorna il valore Q basandosi sulla ricompensa ricevuta e la stima futura
        new_val = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_val # Salva il nuovo valore


@st.cache_resource 
def train_agent(): # Funzione per addestrare l'IA
    env = SetteMezzoEnv() # Crea l'ambiente
    agent = QLearningAgent() # Crea l'agente
    history = [] # Lista per salvare lo storico vittorie 
    wins = 0 # Contatore vittorie
    episodes = 50000 # Numero di partite di allenamento
    
    for i in range(episodes): 
        state = env.reset() # Inizia nuova partita
        done = False # Flag fine partita
        while not done: # Finché la partita non finisce
            action = agent.get_action(state) # Agente sceglie azione
            next_state, reward, done = env.step(action) # Ambiente risponde
            agent.update(state, action, reward, next_state) # Agente impara
            state = next_state # Aggiorna stato corrente
            if done and reward == 1: wins += 1 # Conta vittoria
        
        if agent.epsilon > 0.01: agent.epsilon *= 0.9995 # Riduce epsilon (meno casualità nel tempo)
        
        if (i+1) % 1000 == 0: # Ogni 1000 episodi
            history.append(wins / 1000) # Salva win rate
            wins = 0 # Resetta contatore
            
    return agent, history # Restituisce l'agente addestrato


# INTERFACCIA DI GIOCO

st.title("🃏 (IA)'M FINE") 
st.markdown("### Sfida l'algoritmo Q-Learning a Sette e Mezzo") 

with st.spinner("🧠 L'IA sta studiando 50.000 partite..."): 
    agent, history = train_agent() 

# Inizializzazione variabili di sessione (per mantenere lo stato tra i ricaricamenti della pagina)
if 'game_active' not in st.session_state: st.session_state.game_active = False # Stato gioco attivo/fermo
if 'env' not in st.session_state: st.session_state.env = SetteMezzoEnv() # Istanza ambiente persistente
if 'msg' not in st.session_state: st.session_state.msg = "" # Messaggio esito partita

def format_card(card): # Funzione per rendere leggibili le carte
    if card == 'M': return "🃏 Matta" # Icona per Matta
    if isinstance(card, int) and card >= 8: return f"{card} (0.5)" # Specifica valore 0.5 per figure
    return str(card) # Ritorna il numero come stringa

# Creazione colonne layout: Gioco (sinistra) e Tabella (destra)
col_game, col_stats = st.columns([1, 1])

# --- COLONNA SINISTRA: AREA DI GIOCO ---
with col_game:
    st.markdown("#### 🟢 Il Tuo Tavolo") # Intestazione area gioco
    
    # Pulsante per iniziare nuova partita (visibile solo se gioco non attivo)
    if not st.session_state.game_active:
        if st.button("▶️ NUOVA PARTITA", use_container_width=True, type="primary"):
            st.session_state.env.reset() # Resetta ambiente
            st.session_state.game_active = True # Attiva gioco
            st.session_state.msg = "" # Pulisce messaggi precedenti
            st.rerun() # Ricarica la pagina per aggiornare l'interfaccia
    
    # Se la partita è in corso
    if st.session_state.game_active:
        p_hand = st.session_state.env.player_hand # Recupera mano giocatore
        d_hand = st.session_state.env.dealer_hand # Recupera mano banco
        p_score = st.session_state.env.calculate_score(p_hand) # Calcola punti
        
        # Mostra le carte del giocatore
        st.info(f"🃏 **Le tue carte:** {'  |  '.join([format_card(c) for c in p_hand])}")
        st.markdown(f"### Punteggio: `{p_score}`") # Mostra punteggio grande
        
        st.markdown("---") # Linea divisoria
        # Mostra carta banco (la prima scoperta, le altre sarebbero coperte in teoria)
        st.warning(f"🤖 **Banco:** {format_card(d_hand[0])} | ❓ [Coperta]")
        
        # Logica Suggerimento IA
        state = st.session_state.env.get_state() # Ottiene lo stato attuale
        vals = agent.q_table.get(state, [0,0]) # Guarda nella Q-Table cosa fare
        ai_action = "CARTA" if np.argmax(vals) == 1 else "STO BENE" # Traduce 0/1 in testo
        
        st.markdown(f"💡 Consiglio dell'IA: **{ai_action}**") # Mostra consiglio

        # Pulsanti di gioco
        c1, c2 = st.columns(2) # Due colonne per i pulsanti
        if c1.button("➕ CARTA", use_container_width=True): # Pulsante Hit
            _, r, done = st.session_state.env.step(1) # Esegue step con azione 1
            if done: # Se la partita finisce (sballato)
                st.session_state.game_active = False # Disattiva gioco
                st.session_state.msg = "💥 HAI SBALLATO!" # Imposta messaggio sconfitta
            st.rerun() # Ricarica
            
        if c2.button("✋ STO BENE", use_container_width=True): # Pulsante Stand
            _, r, done = st.session_state.env.step(0) # Esegue step con azione 0
            st.session_state.game_active = False # Disattiva gioco
            st.session_state.msg = "🏆 HAI VINTO!" if r == 1 else "🤖 VINCE IL BANCO" # Imposta esito
            st.rerun() # Ricarica

    # Sezione Risultati (visibile a fine partita)
    if not st.session_state.game_active and st.session_state.msg:
        # Se il messaggio contiene "VINTO" usa stile successo, altrimenti errore
        if "VINTO" in st.session_state.msg: 
            st.success(st.session_state.msg)
        else: 
            st.error(st.session_state.msg)
            
            if "SBALLATO" in st.session_state.msg: # Controllo specifico se l'utente ha sballato
                full_hand = st.session_state.env.player_hand # Prende tutta la mano
                last_card = full_hand[-1] # Prende l'ultima carta (quella fatale)
                prev_hand = full_hand[:-1] # Prende la mano precedente (senza l'ultima)
                
                # Calcola i valori per la spiegazione
                prev_score = st.session_state.env.calculate_score(prev_hand) # Punti prima di pescare
                card_val = st.session_state.env.get_card_value(last_card) # Valore carta pescata
                # Se è matta o figura, lo rendiamo leggibile per la stringa
                val_display = 0.5 if (last_card == 'M' or (isinstance(last_card, int) and last_card >=8)) else last_card
                
                final_score = st.session_state.env.calculate_score(full_hand) # Punteggio finale sballato
                
                # Scrive la spiegazione matematica dell'errore
                st.markdown(f"""
                **Analisi della sconfitta:**
                - Punteggio precedente: **{prev_score}**
                - Carta pescata: **{format_card(last_card)}** (Valore: {val_display})
                - Calcolo: {prev_score} + {val_display} = **{final_score}** (> 7.5)
                """)
            # ========================================
        
        st.markdown("---") # Linea divisoria
        full_d_hand = st.session_state.env.dealer_hand # Mano completa del banco
        d_score = st.session_state.env.calculate_score(full_d_hand) # Punteggio banco
        # Mostra le carte finali del banco
        st.write(f"Mano finale Banco: {' | '.join([format_card(c) for c in full_d_hand])} (Tot: {d_score})")

# --- COLONNA DESTRA: TABELLA STRATEGICA ---
with col_stats:
    st.markdown("#### 🧠 Strategia Appresa (Policy)") # Titolo sezione
    st.markdown("La tabella mostra cosa farebbe l'IA per ogni combinazione.") # Descrizione
    
    # Definizione indici righe (Punteggi Giocatore)
    row_indices = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    # Definizione colonne (Carte Banco)
    col_indices = [1, 2, 3, 4, 5, 6, 7, 0.5]

    # Costruzione della matrice dei dati per la tabella
    matrix_data = [] # Lista vuota
    for p in row_indices: # Per ogni punteggio giocatore
        row = [] # Nuova riga
        for d in col_indices: # Per ogni carta banco
            state = (float(p), float(d)) # Crea lo stato corrispondente
            values = agent.q_table.get(state, [0,0]) # Ottiene valori Q
            action = np.argmax(values) # Trova azione migliore (0 o 1)
            row.append("CARTA" if action == 1 else "STO") # Aggiunge testo alla riga
        matrix_data.append(row) # Aggiunge riga alla matrice

    # Creazione DataFrame Pandas
    df_policy = pd.DataFrame(
        matrix_data, 
        index=[f"{x}" for x in row_indices], # Etichette righe
        columns=[f"{x}" if x!=0.5 else "Fig." for x in col_indices] # Etichette colonne (gestisce Fig.)
    )

    # Funzione per colorare le celle in base al contenuto
    def color_policy(val):
        color = '#d4edda' if val == 'CARTA' else '#f8d7da' # Verde chiaro se Carta, Rosso chiaro se Sto
        text_color = '#155724' if val == 'CARTA' else '#721c24' # Testo verde scuro o rosso scuro
        return f'background-color: {color}; color: {text_color}; border: 1px solid white' # Stringa CSS

    st.markdown("**Legenda:** :leaves: Verde = Chiedi Carta | :no_entry_sign: Rosso = Sto Bene") # Legenda visuale
    
    # Rendering della tabella con stile applicato
    st.dataframe(
        df_policy.style.applymap(color_policy), # Applica funzione colore
        height=600, # Altezza fissa in pixel
        use_container_width=True # Adatta alla larghezza colonna
    )
    
    st.caption("Righe: Il tuo punteggio | Colonne: Carta visibile del Banco") # Didascalia finale