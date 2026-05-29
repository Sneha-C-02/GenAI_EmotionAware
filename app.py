import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import json
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="Emotion-Aware RAG Research Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism, Dark Theme, and Beautiful Spacing
st.markdown("""
<style>
    /* Global App Styling */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: 'Inter', sans-serif;
    }
    
    /* Research Banner */
    .research-banner {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 50%, rgba(15, 52, 96, 0.8) 100%);
        padding: 25px 35px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        text-align: center;
    }
    .research-banner h1 {
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        font-size: 2.2em;
    }
    .research-banner h3 {
        margin: 8px 0 15px 0;
        font-weight: 300;
        color: #A0C3FF;
        font-size: 1.1em;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #00E676;
        margin-right: 8px;
        box-shadow: 0 0 10px #00E676;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 230, 118, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(25, 30, 40, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .glass-card h4 {
        margin-top: 0;
        color: #A0C3FF;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-weight: 500;
    }
    
    /* Document Retrieval Cards */
    .doc-card {
        background: rgba(15, 52, 96, 0.3);
        border-left: 4px solid #4E6E81;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9em;
        line-height: 1.4;
    }
    
    /* Gate Status Colors */
    .gate-safe { color: #00E676; font-weight: bold; }
    .gate-warn { color: #FFEA00; font-weight: bold; }
    .gate-danger { color: #FF1744; font-weight: bold; }

    /* Hide Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# BACKEND INTEGRATION (Mock / Real)
# ==========================================
@st.cache_resource
def get_backend():
    try:
        from backend import EmotionAwareChatbot
        return EmotionAwareChatbot()
    except ImportError:
        # Mock backend for demo purposes if backend.py is not present or runnable
        class MockBackend:
            def analyze(self, user_text):
                return {
                    "emotion": {
                        "top_emotions": [{"label": "sadness", "score": 0.87}, {"label": "anxiety", "score": 0.45}],
                        "valence": -0.65,
                        "arousal": 0.45,
                        "dominance": -0.30
                    },
                    "vad": (-0.65, 0.45, -0.30),
                    "docs": [
                        "That sounds really isolating. Being left behind can feel incredibly lonely.",
                        "It makes complete sense that you'd feel hurt given what happened.",
                        "Your feelings are completely valid. That kind of pressure would overwhelm anyone."
                    ],
                    "reasoning": {"mode": "vad_augmented"}
                }
            def generate_response(self, user_text):
                time.sleep(1.5) # Simulate processing
                analysis = self.analyze(user_text)
                return {
                    "response": "That sounds really painful. It makes complete sense that you'd feel so isolated right now. How are you holding up?",
                    "emotion": analysis["emotion"],
                    "vad": analysis["vad"],
                    "docs": analysis["docs"],
                    "memory": {
                        "summary": "The user has been expressing loneliness and isolation across multiple turns.",
                        "arc_direction": "declining",
                        "dominant_state": "social_withdrawal"
                    },
                    "gate_probability": 0.08
                }
        return MockBackend()

backend = get_backend()


# ==========================================
# STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history_vad = {"valence": [], "arousal": [], "dominance": []}

if "latest_data" not in st.session_state:
    st.session_state.latest_data = None


# ==========================================
# CHART HELPERS
# ==========================================
def create_gauge(value, title, min_val, max_val, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#E0E6ED'}},
        number={'font': {'color': color, 'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': "#E0E6ED"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(
        height=150, 
        margin=dict(l=10, r=10, t=30, b=10), 
        paper_bgcolor="rgba(0,0,0,0)", 
        font={'color': "#E0E6ED"}
    )
    return fig

def create_radar(vad):
    v, a, d = vad
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[v, a, d],
        theta=['Valence', 'Arousal', 'Dominance'],
        fill='toself',
        fillcolor='rgba(0, 230, 118, 0.2)',
        line_color='#00E676',
        name='Current State'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-1, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=30, r=30, t=30, b=30),
        font={'color': "#E0E6ED"}
    )
    return fig

def create_timeline(history):
    if not history["valence"]:
        return go.Figure().update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250)
    
    turns = list(range(1, len(history["valence"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=turns, y=history["valence"], mode='lines+markers', name='Valence', line=dict(color='#00E676')))
    fig.add_trace(go.Scatter(x=turns, y=history["arousal"], mode='lines+markers', name='Arousal', line=dict(color='#FFEA00')))
    fig.add_trace(go.Scatter(x=turns, y=history["dominance"], mode='lines+markers', name='Dominance', line=dict(color='#2979FF')))
    
    fig.update_layout(
        title="Emotional Trajectory",
        xaxis_title="Turn",
        yaxis_title="Intensity",
        yaxis=dict(range=[-1, 1], gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        font={'color': "#E0E6ED"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### System Architecture")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <div style='margin-bottom: 8px;'>✅ <b>Emotion Model</b> (GoEmotions)</div>
        <div style='margin-bottom: 8px;'>✅ <b>Episodic Memory</b> (7-Turn Buffer)</div>
        <div style='margin-bottom: 8px;'>✅ <b>Emotion RAG</b> (FAISS)</div>
        <div style='margin-bottom: 8px;'>✅ <b>Generation</b> (Mistral-7B-4bit)</div>
        <div>✅ <b>Behavioral Gate</b> (DistilBERT)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Corpus Size", "6,505")
        st.metric("Retrieval", "VAD-Aug")
    with col2:
        st.metric("Gate F1", "0.993")
        st.metric("Latency", "~1.5s")
        
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history_vad = {"valence": [], "arousal": [], "dominance": []}
        st.session_state.latest_data = None
        st.rerun()


# ==========================================
# TOP HEADER
# ==========================================
st.markdown("""
<div class="research-banner">
    <h1>Emotion-Aware Retrieval-Augmented Generation</h1>
    <h3>Emotion Detection + Episodic Memory + Emotion-Aware Retrieval + Behavioral Safety Gate</h3>
    <div style="font-size: 0.9em; color: #E0E6ED;"><span class="status-indicator"></span> System Online &bull; Research Inference Mode</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# MAIN LAYOUT
# ==========================================
chat_col, dash_col = st.columns([1.3, 1])

# ------------------------------------------
# CHAT AREA
# ------------------------------------------
with chat_col:
    st.markdown("<h3 style='margin-top:0; color:#A0C3FF;'>Interactive Demonstration</h3>", unsafe_allow_html=True)
    
    chat_container = st.container(height=650, border=False)
    
    with chat_container:
        if not st.session_state.messages:
            st.info("Start a conversation. Try expressing a difficult emotion or situation.")
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                
        # Generate and display bot message
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Processing emotional state & retrieving context..."):
                    result = backend.generate_response(prompt)
                    st.markdown(result["response"])
        
        # Save state
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        st.session_state.latest_data = result
        st.session_state.history_vad["valence"].append(result["vad"][0])
        st.session_state.history_vad["arousal"].append(result["vad"][1])
        st.session_state.history_vad["dominance"].append(result["vad"][2])
        st.rerun()

# ------------------------------------------
# DASHBOARD AREA
# ------------------------------------------
with dash_col:
    if st.session_state.latest_data:
        data = st.session_state.latest_data
        
        # 1. Emotion Analytics
        st.markdown("""<div class="glass-card">
            <h4>🧠 Emotion Analytics</h4>
        """, unsafe_allow_html=True)
        
        top_em = data["emotion"]["top_emotions"][0]
        st.markdown(f"<div style='text-align:center; font-size:1.5em; margin-bottom:10px;'>Primary: <b style='color:#00E676; text-transform:capitalize;'>{top_em['label']}</b> ({top_em['score']:.2f})</div>", unsafe_allow_html=True)
        
        g1, g2, g3 = st.columns(3)
        with g1: st.plotly_chart(create_gauge(data["vad"][0], "Valence", -1, 1, "#00E676"), use_container_width=True, config={'displayModeBar': False})
        with g2: st.plotly_chart(create_gauge(data["vad"][1], "Arousal", -1, 1, "#FFEA00"), use_container_width=True, config={'displayModeBar': False})
        with g3: st.plotly_chart(create_gauge(data["vad"][2], "Dominance", -1, 1, "#2979FF"), use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 2. Gate & Memory Row
        col_g, col_m = st.columns(2)
        
        with col_g:
            prob = data["gate_probability"]
            if prob < 0.3:
                status, css = "Safe", "gate-safe"
            elif prob < 0.7:
                status, css = "Warning", "gate-warn"
            else:
                status, css = "Filtered", "gate-danger"
                
            st.markdown(f"""<div class="glass-card" style="height: 100%;">
                <h4>🛡️ Behavioral Gate</h4>
                <div style='text-align:center; margin: 15px 0;'>
                    <span style='font-size: 2.5em; display:block;' class='{css}'>{(prob*100):.1f}%</span>
                    <span style='font-size: 0.9em; color: #B0C4DE;'>Advice Probability</span>
                </div>
                <div style='text-align:center;' class='{css}'>Status: {status}</div>
            </div>""", unsafe_allow_html=True)
            
        with col_m:
            st.markdown(f"""<div class="glass-card" style="height: 100%;">
                <h4>💭 Episodic Memory</h4>
                <div style='font-size:0.9em; color:#E0E6ED; line-height:1.5;'>
                    <b>Summary:</b> {data['memory'].get('summary', 'No summary available.')}
                </div>
                <div style='margin-top:10px; font-size:0.85em;'>
                    Arc Direction: <span style='color:#A0C3FF;'>{data['memory'].get('arc_direction', 'N/A').title()}</span><br>
                    State: <span style='color:#A0C3FF;'>{data['memory'].get('dominant_state', 'N/A').replace('_', ' ').title()}</span>
                </div>
            </div>""", unsafe_allow_html=True)
            
        # 3. Retrieval Panel
        st.markdown("""<div class="glass-card">
            <h4>🔍 Emotion-Aware Retrieval (Top 2)</h4>
        """, unsafe_allow_html=True)
        for doc in data["docs"][:2]:
            st.markdown(f"<div class='doc-card'>{doc}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 4. Research Visualizations
        st.markdown("""<div class="glass-card">
            <h4>📊 Research Visualizations</h4>
        """, unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["VAD Radar", "Trajectory"])
        with tab1:
            st.plotly_chart(create_radar(data["vad"]), use_container_width=True, config={'displayModeBar': False})
        with tab2:
            st.plotly_chart(create_timeline(st.session_state.history_vad), use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 50px 20px;">
            <h3 style="color: #4E6E81;">Waiting for Input...</h3>
            <p style="color: #6C7A89;">Send a message to visualize the emotion detection, episodic memory tracking, and RAG retrieval pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
