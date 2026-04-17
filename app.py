import streamlit as st
import pandas as pd
import torch
import torchvision.transforms as transforms
from nst_utils import image_loader, run_style_transfer
from PIL import Image, ImageOps
import io
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Masterpiece Studio", layout="wide", initial_sidebar_state="expanded")

# --- DATA LOADING ---
@st.cache_data
def load_artist_data():
    csv_path = os.path.join(os.getcwd(), "dataset", "artists.csv")
    df = pd.read_csv(csv_path)
    target_names = [
        "William Turner", "Vincent van Gogh", "El Greco", 
        "Diego Velazquez", "Claude Monet", "Camille Pissarro", "Alfred Sisley"
    ]
    df['display_name'] = df['name'].str.replace('_', ' ')
    return df[df['display_name'].isin(target_names)]

df_artists = load_artist_data()
artist_list = sorted(df_artists['display_name'].tolist())

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'last_result' not in st.session_state: st.session_state.last_result = None
if 'last_content' not in st.session_state: st.session_state.last_content = None
if 'selected_style' not in st.session_state: st.session_state.selected_style = None
if 'steps' not in st.session_state: st.session_state.steps = 150
if 'choice' not in st.session_state: st.session_state.choice = artist_list[0]

# --- CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Open+Sans:wght@400;600&display=swap');
    .stApp { background-color: #0a0e14; color: #e1e7ef; font-family: 'Open Sans', sans-serif; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; min-width: 320px !important; }
    
    div.stButton > button {
        background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 100%) !important;
        color: white !important; border-radius: 10px !important; width: 100% !important;
        font-family: 'Montserrat', sans-serif !important; font-weight: 700 !important;
        text-transform: uppercase !important; border: none !important;
    }
    .artist-card { background: linear-gradient(135deg, #1f6feb 0%, #111d2e 100%); padding: 20px; border-radius: 15px; border-left: 6px solid #58a6ff; margin-bottom: 20px; }
    .data-dashboard { background: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #30363d; margin-top: 10px; margin-bottom: 20px;}
    .data-label { color: #58a6ff; font-weight: bold; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 2px; }
    .data-value { color: #e1e7ef; font-size: 0.9rem; margin-bottom: 8px; }
    .main-title { font-family: 'Montserrat', sans-serif; text-align: center; font-size: 42px; background: linear-gradient(to right, #58a6ff, #bc8cf2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🎨 Style Assets")
selected_artist = st.sidebar.selectbox("Artist Filter", artist_list, index=artist_list.index(st.session_state.choice))
st.session_state.choice = selected_artist

artist_row = df_artists[df_artists['display_name'] == selected_artist].iloc[0]
folder_name = artist_row['name'].replace(" ", "_") 
BASE_PATH = os.path.join(os.getcwd(), "dataset", "style_images", "images", folder_name)

if os.path.exists(BASE_PATH):
    imgs = [f for f in os.listdir(BASE_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:20]
    for img_name in imgs:
        img_path = os.path.join(BASE_PATH, img_name)
        st.sidebar.image(img_path, use_container_width=True)
        if st.sidebar.button(f"APPLY {img_name}", key=f"btn_{img_name}"):
            st.session_state.selected_style = img_path
else:
    st.sidebar.error(f"Folder not found: {folder_name}")

# --- MAIN LAYOUT ---
col_main, col_spacer, col_ctrl = st.columns([2.4, 0.1, 1.1])

with col_main:
    st.markdown('<h1 class="main-title">AI MASTERPIECE STUDIO</h1>', unsafe_allow_html=True)
    tab_canvas, tab_lab, tab_history = st.tabs(["🖌️ Active Canvas", "🔬 Side-by-Side Lab", "⏳ Gallery History"])

    with tab_canvas:
        c1, c2 = st.columns(2)
        content_file = c1.file_uploader("Upload Source", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if content_file: 
            st.session_state.last_content = Image.open(content_file)
            c1.image(content_file, use_container_width=True)
        
        if st.session_state.selected_style:
            c2.image(st.session_state.selected_style, caption="Selected Style", use_container_width=True)

        if content_file and st.session_state.selected_style:
            if st.button("🚀 EXECUTE NEURAL TRANSFER"):
                with st.status("Synthesizing Art...", expanded=True):
                    c_img = image_loader(content_file)
                    s_img = image_loader(st.session_state.selected_style)
                    output = run_style_transfer(c_img, s_img, c_img.clone(), num_steps=st.session_state.steps)
                
                final_art = transforms.ToPILImage()(output.cpu().squeeze(0))
                st.session_state.last_result = final_art
                st.session_state.history.append(final_art)
                st.image(final_art, use_container_width=True)

    with tab_lab:
        if st.session_state.last_result and st.session_state.last_content:
            l1, l2 = st.columns(2)
            l1.image(st.session_state.last_content, caption="Original Content", use_container_width=True)
            l2.image(st.session_state.last_result, caption="AI Masterpiece", use_container_width=True)
        else:
            st.info("Run a transfer on the Active Canvas to see the side-by-side comparison.")

    with tab_history:
        if st.session_state.history:
            st.write("### Previous Generations")
            # Display history in a grid
            h_cols = st.columns(3)
            for idx, img in enumerate(reversed(st.session_state.history)):
                h_cols[idx % 3].image(img, use_container_width=True)
        else:
            st.info("Your generated artwork will appear here.")

with col_ctrl:
    st.markdown('<h3 style="color:#58a6ff; margin-top:0;">ENGINE SETTINGS</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="artist-card"><h4 style="margin:0;">{selected_artist}</h4><p style="font-size:0.85rem; margin-top:10px;">{artist_row["bio"]}</p></div>', unsafe_allow_html=True)

    st.markdown(f'''
        <div class="data-dashboard">
            <div class="data-label">Nationality</div><div class="data-value">{artist_row['nationality']}</div>
            <div class="data-label">Main Genre</div><div class="data-value">{artist_row['genre']}</div>
            <div class="data-label">Lifespan</div><div class="data-value">{artist_row['years']}</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<p style="color:#58a6ff; font-weight:bold; margin-bottom:5px;">TEXTURE SPECIMEN (100X ZOOM)</p>', unsafe_allow_html=True)
    if st.session_state.selected_style:
        style_img = Image.open(st.session_state.selected_style)
        w, h = style_img.size
        zoom_box = (w//2-75, h//2-75, w//2+75, h//2+75)
        zoomed_img = style_img.crop(zoom_box).resize((300, 300), Image.LANCZOS)
        st.image(zoomed_img, use_container_width=True)
    else:
        st.info("Select a style to see texture zoom.")

    st.divider()
    quality = st.radio("Optimization", ["Fast Preview", "Studio Quality"])
    st.session_state.steps = 150 if quality == "Fast Preview" else 350
    st.select_slider("Style Influence", options=[1e4, 1e5, 1e6, 1e7, 1e8], value=1e6)

    if st.session_state.last_result:
        buf = io.BytesIO()
        st.session_state.last_result.save(buf, format="PNG")
        st.download_button(label="💾 Download Result", data=buf.getvalue(), file_name=f"{selected_artist}.png")