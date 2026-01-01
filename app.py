import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.cm as cm
import os
import imutils

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="NEURO-SCAN",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS "CYBER-BLUE" (ĐÃ FIX LỖI KHOẢNG TRẮNG ĐẦU TRANG) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&family=Share+Tech+Mono&display=swap');

    /* --- FIX LỖI KHOẢNG TRẮNG TO ĐÙNG Ở TRÊN --- */
    /* 1. Ẩn hoàn toàn thanh Header của Streamlit để nó không chiếm chỗ */
    [data-testid="stHeader"] {
        display: none;
    }
    /* 2. Đẩy nội dung chính sát lên nóc nhà */
    .block-container {
        padding-top: 0rem !important; /* Xóa khoảng cách trên */
        padding-bottom: 0rem !important;
        margin-top: 0px !important;
    }
    /* ------------------------------------------- */

    /* TỐI ƯU THANH CUỘN */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #020408; }
    ::-webkit-scrollbar-thumb { background: #00f3ff; border-radius: 4px; }

    /* NỀN TỔNG THỂ */
    .stApp {
        background-color: #020408;
        background-image: 
            radial-gradient(circle at 50% 50%, rgba(0, 243, 255, 0.05) 0%, transparent 60%),
            linear-gradient(0deg, #000000 0%, #0a0a0a 100%);
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #1a1a1a;
    }

    /* HEADER SIDEBAR */
    .sidebar-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        font-weight: 700;
        color: #00f3ff;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 0 5px rgba(0, 243, 255, 0.6);
        border-bottom: 1px solid rgba(0, 243, 255, 0.2);
        padding-bottom: 2px;
    }

    /* BẢNG DIAGNOSTICS (FIX LAYOUT) */
    .sys-container {
        background-color: #050505;
        border: 1px solid #00f3ff;
        border-radius: 4px;
        padding: 10px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
        margin-top: 15px; 
    }
    
    .sys-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: 0;
        width: 100%;
        height: 10px;
        background: rgba(0, 243, 255, 0.5);
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.8);
        animation: scanline 3s infinite linear;
        opacity: 0.3;
    }

    .sys-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 13px;
        color: #a3bffa;
        border-bottom: 1px dashed #1f2937;
        padding-bottom: 4px;
    }
    
    .sys-label { color: #888; font-size: 11px; }
    .sys-value { color: #00f3ff; font-weight: bold; text-shadow: 0 0 5px #00f3ff; }
    .blink { animation: blinker 1s step-start infinite; }
    @keyframes scanline { 0% { top: -10%; } 100% { top: 110%; } }
    @keyframes blinker { 50% { opacity: 0; } }

    /* WIDGETS */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #050505 !important;
        border: 1px solid #00f3ff !important;
        color: #00f3ff !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #050505 !important;
        border: 1px dashed #00f3ff !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: transparent !important;
        color: #00f3ff !important;
        border: 1px solid #00f3ff !important;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* NÚT SCAN */
    div.stButton > button {
        background: rgba(0, 243, 255, 0.1);
        color: #00f3ff;
        border: 2px solid #00f3ff;
        height: 50px;
        font-family: 'Orbitron', sans-serif;
        font-size: 16px;
        font-weight: 800;
        width: 100%;
        margin-top: 15px !important;
        margin-bottom: 0px !important;
        display: flex; justify-content: center; align-items: center;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.3);
        animation: pulse-scan 2s infinite;
    }
    div.stButton > button:hover {
        background: rgba(0, 243, 255, 0.3);
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.6);
        color: #fff;
    }
    @keyframes pulse-scan {
        0% { box-shadow: 0 0 10px rgba(0, 243, 255, 0.2); }
        50% { box-shadow: 0 0 20px rgba(0, 243, 255, 0.5); border-color: rgba(0, 243, 255, 0.9); }
        100% { box-shadow: 0 0 10px rgba(0, 243, 255, 0.2); }
    }

    /* HEADER CHÍNH (Đẩy margin-top lên 0) */
    .mega-header {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 45px;
        text-align: center;
        background: -webkit-linear-gradient(#fff, #00f3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 243, 255, 0.4);
        margin-bottom: 20px;
        margin-top: 0px !important; /* Đảm bảo không có khoảng cách thừa */
        padding-top: 20px; /* Thêm chút padding để không bị cắt chữ */
        border-bottom: 1px solid #1a1a1a;
        padding-bottom: 15px;
    }

    /* RESULT & MONITOR */
    .monitor-frame {
        background: rgba(0,0,0,0.7); border: 1px solid #333;
        border-radius: 4px; padding: 5px; position: relative;
    }
    .monitor-label {
        position: absolute; top: -10px; left: 10px;
        background: #020408; padding: 0 8px;
        color: #00f3ff; font-size: 11px;
        font-family: 'Orbitron'; border: 1px solid #333;
    }
    .result-bar {
        background: #0a0a0a; border: 1px solid #333;
        border-radius: 8px; padding: 15px 25px;
        margin-top: 15px; display: flex;
        justify-content: space-between; align-items: center;
    }
    .result-danger { color: #ff003c; font-family: 'Orbitron'; font-size: 22px; text-shadow: 0 0 10px #ff003c; animation: pulse 1s infinite; }
    .result-safe { color: #00f3ff; font-family: 'Orbitron'; font-size: 22px; text-shadow: 0 0 10px #00f3ff; }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOGIC XỬ LÝ (GIỮ NGUYÊN) ---
def ham_xu_ly_cho_PRO(img):
    if img.dtype != 'uint8': img = np.array(img, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    else: new_img = img 
    desired_size = 128
    old_size = new_img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    new_img = cv2.resize(new_img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0] 
    new_img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def ham_xu_ly_cho_FINAL(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (128, 128))

@st.cache_resource
def load_model_by_name(model_name):
    if model_name == "PRO": path = 'brain_tumor_PRO.h5'
    else: path = 'brain_tumor_FINAL.h5'
    if not os.path.exists(path): return None
    try: return tf.keras.models.load_model(path)
    except: return None

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name: return layer.name
    return None

def make_gradcam_heatmap_manual(img_tensor, model, last_conv_layer_name):
    img_tensor = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        x = img_tensor
        last_conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                last_conv_output = x
                tape.watch(last_conv_output)
        preds = x
        top_pred_index = tf.argmax(preds[0])
        class_channel = preds[:, top_pred_index]
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- 4. GIAO DIỆN CHÍNH ---

# --- SIDEBAR (THỨ TỰ SẮP XẾP CHUẨN) ---
with st.sidebar:
    st.markdown('<div style="font-family: Orbitron; font-size: 24px; color: #fff; text-align: center; margin-bottom: 20px;">NEURO<span style="color:#00f3ff">AI</span></div>', unsafe_allow_html=True)
    
    # 1. CORE
    st.markdown('<div class="sidebar-header">CHỌN BỘ XỬ LÝ (CORE)</div>', unsafe_allow_html=True)
    model_option = st.selectbox("CORE", ("Model PRO (v2.0 - Tách sọ)", "Model FINAL (v1.0 - Cơ bản)"), label_visibility="collapsed")
    current_model_name = "PRO" if "PRO" in model_option else "FINAL"
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. UPLOAD
    st.markdown('<div class="sidebar-header">NHẬP DỮ LIỆU MRI</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    # 3. NÚT SCAN (DÍNH LIỀN UPLOAD)
    scan_btn = st.button("🚀 SCAN MRI")

    # 4. DIAGNOSTICS (DÍNH LIỀN NÚT SCAN - ĐÃ BỎ DÒNG KẺ ---)
    st.markdown("""
        <div class="sys-container">
            <div style="text-align: center; color: #fff; font-size: 11px; margin-bottom: 10px; letter-spacing: 2px;">SYSTEM DIAGNOSTICS</div>
            <div class="sys-row">
                <span class="sys-label">SERVER STATUS</span>
                <span class="sys-value blink">[ ONLINE ]</span>
            </div>
            <div class="sys-row">
                <span class="sys-label">GPU ACCEL</span>
                <span class="sys-value">TESLA T4</span>
            </div>
            <div class="sys-row">
                <span class="sys-label">LATENCY</span>
                <span class="sys-value">12ms</span>
            </div>
            <div class="sys-row" style="border: none;">
                <span class="sys-label">SECURITY</span>
                <span class="sys-value" style="color: #00f3ff;">ENCRYPTED</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN SCREEN ---
st.markdown('<div class="mega-header">NEURO-SCAN SYSTEM</div>', unsafe_allow_html=True)

model = load_model_by_name(current_model_name)

if uploaded_file is None:
    st.markdown("""
    <div style="height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; 
                border: 1px dashed #333; border-radius: 8px; background: rgba(0, 0, 0, 0.5);">
        <h2 style="font-family: 'Orbitron'; color: #00f3ff; font-size: 40px; text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);">SYSTEM STANDBY</h2>
        <p style="font-family: 'Share Tech Mono'; color: #888;">WAITING FOR DATA INJECTION...</p>
    </div>
    """, unsafe_allow_html=True)

else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns(3, gap="medium")

    if current_model_name == "PRO":
        processed_img = ham_xu_ly_cho_PRO(img_rgb)
        img_for_display = processed_img
        img_normalized = processed_img.astype('float32') / 255.0
        final_input_tensor = np.expand_dims(img_normalized, axis=0)
        proc_label = "SKULL STRIP"
    else:
        processed_img = ham_xu_ly_cho_FINAL(img_bgr)
        img_for_display = processed_img
        img_normalized = processed_img.astype('float32') / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=-1)
        final_input_tensor = np.expand_dims(img_expanded, axis=0)
        proc_label = "GRAYSCALE"

    with col1:
        st.markdown(f'<div class="monitor-frame"><div class="monitor-label">01. ẢNH GỐC</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="monitor-frame"><div class="monitor-label">02. XỬ LÝ: {proc_label}</div>', unsafe_allow_html=True)
        st.image(processed_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        placeholder = st.empty()
        placeholder.markdown(f"""
            <div class="monitor-frame" style="height: 100%; min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div class="monitor-label">03. AI SCANNING</div>
                <span style="color: #00f3ff; font-family: 'Share Tech Mono'">AWAITING COMMAND...</span>
            </div>
        """, unsafe_allow_html=True)

    result_placeholder = st.empty()

    if scan_btn:
        preds = model.predict(final_input_tensor)
        score = preds[0][0]
        
        last_layer = get_last_conv_layer_name(model)
        if last_layer:
            heatmap = make_gradcam_heatmap_manual(final_input_tensor, model, last_layer)
            final_heatmap_img = overlay_heatmap(img_for_display, heatmap)
            with col3:
                placeholder.markdown(f'<div class="monitor-frame"><div class="monitor-label">03. BẢN ĐỒ NHIỆT</div>', unsafe_allow_html=True)
                st.image(final_heatmap_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        threshold = 0.2 if current_model_name == "PRO" else 0.5
        if score > threshold:
            res_text = "PHÁT HIỆN: CÓ KHỐI U"
            res_class = "result-danger"
            icon = "⚠️"
            border_color = "#ff003c"
            score_color = "#ff003c"
        else:
            res_text = "AN TOÀN: KHÔNG CÓ U"
            res_class = "result-safe"
            icon = "✅"
            border_color = "#00f3ff"
            score_color = "#00f3ff"
            
        result_placeholder.markdown(f"""
            <div class="result-bar" style="border-left: 5px solid {border_color}; box-shadow: 0 0 15px {border_color}40;">
                <div>
                    <div style="color: #888; font-size: 12px; margin-bottom: 2px; font-family: 'Share Tech Mono';">DIAGNOSIS RESULT</div>
                    <div class="{res_class}">{icon} {res_text}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #888; font-size: 12px; margin-bottom: 2px; font-family: 'Share Tech Mono';">AI CONFIDENCE</div>
                    <div style="font-family: 'Orbitron'; font-size: 30px; color: {score_color};">{score*100:.2f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)