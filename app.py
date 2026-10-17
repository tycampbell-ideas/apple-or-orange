import pathlib, platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        import pathlib._local
        pathlib._local.PosixPath = pathlib.WindowsPath   # needed on Python 3.13+
    except ImportError:
        pass

import streamlit as st 
from fastai.vision.all import * 

st.title("Is it an apple or an orange??")

st.caption("Upload an apple or orange to see if the model guesses correctly!")

@st.cache_resource

def get_learner():
    model_path = Path(__file__).parent / 'model.pkl'
    return load_learner(model_path)

learn = get_learner()  

uploaded = st.file_uploader("Upload your image", type = ['jpg', 'jpeg', 'png', 'webp'])

if uploaded is not None: 
    img = PILImage.create(uploaded.getvalue())
    st.image(img, width = 300)
    pred, idx, probs = learn.predict(img)
    st.subheader(f"Prediction: {pred}")
    for cat, p in zip(learn.dls.vocab, probs):
        st.progress(float(p), text = f"{cat}: {float(p):.1%}")