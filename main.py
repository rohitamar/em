import io
import numpy as np 
from PIL import Image 
from scipy.stats import multivariate_normal as mvn 
import streamlit as st 
import time 

seed = 1010
def init(X, K):
    pi = np.full(K, 1.0 / K)

    rng = np.random.default_rng(seed)
    rand_idx = rng.choice(X.shape[0], size=K, replace=False)
    mu = X[rand_idx].copy()

    var = np.cov(X, rowvar=False)
    var = np.expand_dims(var, axis=0)
    var = np.repeat(var, K, axis=0)

    return pi, mu, var

def e_step(X, K, pi, mu, var):
    N = X.shape[0]
    gamma = np.zeros((N, K))
    mvn_dists = [mvn(mean=mu[i], cov=var[i]) for i in range(K)]
    
    for j in range(K):
        gamma[:, j] = mvn_dists[j].pdf(X) * pi[j]
    
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma 

def m_step(X, K, gamma, mu, var):
    
    N, D = X.shape
    gamma_sum = gamma.sum(axis=0)
    pi_new = gamma_sum / N

    mu_new = (gamma.T @ X) / gamma_sum[:, None]

    var_new = np.zeros_like(var)
    for k in range(K):
        diff = X - mu_new[k]
        g = gamma[:, k][:, None]
        var_new[k] = (g * diff).T @ diff / gamma_sum[k] + 1e-6 * np.eye(D)

    return pi_new, mu_new, var_new

def compute_likelihood(X, pi, mu, var):
    N, K = X.shape[0], pi.shape[0]
    likelihood = np.zeros(N)
    for i in range(K):
        likelihood += pi[i] * mvn(mean=mu[i], cov=var[i]).pdf(X)
    return np.log(likelihood).sum()

def get_quantized(mu, gamma, H, W):
    labels = np.argmax(gamma, axis=1)
    colors = mu * 255
    quantized_img = colors[labels].astype(np.uint8).reshape(H, W, 3)
    return quantized_img

@st.cache_data(show_spinner=False)
def gmm(_X, K, max_iter):
    pi, mu, var = init(_X, K)
    likelihood, state = [], []
    for i in range(max_iter):
        gamma = e_step(_X, K, pi, mu, var)
        pi, mu, var = m_step(_X, K, gamma, mu, var)
        likelihood.append(
            compute_likelihood(_X, pi, mu, var)
        )
        state.append((pi, mu, var, gamma))

    return state

def load_from_path(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 3)
    return pixels, h, w

def load_from_buffer(buf):
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    pixels = arr.reshape(-1, 3)
    return pixels, h, w

if __name__ == '__main__':
    st.title("Image Compression with GMM")

    uploaded = st.file_uploader("Pick an image")
    K = st.slider("Clusters", 2, 32, 8)
    iters = st.slider("Iterations", 1, 50, 15)
    run_btn = st.button("Run")
    
    if 'snapshots' not in st.session_state: 
        st.session_state.snapshots = None 

    if run_btn and uploaded:
        with st.spinner("Running..."):
            X, H, W = load_from_buffer(uploaded.getbuffer())
            state = gmm(X, K, iters)
            st.session_state.snapshots = []
            for (_, mu, _, gamma) in state:
                st.session_state.snapshots.append(
                    get_quantized(mu, gamma, H, W)
                )
            st.session_state.play = False

        st.success("Finished")

    if st.session_state.snapshots is not None:
        placeholder = st.empty()
        progress = st.progress(0.0)
        animate = st.button("Play animation")

        if animate:
            L = len(st.session_state.snapshots) 
            for i, frame in enumerate(st.session_state.snapshots):
                placeholder.image(frame, caption=f"Iteration {i}")
                progress.progress((i + 1) / L)
                time.sleep(0.25)

