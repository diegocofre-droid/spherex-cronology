"""
SPHEREx GIF cronológico
-----------------------
Pones RA/Dec -> busca imágenes espectrales SPHEREx (QR2) en IRSA que cubren
ese punto -> descarga recortes -> arma un GIF ordenado por fecha.

Instalar:   pip install streamlit astroquery astropy numpy pillow requests
Correr:     streamlit run spherex_gif_app.py
"""

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import streamlit as st
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery.ipac.irsa import Irsa
from PIL import Image, ImageDraw

st.set_page_config(page_title="SPHEREx GIF", page_icon="🔭", layout="centered")
st.title("🔭 SPHEREx: GIF cronológico")
st.caption("Datos públicos de IRSA (colección spherex_qr2). Cada cuadro es una "
           "imagen espectral: cambia el tiempo y también la longitud de onda.")

# ---------- Entradas ----------
c1, c2 = st.columns(2)
ra = c1.number_input("RA (grados)", value=156.09328, format="%.6f")
dec = c2.number_input("Dec (grados)", value=-41.64466, format="%.6f")
size_arcmin = st.slider("Tamaño del recorte (arcmin)", 1.0, 15.0, 6.0, 0.5)
max_frames = st.slider("Máximo de cuadros", 5, 100, 30)
fps = st.slider("Cuadros por segundo", 1, 10, 3)
out_px = st.select_slider("Tamaño del GIF (px)", [256, 384, 512, 640], value=384)
go = st.button("Generar GIF", type="primary")


# ---------- Funciones ----------
@st.cache_data(show_spinner=False)
def buscar(ra, dec):
    coord = SkyCoord(ra, dec, unit="deg")
    tabla = Irsa.query_sia(pos=(coord, 1 * u.arcsec), collection="spherex_qr2")
    return tabla.to_pandas()


def descargar_recorte(args):
    url, ra, dec, size_deg = args
    try:
        r = requests.get(f"{url}?center={ra},{dec}&size={size_deg}", timeout=120)
        r.raise_for_status()
        with fits.open(io.BytesIO(r.content)) as h:
            img = np.array(h["IMAGE"].data, dtype=float)
        return img
    except Exception:
        return None


def a_cuadro(img, etiqueta, out_px):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(img, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    x = np.arcsinh(10 * x) / np.arcsinh(10)  # estiramiento asinh
    im = Image.fromarray((x * 255).astype(np.uint8)).convert("RGB")
    im = im.resize((out_px, out_px), Image.NEAREST)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, out_px, 18], fill=(0, 0, 0))
    d.text((4, 3), etiqueta, fill=(255, 255, 255))
    return im


# ---------- Lógica ----------
if go:
    with st.spinner("Buscando imágenes en IRSA..."):
        try:
            df = buscar(ra, dec)
        except Exception as e:
            st.error(f"Error consultando IRSA: {e}")
            st.stop()

    if df.empty:
        st.warning("No hay imágenes SPHEREx que cubran esas coordenadas (aún).")
        st.stop()

    # Orden cronológico
    if "t_min" in df.columns:
        df = df.sort_values("t_min")
    else:
        df = df.sort_values("obs_id")
    df = df.head(max_frames).reset_index(drop=True)
    st.write(f"{len(df)} imágenes encontradas (mostrando hasta {max_frames}).")

    size_deg = size_arcmin / 60.0
    jobs = [(u_, ra, dec, size_deg) for u_ in df["access_url"]]

    prog = st.progress(0.0, text="Descargando recortes...")
    imgs = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, im in enumerate(ex.map(descargar_recorte, jobs)):
            imgs.append(im)
            prog.progress((i + 1) / len(jobs), text=f"Descargando {i + 1}/{len(jobs)}")
    prog.empty()

    cuadros = []
    for i, img in enumerate(imgs):
        if img is None or img.size == 0:
            continue
        fila = df.iloc[i]
        partes = []
        if "t_min" in df.columns:
            try:
                partes.append(Time(float(fila["t_min"]), format="mjd").iso[:10])
            except Exception:
                pass
        if "em_min" in df.columns and "em_max" in df.columns:
            try:
                w = 0.5 * (float(fila["em_min"]) + float(fila["em_max"]))
                w = w * 1e6 if w < 1e-3 else w  # metros -> micras
                partes.append(f"{w:.2f} um")
            except Exception:
                pass
        cuadros.append(a_cuadro(img, "  ".join(partes) or f"#{i}", out_px))

    if not cuadros:
        st.error("No se pudo descargar ningún recorte. Revisa la conexión o las coordenadas.")
        st.stop()

    buf = io.BytesIO()
    cuadros[0].save(buf, format="GIF", save_all=True, append_images=cuadros[1:],
                    duration=int(1000 / fps), loop=0)
    gif = buf.getvalue()

    st.image(gif)
    st.download_button("Descargar GIF", gif, file_name="spherex.gif", mime="image/gif")
    st.caption(f"{len(cuadros)} cuadros. Cita: DOI 10.26131/IRSA652 (SPHEREx QR2).")
