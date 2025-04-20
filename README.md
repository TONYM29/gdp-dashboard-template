# WODFACE - Prototype Streamlit App
import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="WODFACE", layout="centered")
st.title("📸 WODFACE - Détection automatique de photos d'athlètes")

st.markdown("""
WODFACE utilise la reconnaissance faciale pour retrouver automatiquement les photos d'un athlète dans une compétition CrossFit. 
**Étapes :**
1. Upload une photo de l'athlète (portrait clair)
2. Upload plusieurs photos de la compétition
3. Clique sur "Analyser" et découvre où apparaît l'athlète !
""")

# Upload de la photo athlète
athlete_img = st.file_uploader("1. Upload la photo de l'athlète", type=['jpg', 'jpeg', 'png'])

# Upload des photos d'événement
event_imgs = st.file_uploader("2. Upload les photos de la compétition (plusieurs)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Lancer l'analyse
if st.button("3. Analyser les photos"):
    if not athlete_img or not event_imgs:
        st.warning("Merci d'uploader une photo de l'athlète et au moins une photo d'événement.")
    else:
        st.info("Analyse en cours... ⏳")

        athlete_ref = face_recognition.load_image_file(athlete_img)
        athlete_encoding = face_recognition.face_encodings(athlete_ref)

        if len(athlete_encoding) == 0:
            st.error("Aucun visage détecté dans la photo de l'athlète. Essaie une autre photo.")
        else:
            athlete_encoding = athlete_encoding[0]
            matched_photos = []

            for img_file in event_imgs:
                img = face_recognition.load_image_file(img_file)
                face_locations = face_recognition.face_locations(img)
                face_encodings = face_recognition.face_encodings(img, face_locations)

                for encoding in face_encodings:
                    match = face_recognition.compare_faces([athlete_encoding], encoding, tolerance=0.5)
                    if match[0]:
                        matched_photos.append(img_file)
                        break

            if matched_photos:
                st.success(f"{len(matched_photos)} photo(s) trouvée(s) avec l'athlète ✅")
                for img in matched_photos:
                    st.image(img, caption="Photo matchée", use_column_width=True)
            else:
                st.warning("Aucune photo ne correspond à l'athlète sur les fichiers uploadés.")
