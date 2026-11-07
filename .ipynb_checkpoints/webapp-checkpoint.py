import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile

model= joblib.load("Music_Genre_Classifier.pkl")
scaler= joblib.load("scaler.pkl")
st.title("Music Genre Classifier")
st.write("Upload an audio file")

uploaded_file=st.file_uploader("Choose an audio file",type=["wav","mp3"])

def extract_features(file_path):
    audio,sr=librosa.load(file_path,duration=30,sr=22050)
    
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=20)      
    mfcc_mean=np.mean(mfcc,axis=1)
    mfcc_std=np.std(mfcc,axis=1)

    chroma=librosa.feature.chroma_stft(y=audio,sr=sr)        
    chroma_mean=np.mean(chroma,axis=1)
    chroma_std=np.std(chroma,axis=1)

    centroid = librosa.feature.spectral_centroid(y=audio,sr=sr)    
    centroid_mean= np.mean(centroid)
    centroid_std=np.std(centroid)

           
    bandwidth=librosa.feature.spectral_bandwidth(y=audio,sr=sr)      
    bandwidth_mean=np.mean(bandwidth)
    bandwidth_std=np.std(bandwidth)

            
    rolloff=librosa.feature.spectral_rolloff(y=audio,sr=sr)        
    rolloff_mean=np.mean(rolloff)
    rolloff_std=np.std(rolloff)

    rms=librosa.feature.rms(y=audio)                 
    rms_mean=np.mean(rms)
    rms_std=np.std(rms)
 
    tempo,_=librosa.beat.beat_track(y=audio,sr=sr)   
    
          
    feature_vector=np.hstack([mfcc_mean,mfcc_std,chroma_mean,chroma_std,centroid_mean,
    centroid_std,bandwidth_mean,bandwidth_std,rolloff_mean,rolloff_std,rms_mean,rms_std,tempo])

    return feature_vector.reshape(1,-1)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path=tmp.name

    st.audio(uploaded_file)

    if st.button("Predict Genre"):
        feature_vector = extract_features(file_path)
        feature_vector = scaler.transform(feature_vector)

        prediction = model.predict(feature_vector)[0]

        st.success(f"🎶 Predicted Genre: {prediction}")
         