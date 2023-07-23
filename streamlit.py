import streamlit as st
import pandas as pd
import numpy as np 
import os
from test_pipeline import load_model, predict
from model import cosine_sim_output
from youtube import load_youtube

tokenizer, model = load_model()

st.title('음악 추천 서비스')

user_input = st.text_area('사용자의 글을 입력하세요.')

if 'analysis_result' not in st.session_state:
   st.session_state['final_dataframe']= []

if 'music_result' not in st.session_state:
    st.session_state['music_result'] = pd.DataFrame()

if st.button('분석'):
    st.session_state['final_dataframe'] = predict(user_input, tokenizer, model)

    if st.session_state['final_dataframe']:
        st.session_state['music_result'] = cosine_sim_output(st.session_state['final_dataframe'])

        for i in range(3):
            artist = st.session_state['music_result'].iloc[i]['artist']
            name = st.session_state['music_result'].iloc[i]['title']
        
            st.success(f'{artist}의 {name}을 추천합니다!')
            st.video(load_youtube(artist, name))
        #emotion = ','.join([e for e in emo])
        #st.success(f'{emotion}의 감정!')
      
    else:
        st.warning('분석 결과를 찾을 수 없습니다.')
    