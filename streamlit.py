import streamlit as st
import pandas as pd
import numpy as np 
import os
from test_pipeline import load_model, predict
from model import cosine_sim_output
from youtube import load_youtube

tokenizer, model = load_model()

st.title('감정 기반 음악 추천 서비스')

user_input = st.text_area('사용자의 글을 입력하세요.(감정이 담겨 있으면 더 좋아요!)')

if 'analysis_result' not in st.session_state:
   st.session_state['final_dataframe']= []

if 'music_result' not in st.session_state:
    st.session_state['music_result'] = pd.DataFrame()

if st.button('분석'):
    st.session_state['final_dataframe'] = predict(user_input, tokenizer, model)

    if st.session_state['final_dataframe']:
        st.session_state['music_result'] = cosine_sim_output(st.session_state['final_dataframe'])

        emo = []
        for k, j in enumerate(st.session_state['music_result'].iloc[0][3::]):
            if j > 0.5:
                x = st.session_state['music_result'].columns[k+3]
                emo.append(x)
        emotion = ','.join([e for e in emo])
        if emo:
            st.success(f'{emotion}의 감정!')
        else:
            st.success(f'이렇다 할 감정이 없어요')

        for i in range(3):
            artist = st.session_state['music_result'].iloc[i]['artist']
            name = st.session_state['music_result'].iloc[i]['title']
        
            st.success(f'{artist}의 {name}을 추천합니다!')
            st.video(load_youtube(artist, name))
      
    else:
        st.warning('분석 결과를 찾을 수 없습니다.')
    