import streamlit as st
import pandas as pd
from test_pipeline import load_model, predict
from model import cosine_sim_output
from youtube import load_youtube

tokenizer, model = load_model()

st.title('음악 추천 서비스')

user_name = st.text_input('사용자의 별명을 입력하세요.', max_chars=5)

user_input = st.text_area('사용자의 글을 입력하세요.')


if st.button('분석'):
    analysis_result = predict(user_input, tokenizer, model)

    if analysis_result:
        name, artist, emo = cosine_sim_output(analysis_result)
        st.success(f'{artist}의 {name}을 추천합니다!')
        video_sd = load_youtube(artist, name)
        st.success(st.video(video_sd))
        emotion = ','.join([e for e in emo])
        st.success(f'{emotion}한 감정!')

       
        st.session_state["like"] = st.checkbox('맘에 들어요')
        st.session_state["dis_like"] = st.checkbox('맘에 안들어요')
        
        while True:
            if st.session_state["like"] or st.session_state["dis_like"]:
                print('yes')
                x = pd.read_csv('Feedback.csv')
                result = pd.DataFrame({'user' : [user_name], 
                                 'score' : [1 if st.session_state["like"] else 0],
                                 'music' : [name],
                                 'artist' : [artist]})
                x = x.append(result, ignore_index=True)
                x.to_csv('Feedback.csv', index=False)
                st.success(f'다시 하시고 싶으시면 checkbox를 해제해주세요!')
                break
    else:
        st.warning('분석 결과를 찾을 수 없습니다.')