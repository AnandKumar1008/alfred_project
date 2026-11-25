import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title='Customer Message Router',page_icon='📨',layout='wide')
st.title('Customer message router')
st.caption('BANKING77 · CPU-friendly local inference · uncertain cases are sent to a person')
model_path=Path('models/router.joblib')
if not model_path.exists():
    st.warning('No trained model found. Follow README: download BANKING77 and run the training command.'); st.stop()
router=joblib.load(model_path)
message=st.text_area('Customer message',placeholder='e.g. I do not recognise a cash withdrawal from yesterday…',height=130)
if st.button('Route message',type='primary',disabled=not message.strip()):
    result=router.predict(message)
    st.subheader('Decision')
    st.error('Send to a person' if result['review'] else f"Queue: {result['queue']}")
    st.metric('Top-class confidence',f"{result['confidence']:.1%}")
    st.write('Top class probabilities')
    st.dataframe([{'Category':k,'Probability':f'{v:.1%}'} for k,v in result['probabilities']],use_container_width=True,hide_index=True)
    st.write('Words supporting the top model class')
    if result['explanation']: st.dataframe([{'Token / phrase':k,'Contribution':round(v,4)} for k,v in result['explanation']],use_container_width=True,hide_index=True)
    else: st.info('Token-level explanation unavailable for this estimator.')
    st.caption(f"Top-two margin: {result['margin']:.3f}. Low confidence, ambiguous predictions, or high expected routing cost are escalated.")
