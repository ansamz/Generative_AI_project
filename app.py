import streamlit as st
from langchain.llms import HuggingFaceHub
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import warnings
warnings.filterwarnings("ignore")

def reset_query():
    if 'query' in st.session_state:
        del st.session_state['query']

st.set_page_config(page_title="Chat",
                   page_icon=":hugging_face:",                   
                   layout="wide",
                   menu_items={
                       'About': "Ask me anything"
                   })

st.markdown("<h1 style='text-align: center; color: red;'>Welcome to our chatbat</h1>", unsafe_allow_html=True)

url = "huggingface.co"

with st.sidebar:
    st.image("imgs/logo.jpg")
    st.markdown(':red[Steps to get HuggingFace Access Token:]')
    st.markdown("- Go to [HuggingFace website](%s)" % url)
    st.markdown('''
                - Click on your profile icon at the top-right corner -> Settings
                - In the sidebar -> Access Tokens
                - Generate a new access token -> new token -> Role:write
                ''')
    model_selection = st.selectbox('Select Model', ('T5 translator', 'Zephyr'))

user_query = st.text_area("Enter your query and click Ctrl+Enter:", key='query', value=st.session_state.get('query', ''))

if model_selection == 'Zephyr':
    if 'api_token' not in st.session_state:
        api_token = st.text_input("Enter your HuggingFace API Token:", type="password")
        if api_token:
            st.session_state.api_token = api_token
    else:
        api_token = st.session_state.api_token

    if api_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token  # Set the environment variable for the API token
    
    if user_query and st.button('Get Response'):
        llm = HuggingFaceHub(
                            repo_id="huggingfaceh4/zephyr-7b-alpha",
                            model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
                            )
        prompt = f"""
        You are an AI assistant that follows instruction extremely well.
        Please be truthful and give direct answers
        </s>
        {user_query}
        </s>

        """
        response = llm.predict(prompt)

        with st.expander('Response from the model:', expanded=True):
            st.write(response)

        if st.button("Reset", on_click=reset_query):
            pass

elif model_selection == 'T5 translator':
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    with st.sidebar:
        language = st.selectbox('Which language?',('English', 'German', 'Arabic', 'Hebrew', 'Polish'))
    
    if user_query and st.button('Get Response'):
        input_text = f"translate English to {language}: {user_query}"

        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=256,  # You can adjust this based on the length of the response you expect
            temperature=0.9,  # Adjusting temperature controls the randomness of the output
            num_return_sequences=1  # Number of output sequences to generate
        )

        response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        with st.expander('Response from the model:', expanded=True):
            st.write(response)

        if st.button("Reset", on_click=reset_query):
            pass
