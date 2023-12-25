from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, TextStreamer
from threading import Thread
import transformers
import torch
import re
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os

from colorama import init, Fore, Back, Style
init(autoreset=True)

from vlite_db.main import VLite
from vlite_db.utils import *
import time

db = VLite('vlite_20231126_222045.npz')
print("Vlite has been initialized")

HF_TOKEN = os.environ.get('HUGGING_FACE_TOKEN', default='')

PROMPT = '''You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'.
You only respond once as Assistant. You are allowed to use only the given context below to answer the user's queries, 
and if the answer is not present in the context, say you don't know the answer.
CONTEXT:
'''

MEMORY_PROMPT = """Given a chat history and the latest user question which might reference the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Question : 
"""

st.set_page_config(page_title="RAG Chatbot with Memory", page_icon="🦙", layout="wide")
st.header("RAG Chatbot with Memory🦙")

def render_app():

    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.sidebar.header("Parameters")

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    response_container = st.container()
    container = st.container()

    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = []
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.1
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.9
    if 'max_seq_len' not in st.session_state:
        st.session_state['max_seq_len'] = 512
    if 'pre_prompt' not in st.session_state:
        st.session_state['pre_prompt'] = PROMPT
    if 'string_dialogue' not in st.session_state:
        st.session_state['string_dialogue'] = ''

    st.session_state['temperature'] = st.sidebar.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    st.session_state['top_p'] = st.sidebar.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state['max_seq_len'] = st.sidebar.slider('Max Sequence Length:', min_value=64, max_value=32000, value=4000, step=8)

    NEW_P = st.sidebar.text_area('Prompt before the chat starts. Edit here if desired:', PROMPT, height=60)
    if NEW_P != PROMPT and NEW_P != "" and NEW_P != None:
        st.session_state['pre_prompt'] = NEW_P + "\n\n"
    else:
        st.session_state['pre_prompt'] = PROMPT

    st.session_state['token_usage'] = 0 

    btn_col1, btn_col2 = st.sidebar.columns(2)


    def clear_history():
        st.session_state['chat_dialogue'] = []
    clear_chat_history_button = btn_col1.button("Clear History",
                                            use_container_width=True,
                                            on_click=clear_history)
    

    @st.cache_resource()
    def load_model():
        model_name = "meta-llama/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                use_fast = True,
                                                token = 'enter your hf token here')

        model = AutoModelForCausalLM.from_pretrained(model_name,  
                                                    load_in_4bit=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                                    device_map="auto",
                                                    token = 'enter your hf token here')
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer = tokenizer,
            torch_dtype=torch.float16,
        )
        
        print(Fore.GREEN + "Model is loaded" + Style.RESET_ALL)

        return model, tokenizer, pipeline

    model, tokenizer, pipeline = load_model()
    
    def generate(prompt, max_tokens):
        sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_tokens,
        )
        return sequences[0]['generated_text']

    st.sidebar.write(" ")
    st.sidebar.markdown("*RAG With Memory ❤️*")

    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your question here to talk to LLaMA2"):


        st.session_state.chat_dialogue.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if len(st.session_state['chat_dialogue']) > 1:
                string_dialogue = 'CHAT HISTORY : '
                for dict_message in st.session_state.chat_dialogue:
                    if dict_message["role"] == "user":
                        string_dialogue = string_dialogue + "\n" + "User: " + dict_message["content"] + "\n"
                    else:
                        string_dialogue = string_dialogue + "Assistant: " + dict_message["content"] + "\n"
                
                string_dialogue += MEMORY_PROMPT
                standalone_prompt = generate(string_dialogue, max_tokens = 500)
                question_pattern = re.compile(r'Question :(.+)', re.DOTALL)
                match = question_pattern.search(standalone_prompt)
                print('*****')
                if match:
                    last_question = match.group(1).strip()
                    print(f"Standalone Question : {last_question}")
                else:
                    print("Standalone Question couldnot be generated")
                print('*****')

                extracted_chunks, _ = db.remember(last_question, top_k=3)
                context = ""
                for idx, chunk in enumerate(extracted_chunks):
                    context += f'''\n Chunk {idx} {chunk}'''
                
                string_dialogue = st.session_state['pre_prompt']
                string_dialogue += context
                string_dialogue = string_dialogue + "\n" + "User: " + last_question
                final_prompt = string_dialogue + "Assistant: "


            else:
                extracted_chunks, _ = db.remember(prompt, top_k=3)
                context = ""
                for idx, chunk in enumerate(extracted_chunks):
                    context += f'''\n Chunk {idx} {chunk}'''
                string_dialogue = st.session_state['pre_prompt']
                string_dialogue += context
                string_dialogue = string_dialogue + "\n" + "User: " + st.session_state.chat_dialogue[0]['content']
                final_prompt = string_dialogue + "Assistant: "

            # print('************************************************')
            # print(string_dialogue)
            # print('************************************************')
            
            inputs = tokenizer([final_prompt], return_tensors="pt")
            inputs.to('cuda')
            streamed_output = TextIteratorStreamer(tokenizer)

            generation_kwargs = dict(inputs, 
                                    streamer=streamed_output, 
                                    max_new_tokens = st.session_state['max_seq_len'], 
                                    temperature = st.session_state['temperature'], 
                                    top_p = st.session_state['top_p'],
                                    top_k = 10,
                                    eos_token_id=tokenizer.eos_token_id
                                    )
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            full_response = ""

            for idx, item in enumerate(streamed_output):
                if idx == 0:
                    continue
                
                full_response += item
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.chat_dialogue.append({"role": "assistant", "content": full_response})

        total_tokens_generated = final_prompt + full_response
        st.session_state['token_usage'] += tokenizer(total_tokens_generated, return_tensors="pt")['input_ids'].shape[1]
        st.sidebar.metric(label = "Tokens Used", value = st.session_state['token_usage'])

render_app()