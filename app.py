from elevenlabs import clone, generate, play, set_api_key, VOICES_CACHE, voices, VoiceDesign
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import logging  
import gradio as gr
from gradio.components import Text
from gradio.templates import TextArea
from langchain.document_loaders import WhatsAppChatLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import shutil
import nltk
nltk.download('punkt')

persist_directory = 'docs/chroma'

def remove_all_files():
    for file in os.listdir():
        if os.path.isfile(file):
            os.remove(file)
remove_all_files()   

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(f"""The following is a friendly conversation between two friends."""),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


llm = ChatOpenAI(temperature=0, max_tokens=100) # , max_tokens=100, stop=['\n','.']
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)


def get_response(prompt):
   try:
    if os.path.exists(persist_directory):
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        retriever=vectordb.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
            chain_type='map_reduce'
        )
        result = qa({"question": prompt})
        return result['answer']

    _prompt = f"""
        Based on the input provided, Compose a short heartfelt message as a friend to convey feelings of love. 
        Your message should be sincere and authentic, expressing your deep affection and care. 
        You may draw inspiration from shared memories, inside jokes, or qualities you admire in the person. 
        Your message should be personal and tailored to the individual, reflecting your unique friendship and bond. 
        Please ensure that your message is appropriate, respectful, and considerate of the person feelings and relationship dynamics.
        Limit to 100 words.
        input: {prompt}.
        
        """

    response = conversation.predict(input=_prompt)
    return response 
    
   except Exception as e:
      return f"{e}"

set_api_key(os.environ.get("ELEVENLABS_API_KEY"))
def generate_message(message):
    try:
      response_text = get_response(message)
      voice = clone(
          name=f'talk2love',
          files=['custom_audio.mp3'],
      )

      audio = generate(text=response_text, voice=voice, latency=0.75)
      voice.delete()
      audio_path = 'temp.mp3'
      if os.path.exists(audio_path):
        os.remove(audio_path)
      with open(audio_path, 'wb') as f:
            f.write(audio)
        
      return 'temp.mp3'
    
    except Exception as e:
        print(e)
        
        return ""

def save_audio(audio_file_path):
  with open(audio_file_path, "rb") as audio_file:
    audio_data = audio_file.read()
    with open("custom_audio.mp3", "wb") as f:
      f.write(audio_data)
  return "Your voice is temporarily Saved!"

def save_whatsapp_data(whatsapp_file_path):
  try:
    if os.path.exists('docs'):
      shutil.rmtree('docs')
    if os.path.exists('chat.txt'):
      os.remove('chat.txt')
    with open(whatsapp_file_path.name, "rb") as whatsapp_file:
      whatsapp_data = whatsapp_file.read()
      with open("chat.txt", "wb") as f:
        f.write(whatsapp_data)
    loader = WhatsAppChatLoader('chat.txt')
    whatsappdoc = loader.load()
    embeddings = OpenAIEmbeddings()
    if os.path.exists(persist_directory):
        os.remove(persist_directory)
    text_splitter = NLTKTextSplitter()
    chunks = text_splitter.split_documents(whatsappdoc)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    db.persist()
    return f"Your Whatsapp data is temporarily Saved!"
  except Exception as e:
    return f"{e}"


with gr.Blocks() as talk2love:
    gr.Markdown("#Talk2Love <br /> #About: <br /> Talk2Love is a mobile app that allows users to talk to their loved ones even when they are not physically present. The app uses voice cloning technology to create a digital replica of the user's voice, which can then be used to generate new audio recordings. This means that users can conversation with their loved ones, even if they are not present. <br /> Note: Upload Your loved one audio from Love Tab before starting the conversations")
    with gr.Tab("Talk"):
        text_input = gr.inputs.Textbox(label='Your Message',placeholder='Type a message')
        outputs = gr.outputs.Audio(type="filepath")
        send_button = gr.Button("Send")
        send_button.click(generate_message, text_input, outputs)

    with gr.Tab("Love"):
        audio_input = gr.inputs.Audio(type="filepath")
        save_button = gr.Button("Save Audio")
        save_button.click(save_audio, audio_input, outputs=Text())

    with gr.Tab("Whatsapp Data"):
        file_input = gr.File(file_types=['text'])
        save_file_button = gr.Button("Save Data")
        save_file_button.click(save_whatsapp_data, file_input, outputs=Text())

talk2love.launch()