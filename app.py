from elevenlabs import clone, generate, play, set_api_key, VOICES_CACHE, voices
from elevenlabs.api import History
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

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(f"""The following is a friendly conversation between two friends."""),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


llm = ChatOpenAI(temperature=0, max_tokens=100) # , max_tokens=100, stop=['\n','.']
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)


def get_response(prompt):
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

    with gr.Tab("WhatsApp Data"):
        gr.Markdown('Under Development: Once it is developed you can talk from your loved one whatsapp chat conversations(chat.txt)')

talk2love.launch()