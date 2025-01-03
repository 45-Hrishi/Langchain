from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
from langserve import add_routes
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

app = FastAPI(
    title = "Langserve Testing",
    version="0.1",
    description="A API server to test the llama3 and mistral models"
)

# Prompt template
prompt_template = ChatPromptTemplate(
    messages=[
        ("system", "You are a cricket analyst, who gives stats of the mentioned player."),
        ("user", "{player_name}")
    ]
)

# Model configurations
groq_model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.5, max_retries=2)
mistral_model = ChatMistralAI(model="mistral-large-latest", temperature=0, max_retries=2)

class PlayerRequest(BaseModel):
    player_name: str

@app.post("/llama3/invoke")
async def llama3_invoke(request: PlayerRequest):
    # Generate prompt
    prompt = prompt_template.invoke(input={"player_name": request.player_name})
    # Get response from the groq_model
    response = groq_model.invoke(prompt)
    return {"output": response.content}

@app.post("/mistral/invoke")
async def mistral_invoke(request: PlayerRequest):
    # Generate prompt
    prompt = prompt_template.invoke(input={"player_name": request.player_name})
    # Get response from the mistral_model
    response = mistral_model.invoke(prompt)
    return {"output": response.content}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)