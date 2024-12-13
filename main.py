from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar FastAPI
app = FastAPI()

# Classes para validação de dados
class CrewAIRequest(BaseModel):
    query: str
    tools: List[str] = []
    website: Optional[str] = None

# Rota padrão
@app.get("/")
async def root():
    return {"message": "CrewAI API está funcionando!"}

# Rota para busca com CrewAI
@app.post("/api/search")
async def search_with_crewai(request: CrewAIRequest):
    try:
        # Configurar ferramenta de busca
        serper_tool = SerperDevTool()
        
        # Criar agente
        agent = Agent(
            role="Pesquisador Web",
            goal=f"Pesquisar informações sobre: {request.query}",
            tools=[serper_tool],
            verbose=True
        )
        
        # Criar tarefa
        task = Task(
            description=f"Pesquise e analise: {request.query}",
            agent=agent
        )
        
        # Executar crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        return {"success": True, "result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rota para verificar status
@app.get("/health")
async def health_check():
    return {"status": "online"}
