# --- 1. IMPORTAÇÕES COMPLETAS ---
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- MUDANÇA NA IMPORTAÇÃO DO BANCO DE VETORES ---
from langchain_community.vectorstores import FAISS

# Carrega as variáveis do arquivo .env
load_dotenv()

# --- 2. CONFIGURAÇÃO DA PÁGINA E CACHE ---
st.set_page_config(page_title="Especialista DiamondOne", layout="wide")

@st.cache_resource
def carregar_e_processar_dados():
    caminho_dados = "dados/"
    documentos_padrao = []
    
    for nome_arquivo in os.listdir(caminho_dados):
        if nome_arquivo == "GlossarioDiamondone.txt": continue
        caminho_completo = os.path.join(caminho_dados, nome_arquivo)
        try:
            if nome_arquivo.endswith(".pdf"): loader = PyPDFLoader(caminho_completo)
            elif nome_arquivo.endswith(".docx"): loader = Docx2txtLoader(caminho_completo)
            documentos_padrao.extend(loader.load())
        except Exception as e: print(f"Erro ao carregar {nome_arquivo}: {e}")

    text_splitter_padrao = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks_padrao = text_splitter_padrao.split_documents(documentos_padrao)

    chunks_glossario = []
    caminho_glossario = os.path.join(caminho_dados, "GlossarioDiamondone.txt")
    try:
        with open(caminho_glossario, 'r', encoding='utf-8') as f:
            conteudo_glossario = f.read()
        entradas_glossario = conteudo_glossario.strip().split('\n\n')
        for entrada in entradas_glossario:
            chunks_glossario.append(Document(page_content=entrada, metadata={"source": "GlossarioDiamondone.txt"}))
    except Exception as e: print(f"Erro ao processar glossário: {e}")

    todos_os_chunks = chunks_padrao + chunks_glossario
    embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    
    # --- MUDANÇA NA CRIAÇÃO DO BANCO DE VETORES ---
    db = FAISS.from_documents(todos_os_chunks, embeddings)
    
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# --- 3. DEFINIÇÃO DAS PERSONAS ---
prompt_template_geral = ChatPromptTemplate.from_template("""
Você é um consultor especialista no sistema DiamondOne para indústrias de manufatura. Sua tarefa é responder à pergunta do usuário de forma clara, profissional e objetiva. Baseie sua resposta estritamente no seguinte contexto extraído da documentação:
<context>{context}</context>
Pergunta: {input}
""")
prompt_template_marketing = ChatPromptTemplate.from_template("""
Você é o "Marketer DiamondOne", um especialista em marketing de produto B2B para a indústria de manufatura. Sua missão é traduzir características técnicas em benefícios de negócio claros e atraentes, usando uma linguagem persuasiva e autêntica. Para estabelecer credibilidade, incorpore termos do léxico da indústria de forma natural em sua comunicação.
<context>{context}</context>
Tarefa de Marketing: {input}
""")
prompt_template_implementacao = ChatPromptTemplate.from_template("""
Você é um Analista de Implementação Sênior do DiamondOne. Sua tarefa é fornecer respostas técnicas, precisas e em formato de passo-a-passo ou lista, quando apropriado. Seja direto e foque nos detalhes técnicos da implementação, evitando linguagem de marketing ou opiniões. Baseie sua resposta estritamente no seguinte contexto extraído da documentação:
<context>{context}</context>
Pergunta Técnica: {input}
""")
prompt_template_analista = ChatPromptTemplate.from_template("""
Você é um "Analista de Conhecimento" especializado na indústria de manufatura e no sistema DiamondOne.
Sua tarefa é analisar o "Texto para Análise" fornecido e compará-lo com o "Contexto do Glossário Atual".
Sua missão é identificar e extrair apenas os termos, siglas ou jargões técnicos do "Texto para Análise" que AINDA NÃO ESTÃO no glossário.
Apresente os novos termos em uma lista simples, com uma breve definição baseada no texto. Se nenhum termo novo for encontrado, simplesmente responda "Nenhum termo novo encontrado".

Contexto do Glossário Atual:
<context>
{context}
</context>

Texto para Análise: {input}

Novos Termos Sugeridos:
""")

personas = {
    "Consultor Geral": prompt_template_geral,
    "Estrategista de Marketing": prompt_template_marketing,
    "Analista de Implementação": prompt_template_implementacao,
    "Analista de Conhecimento (Beta)": prompt_template_analista
}

# --- 4. CONSTRUÇÃO DA INTERFACE (com lógica condicional) ---
st.title("🤖 Especialista Virtual DiamondOne")
st.caption("Desenvolvido com a mentoria do CriAi")

st.sidebar.title("Configurações")
modo_selecionado_nome = st.sidebar.selectbox("Selecione a Persona:", options=list(personas.keys()))
prompt_selecionado = personas[modo_selecionado_nome]

st.header(f"Conversando com o {modo_selecionado_nome}")

# --- LÓGICA DE INTERFACE CUSTOMIZADA ---
if modo_selecionado_nome == "Analista de Conhecimento (Beta)":
    st.info("Cole abaixo um artigo, e-mail ou qualquer texto para que o especialista sugira novos termos para o nosso glossário.")
    pergunta_usuario = st.text_area("Texto para análise:", height=300)
else:
    pergunta_usuario = st.text_input("Faça sua pergunta ou descreva a tarefa:")
# ----------------------------------------

if pergunta_usuario:
    with st.spinner("Processando..."):
        try:
            if os.getenv("GOOGLE_API_KEY") is None:
                st.error("Chave de API do Google não carregada!")
                st.stop()

            retriever = carregar_e_processar_dados()
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
            document_chain = create_stuff_documents_chain(llm, prompt_selecionado)
            chain = create_retrieval_chain(retriever, document_chain)
            response = chain.invoke({"input": pergunta_usuario})
            
            st.success("Resposta Recebida!")
            st.subheader("Resposta do Especialista:")
            st.markdown(response["answer"])

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")