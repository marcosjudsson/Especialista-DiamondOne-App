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
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# --- 2. CONFIGURAÇÃO E CACHE ---
st.set_page_config(page_title="Especialista DiamondOne", layout="wide")

@st.cache_resource
def carregar_e_processar_dados():
    # ...(código da função sem alterações)...
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
    db = FAISS.from_documents(todos_os_chunks, embeddings)
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# --- 3. DEFINIÇÃO DAS PERSONAS E AVATARES ---
# ...(código das personas sem alterações)...
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
Você é um "Analista de Conhecimento" sênior. Sua tarefa é criar uma definição robusta e otimizada para um termo técnico, baseando-se em múltiplas fontes.
**Processo de 4 Passos:**
1.  **Análise Primária:** Leia a definição inicial fornecida no "Texto para Análise".
2.  **Contexto Interno:** Verifique o "Contexto do Glossário Atual" para ver se o termo já existe ou se há termos relacionados.
3.  **Pesquisa Externa:** Use os "Resultados da Busca na Web" para obter definições alternativas, contexto adicional e exemplos de uso.
4.  **Síntese Final:** Com base em TODAS as fontes, escreva uma única e clara "Definição Otimizada". Esta definição deve ser completa, fácil de entender e estruturada para ser facilmente utilizada por um sistema de IA no futuro. Se as fontes conflitarem, use seu julgamento para criar a melhor definição possível.
**Contexto do Glossário Atual:**
<context>{context}</context>
**Resultados da Busca na Web:**
<web_search_results>{web_search_results}</web_search_results>
**Texto para Análise:** {input}
**Definição Otimizada:**
""")
personas = {
    "Consultor Geral": prompt_template_geral,
    "Estrategista de Marketing": prompt_template_marketing,
    "Analista de Implementação": prompt_template_implementacao,
    "Analista de Conhecimento (Híbrido)": prompt_template_analista
}
avatares = {
    "User": "👤",
    "Consultor Geral": "💬",
    "Estrategista de Marketing": "📈",
    "Analista de Implementação": "🛠️",
    "Analista de Conhecimento (Híbrido)": "🔬"
}

# --- 4. CONSTRUÇÃO DA INTERFACE FINAL V2.5 ---
st.title("🤖 Especialista Virtual DiamondOne V2.5")
st.sidebar.title("Configurações")
modo_selecionado_nome = st.sidebar.selectbox("Selecione a Persona:", options=list(personas.keys()))

st.header(f"Conversando com o {modo_selecionado_nome}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico com nomes e avatares
for message in st.session_state.messages:
    avatar = avatares.get(message["role"], "🤖")
    with st.chat_message(message["role"], avatar=avatar):
        # MUDANÇA AQUI: Exibimos o nome explicitamente
        st.markdown(f'**{message["role"]}**')
        st.markdown(message["content"])

prompt_usuario = st.chat_input("Faça sua pergunta...")

if prompt_usuario:
    st.session_state.messages.append({"role": "User", "content": prompt_usuario})
    with st.chat_message("User", avatar=avatares["User"]):
        # MUDANÇA AQUI: Exibimos o nome "User" explicitamente
        st.markdown(f'**User**')
        st.markdown(prompt_usuario)

    with st.chat_message(modo_selecionado_nome, avatar=avatares[modo_selecionado_nome]):
        # MUDANÇA AQUI: Exibimos o nome da persona explicitamente
        st.markdown(f'**{modo_selecionado_nome}**')
        
        # Cria um placeholder para o efeito de streaming
        placeholder = st.empty()
        resposta_completa = ""
        
        try:
            if os.getenv("GOOGLE_API_KEY") is None or os.getenv("TAVILY_API_KEY") is None:
                st.error("Chaves de API não carregadas! Verifique os segredos no Streamlit Cloud.")
                st.stop()

            retriever = carregar_e_processar_dados()
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, streaming=True)
            prompt_selecionado = personas[modo_selecionado_nome]
            
            def stream_resposta_gerador(stream_chunks):
                for chunk in stream_chunks:
                    content = chunk.get("answer") or chunk.get("text") or (chunk if isinstance(chunk, str) else "")
                    yield content

            if modo_selecionado_nome == "Analista de Conhecimento (Híbrido)":
                search_tool = TavilySearchResults()
                web_results = search_tool.invoke({"query": f"definição de {prompt_usuario} em manufatura"})
                docs_internos = retriever.invoke(prompt_usuario)
                synthesis_chain = create_stuff_documents_chain(llm, prompt_selecionado)
                stream_chunks = synthesis_chain.stream({
                    "input": prompt_usuario, "context": docs_internos, "web_search_results": web_results
                })
                resposta_completa = placeholder.write_stream(stream_resposta_gerador(stream_chunks))
            else:
                document_chain = create_stuff_documents_chain(llm, prompt_selecionado)
                chain = create_retrieval_chain(retriever, document_chain)
                stream_chunks = chain.stream({"input": prompt_usuario})
                resposta_completa = placeholder.write_stream(stream_resposta_gerador(stream_chunks))
            
            # Atualiza o histórico com a resposta final
            st.session_state.messages.append({"role": modo_selecionado_nome, "content": resposta_completa})

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            # Limpa o placeholder em caso de erro
            placeholder.empty()