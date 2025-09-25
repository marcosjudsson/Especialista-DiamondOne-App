# --- 1. IMPORTAÇÕES COMPLETAS ---
import streamlit as st
import os
from dotenv import load_dotenv
from sqlalchemy import text
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

# Carrega as variáveis do arquivo .env
load_dotenv()

# --- 2. GERENCIAMENTO DO BANCO DE DADOS DE PERSONAS (SQLITE) ---
conn = st.connection('personas_db', type='sql', url='sqlite:///personas.db')

def criar_tabela_personas():
    with conn.session as s:
        s.execute(text('CREATE TABLE IF NOT EXISTS personas (nome TEXT PRIMARY KEY, prompt TEXT);'))
        s.commit()

@st.cache_data(ttl=3600)
def buscar_personas():
    df = conn.query('SELECT nome, prompt FROM personas', ttl=600)
    if df.empty:
        return {}
    return {row['nome']: row['prompt'] for index, row in df.iterrows()}

def atualizar_persona(nome, prompt):
    with conn.session as s:
        s.execute(text('UPDATE personas SET prompt = :prompt WHERE nome = :nome;'), params=dict(prompt=prompt, nome=nome))
        s.commit()
    st.cache_data.clear()

def criar_nova_persona(nome, prompt):
    with conn.session as s:
        s.execute(text('INSERT INTO personas (nome, prompt) VALUES (:nome, :prompt);'), params=dict(nome=nome, prompt=prompt))
        s.commit()
    st.cache_data.clear()

def deletar_persona(nome):
    with conn.session as s:
        s.execute(text('DELETE FROM personas WHERE nome = :nome;'), params=dict(nome=nome))
        s.commit()
    st.cache_data.clear()

# --- 3. CONFIGURAÇÃO DO BACKEND DE RAG ---
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
    db = FAISS.from_documents(todos_os_chunks, embeddings)
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# --- 4. DADOS PADRÃO E AVATARES ---
PERSONAS_PADRAO = {
    "Consultor Geral": """Você é um consultor especialista no sistema DiamondOne para indústrias de manufatura. Sua tarefa é responder à pergunta do usuário de forma clara, profissional e objetiva. Baseie sua resposta estritamente no seguinte contexto extraído da documentação:\n<context>{context}</context>\nPergunta: {input}""",
    "Estrategista de Marketing": """Você é o "Marketer DiamondOne", um especialista em marketing de produto B2B para a indústria de manufatura. Sua missão é traduzir características técnicas em benefícios de negócio claros e atraentes, usando uma linguagem persuasiva e autêntica. Para estabelecer credibilidade, incorpore termos do léxico da indústria de forma natural em sua comunicação.\n<context>{context}</context>\nTarefa de Marketing: {input}""",
    "Analista de Implementação": """Você é um Analista de Implementação Sênior do DiamondOne. Sua tarefa é fornecer respostas técnicas, precisas e em formato de passo-a-passo ou lista, quando apropriado. Seja direto e foque nos detalhes técnicos da implementação, evitando linguagem de marketing ou opiniões. Baseie sua resposta estritamente no seguinte contexto extraído da documentação:\n<context>{context}</context>\nPergunta Técnica: {input}""",
    "Analista de Conhecimento (Híbrido)": """Você é um "Analista de Conhecimento" sênior. Sua tarefa é criar uma definição robusta e otimizada para um termo técnico, baseando-se em múltiplas fontes.\n**Processo de 4 Passos:**\n1.  **Análise Primária:** Leia a definição inicial fornecida no "Texto para Análise".\n2.  **Contexto Interno:** Verifique o "Contexto do Glossário Atual" para ver se o termo já existe ou se há termos relacionados.\n3.  **Pesquisa Externa:** Use os "Resultados da Busca na Web" para obter definições alternativas, contexto adicional e exemplos de uso.\n4.  **Síntese Final:** Com base em TODAS as fontes, escreva uma única e clara "Definição Otimizada". Esta definição deve ser completa, fácil de entender e estruturada para ser facilmente utilizada por um sistema de IA no futuro. Se as fontes conflitarem, use seu julgamento para criar a melhor definição possível.\n**Contexto do Glossário Atual:**\n<context>{context}</context>\n**Resultados da Busca na Web:**\n<web_search_results>{web_search_results}</web_search_results>\n**Texto para Análise:** {input}\n**Definição Otimizada:**"""
}
AVATARES = {
    "human": "👤", "ai": "🤖", "Consultor Geral": "💬", "Estrategista de Marketing": "📈",
    "Analista de Implementação": "🛠️", "Analista de Conhecimento (Híbrido)": "🔬"
}

# --- 5. APLICAÇÃO PRINCIPAL ---
criar_tabela_personas()
st.title("🤖 Especialista Virtual DiamondOne V2.9")

st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Selecione uma página:", ["Chat com Especialista", "Gerenciador de Personas"])

personas_db = buscar_personas()

if pagina == "Chat com Especialista":
    st.sidebar.title("Configurações do Chat")
    
    if not personas_db:
        st.warning("Nenhuma persona encontrada. Vá para o 'Gerenciador de Personas' para criar sua primeira persona.")
        st.stop()
        
    modo_selecionado_nome = st.sidebar.selectbox("Selecione a Persona:", options=list(personas_db.keys()))
    st.header(f"Conversando com o {modo_selecionado_nome}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    
    if modo_selecionado_nome not in st.session_state.chat_history:
        st.session_state.chat_history[modo_selecionado_nome] = []

    for message in st.session_state.chat_history[modo_selecionado_nome]:
        avatar = AVATARES.get(message.type, AVATARES.get(modo_selecionado_nome))
        with st.chat_message(message.type, avatar=avatar):
            st.markdown(f'**{modo_selecionado_nome if message.type == "ai" else "User"}**')
            st.markdown(message.content)

    if prompt_usuario := st.chat_input("Faça sua pergunta..."):
        st.session_state.chat_history[modo_selecionado_nome].append(HumanMessage(content=prompt_usuario))
        with st.chat_message("human", avatar=AVATARES["human"]):
            st.markdown(f'**User**'); st.markdown(prompt_usuario)

        with st.chat_message("ai", avatar=AVATARES.get(modo_selecionado_nome)):
            st.markdown(f'**{modo_selecionado_nome}**')
            placeholder = st.empty()
            try:
                if os.getenv("GOOGLE_API_KEY") is None: 
                    st.error("Chave de API do Google não carregada!"); st.stop()
                
                retriever = carregar_e_processar_dados()
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.5, streaming=True)
                
                prompt_texto_selecionado = personas_db[modo_selecionado_nome]
                
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Dada a conversa abaixo e a última pergunta do usuário, formule uma pergunta autônoma que possa ser entendida sem o histórico da conversa. Não responda à pergunta, apenas a reformule se necessário, caso contrário, retorne-a como está."),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
                
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", prompt_texto_selecionado),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                ])
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                chat_history_langchain = st.session_state.chat_history[modo_selecionado_nome][:-1]
                
                # CORREÇÃO FINAL AQUI
                def stream_resposta_gerador(stream_chunks):
                    for chunk in stream_chunks:
                        # Procura pela chave 'answer' e só retorna o conteúdo se ela existir
                        if answer_chunk := chunk.get("answer"):
                            yield answer_chunk

                stream_chunks = rag_chain.stream({"input": prompt_usuario, "chat_history": chat_history_langchain})
                resposta_completa = placeholder.write_stream(stream_resposta_gerador(stream_chunks))
                
                st.session_state.chat_history[modo_selecionado_nome].append(AIMessage(content=resposta_completa))
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}"); placeholder.empty()

elif pagina == "Gerenciador de Personas":
    st.header("Gerenciador de Personas")
    st.info("Crie, edite e delete as personalidades do seu especialista de IA.")
    
    with st.expander("➕ Criar Nova Persona"):
        with st.form("nova_persona_form"):
            novo_nome = st.text_input("Nome da Nova Persona (ex: Suporte Técnico)")
            novo_prompt = st.text_area("Prompt da Nova Persona:", height=200)
            submitted = st.form_submit_button("Criar Persona")
            if submitted:
                if novo_nome and novo_prompt:
                    criar_nova_persona(novo_nome, novo_prompt)
                    st.success(f"Persona '{novo_nome}' criada com sucesso!")
                    st.rerun()
                else:
                    st.error("Por favor, preencha o nome e o prompt.")

    st.divider()
    
    personas_editor_df = conn.query('SELECT nome, prompt FROM personas', ttl=10)
    
    if personas_editor_df.empty:
        st.warning("O banco de dados de personas está vazio.")
        if st.button("Criar Personas Padrão"):
            with conn.session as s:
                for nome, prompt_texto in PERSONAS_PADRAO.items():
                    s.execute(text('INSERT OR REPLACE INTO personas (nome, prompt) VALUES (:nome, :prompt);'), params=dict(nome=nome, prompt=prompt_texto))
                s.commit()
            st.success("Personas padrão criadas com sucesso!")
            st.rerun()
    else:
        st.subheader("Editar Personas Existentes")
        lista_nomes_personas = personas_editor_df['nome'].tolist()
        persona_para_editar = st.selectbox("Selecione uma persona para editar:", options=lista_nomes_personas)
        
        prompt_atual = personas_editor_df[personas_editor_df['nome'] == persona_para_editar]['prompt'].iloc[0]
        
        prompt_editado = st.text_area("Edite o prompt:", value=prompt_atual, height=300, key=f"editor_{persona_para_editar}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Salvar Alterações"):
                atualizar_persona(persona_para_editar, prompt_editado)
                st.success(f"Persona '{persona_para_editar}' atualizada!")
        with col2:
            if st.button("❌ Deletar Persona", type="primary"):
                deletar_persona(persona_para_editar)
                st.success(f"Persona '{persona_para_editar}' deletada!")
                st.rerun()