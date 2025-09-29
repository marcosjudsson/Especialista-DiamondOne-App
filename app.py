# --- 1. IMPORTAÇÕES COMPLETAS ---
import streamlit as st
import os
import time
from dotenv import load_dotenv
from sqlalchemy import text
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from github import Github
from github.GithubException import UnknownObjectException

# Carrega as variáveis do arquivo .env para desenvolvimento local
load_dotenv()

# --- 2. GERENCIAMENTO DO BANCO DE DADOS LOCAL (SQLITE) ---
conn = st.connection('personas_db', type='sql', url='sqlite:///personas.db')

def criar_tabela_personas():
    with conn.session as s:
        s.execute(text('CREATE TABLE IF NOT EXISTS personas (nome TEXT PRIMARY KEY, prompt TEXT);'))
        s.commit()

@st.cache_data(ttl=600)
def buscar_personas():
    df = conn.query('SELECT nome, prompt FROM personas', ttl=600)
    if df.empty: return {}
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
        if nome_arquivo == ".gitkeep": continue
        caminho_completo = os.path.join(caminho_dados, nome_arquivo)
        try:
            if nome_arquivo.endswith(".pdf"): loader = PyPDFLoader(caminho_completo)
            elif nome_arquivo.endswith(".docx"): loader = Docx2txtLoader(caminho_completo)
            elif nome_arquivo.endswith(".txt"): loader = TextLoader(caminho_completo, encoding='utf-8')
            documentos_padrao.extend(loader.load())
        except Exception as e: print(f"Erro ao carregar {nome_arquivo}: {e}")
    text_splitter_padrao = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks_padrao = text_splitter_padrao.split_documents(documentos_padrao)
    embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    db = FAISS.from_documents(chunks_padrao, embeddings)
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# --- 4. FUNÇÕES DE INTEGRAÇÃO COM O GITHUB (CORRIGIDA) ---
@st.cache_resource
def get_github_repo():
    # Tenta carregar do .env primeiro (para ambiente local)
    token = os.getenv("GITHUB_TOKEN")
    repo_nome = os.getenv("GITHUB_REPO")
    
    # Se não encontrar, tenta carregar dos segredos do Streamlit (para a nuvem)
    if not token:
        try:
            token = st.secrets.get("GITHUB_TOKEN")
            repo_nome = st.secrets.get("GITHUB_REPO")
        except Exception:
             return None # Retorna None se st.secrets não existir ou falhar

    if not token or not repo_nome: return None
    g = Github(token)
    return g.get_repo(repo_nome)

@st.cache_data(ttl=60)
def get_repo_files():
    repo = get_github_repo()
    if repo:
        try:
            contents = repo.get_contents("dados")
            return [content.name for content in contents if content.name != ".gitkeep"]
        except UnknownObjectException: return []
    return []

def upload_file_to_github(file_name, file_content):
    repo = get_github_repo()
    if repo:
        path_no_repo = f"dados/{file_name}"
        try:
            contents = repo.get_contents(path_no_repo, ref="main")
            repo.update_file(contents.path, f"DOCS: Atualiza documento via app: {file_name}", file_content, contents.sha, branch="main")
        except UnknownObjectException:
            repo.create_file(path_no_repo, f"DOCS: Adiciona novo documento via app: {file_name}", file_content, branch="main")
        return True
    return False

def delete_file_from_github(file_name):
    repo = get_github_repo()
    if repo:
        path_no_repo = f"dados/{file_name}"
        try:
            contents = repo.get_contents(path_no_repo, ref="main")
            repo.delete_file(contents.path, f"DOCS: Remove documento via app: {file_name}", contents.sha, branch="main")
            return True
        except Exception as e:
            st.error(f"Erro ao deletar arquivo no GitHub: {e}")
            return False
    return False

# --- 5. DADOS PADRÃO E AVATARES ---
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

# --- 6. APLICAÇÃO PRINCIPAL ---
criar_tabela_personas()
st.title("🤖 Especialista Virtual DiamondOne V3.1")

st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Selecione uma página:", ["Chat com Especialista", "Gerenciador de Personas", "Gerenciador de Conhecimento"])

personas_db = buscar_personas()

if pagina == "Chat com Especialista":
    st.sidebar.title("Configurações do Chat")
    if not personas_db:
        st.warning("Nenhuma persona encontrada. Crie a primeira no 'Gerenciador de Personas'.")
        st.stop()
    
    modo_selecionado_nome = st.sidebar.selectbox("Selecione a Persona:", options=list(personas_db.keys()))
    st.header(f"Conversando com o {modo_selecionado_nome}")

    if "chat_history" not in st.session_state: st.session_state.chat_history = {}
    if modo_selecionado_nome not in st.session_state.chat_history: st.session_state.chat_history[modo_selecionado_nome] = []
    
    for message in st.session_state.chat_history[modo_selecionado_nome]:
        avatar = AVATARES.get(message.type, AVATARES.get(modo_selecionado_nome, "🤖"))
        with st.chat_message(message.type, avatar=avatar):
            st.markdown(f'**{modo_selecionado_nome if message.type == "ai" else "User"}**')
            st.markdown(message.content)

    if prompt_usuario := st.chat_input("Faça sua pergunta..."):
        st.session_state.chat_history[modo_selecionado_nome].append(HumanMessage(content=prompt_usuario))
        with st.chat_message("human", avatar=AVATARES["human"]):
            st.markdown(f'**User**'); st.markdown(prompt_usuario)

        with st.chat_message("ai", avatar=AVATARES.get(modo_selecionado_nome, "🤖")):
            st.markdown(f'**{modo_selecionado_nome}**')
            placeholder = st.empty()
            try:
                # NOME DO MODELO CORRIGIDO E VALIDADO
                llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.5, streaming=True)
                retriever = carregar_e_processar_dados()
                prompt_texto_selecionado = personas_db[modo_selecionado_nome]
                chat_history_langchain = st.session_state.chat_history[modo_selecionado_nome][:-1]

                # LÓGICA DE EXECUÇÃO SEPARADA E CORRIGIDA
                if modo_selecionado_nome == "Analista de Conhecimento (Híbrido)":
                    search_tool = TavilySearchResults()
                    web_results = search_tool.invoke({"query": f"definição de '{prompt_usuario}' em manufatura"})
                    docs_internos = retriever.invoke(prompt_usuario)
                    synthesis_prompt = ChatPromptTemplate.from_template(prompt_texto_selecionado)
                    synthesis_chain = synthesis_prompt | llm
                    
                    def stream_analista(stream_chunks):
                        for chunk in stream_chunks:
                            yield chunk.content # O formato do chunk é diferente aqui
                    
                    stream_chunks = synthesis_chain.stream({
                        "input": prompt_usuario, "context": docs_internos, "web_search_results": web_results
                    })
                    resposta_completa = placeholder.write_stream(stream_analista(stream_chunks))
                else:
                    contextualize_q_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Dada a conversa abaixo e a última pergunta do usuário, formule uma pergunta autônoma que possa ser entendida sem o histórico da conversa. Não responda à pergunta, apenas a reformule se necessário, caso contrário, retorne-a como está."),
                        ("placeholder", "{chat_history}"), ("human", "{input}"),
                    ])
                    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt_texto_selecionado), ("placeholder", "{chat_history}"), ("human", "{input}"),
                    ])
                    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                    
                    def stream_rag(stream_chunks):
                        for chunk in stream_chunks:
                            if answer_chunk := chunk.get("answer"):
                                yield answer_chunk
                    
                    stream_chunks = rag_chain.stream({"input": prompt_usuario, "chat_history": chat_history_langchain})
                    resposta_completa = placeholder.write_stream(stream_rag(stream_chunks))
                
                st.session_state.chat_history[modo_selecionado_nome].append(AIMessage(content=resposta_completa))
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}"); placeholder.empty()

elif pagina == "Gerenciador de Personas":
    st.header("Gerenciador de Personas")
    st.info("Crie, edite e delete as personalidades do seu especialista de IA.")
    
    with st.expander("➕ Criar Nova Persona"):
        with st.form("nova_persona_form"):
            novo_nome = st.text_input("Nome da Nova Persona")
            novo_prompt = st.text_area("Prompt da Nova Persona:", height=200, help="Use {context} e {input} para RAG.")
            submitted = st.form_submit_button("Criar Persona")
            if submitted:
                if novo_nome and novo_prompt:
                    criar_nova_persona(novo_nome, novo_prompt); st.success(f"Persona '{novo_nome}' criada!"); st.rerun()
                else:
                    st.error("Por favor, preencha o nome e o prompt.")

    st.divider()
    
    if not personas_db:
        st.warning("Nenhuma persona encontrada. Crie sua primeira persona acima ou crie os padrões abaixo.")
        if st.button("Criar Personas Padrão"):
            with conn.session as s:
                for nome, prompt_texto in PERSONAS_PADRAO.items():
                    s.execute(text('INSERT OR REPLACE INTO personas (nome, prompt) VALUES (:nome, :prompt);'), params=dict(nome=nome, prompt=prompt_texto))
                s.commit()
            st.success("Personas padrão criadas!"); st.rerun()
    else:
        st.subheader("Editar Personas Existentes")
        persona_para_editar = st.selectbox("Selecione uma persona para editar:", options=list(personas_db.keys()))
        prompt_atual = personas_db.get(persona_para_editar, "")
        prompt_editado = st.text_area("Edite o prompt:", value=prompt_atual, height=300, key=f"editor_{persona_para_editar}")
        
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            if st.button("Salvar Alterações"):
                atualizar_persona(persona_para_editar, prompt_editado); st.success(f"Persona '{persona_para_editar}' atualizada!")
        with col2:
            if persona_para_editar in PERSONAS_PADRAO:
                if st.button("Restaurar Padrão"):
                    prompt_padrao = PERSONAS_PADRAO[persona_para_editar]
                    atualizar_persona(persona_para_editar, prompt_padrao)
                    st.success(f"Persona '{persona_para_editar}' restaurada para o padrão!"); st.rerun()
        with col3:
            if st.button("❌ Deletar", type="primary"):
                deletar_persona(persona_para_editar); st.success(f"Persona '{persona_para_editar}' deletada!"); st.rerun()

elif pagina == "Gerenciador de Conhecimento":
    st.header("📚 Gerenciador de Conhecimento")
    st.info("Adicione ou remova documentos da base de conhecimento do especialista.")
    
    repo = get_github_repo()
    if not repo:
        st.error("Configuração do GitHub não encontrada. Adicione GITHUB_TOKEN e GITHUB_REPO aos seus segredos."); st.stop()

    st.subheader("Documentos Atuais na Base:")
    with st.spinner("Carregando lista de documentos do repositório..."):
        lista_de_arquivos = get_repo_files()
        if not lista_de_arquivos: st.write("Nenhum documento na base de conhecimento.")
        else:
            for arquivo in lista_de_arquivos:
                col1, col2 = st.columns([4, 1])
                with col1: st.write(f"- `{arquivo}`")
                with col2:
                    if st.button("❌ Deletar", key=f"delete_{arquivo}", type="primary"):
                        with st.spinner(f"Deletando '{arquivo}'..."):
                            success = delete_file_from_github(arquivo)
                            if success:
                                st.success(f"'{arquivo}' deletado com sucesso!")
                                st.info("O aplicativo será reiniciado para atualizar a base de conhecimento.")
                                time.sleep(2)
                                st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

    st.divider()

    st.subheader("Adicionar Novos Documentos:")
    uploaded_files = st.file_uploader("Selecione um ou mais arquivos para enviar ao GitHub", type=['pdf', 'docx', 'txt'], accept_multiple_files=True, key="file_uploader")

    if uploaded_files:
        if st.button("Enviar Arquivos para a Base de Conhecimento"):
            success_count = 0
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                with st.spinner(f"Fazendo upload de '{uploaded_file.name}'..."):
                    success = upload_file_to_github(uploaded_file.name, file_content)
                    if success: st.success(f"'{uploaded_file.name}' enviado!"); success_count += 1
            
            if success_count > 0:
                st.info("Limpando cache e reiniciando para incorporar o novo conhecimento.")
                st.session_state.file_uploader = []
                time.sleep(2)
                st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

