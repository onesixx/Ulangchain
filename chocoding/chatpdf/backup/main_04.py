#------------------------------------------------------------
llm = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1,
    num_predict = 256,
)

if os.path.exists("v_store/index.faiss"):
    st.write("Vector store is already loaded.")
    vectorstore = lc_faiss.load_local(V_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    logger.info(f"vectorstore is {vectorstore}")
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Save the uploaded file to disk
        ###### LOAD -------------------------------------------------------------------
        pages = pdf_to_document(uploaded_file)
        ###### SPLIT ------------------------------------------------------------------
        texts = text_splitter.split_documents(pages)

        ### STORE (Vector Store)  --------------------------------------------------------
        res = faiss.StandardGpuResources()  # GPU 리소스 초기화
        cpu_index = faiss.IndexFlatL2(dimension_size)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        vectorstore = lc_faiss(
            embedding_function=embedding_model,
            index=gpu_index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )
        vectorstore.add_documents(documents=texts)
        vectorstore.save_local(V_STORE_PATH)
        # vectorstore = lc_faiss.from_documents(texts, embedding_model)

        st.write("File uploaded and store chroma successfully!")