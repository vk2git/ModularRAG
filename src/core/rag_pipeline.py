from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from src.core.llm import LLMFactory
from src.core.vector_store import VectorStoreFactory
from src.core.embedding import EmbeddingFactory
from src.core.memory import MemoryFactory
from src.core.guardrails import GuardrailsManager
from src.utils.config_loader import load_config

class RAGPipeline:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.config = load_config()

        self.llm_factory = LLMFactory()
        self.llm = self.llm_factory.create_llm()

        self.embed_factory = EmbeddingFactory()
        self.embeddings = self.embed_factory.create_embeddings_model()

        self.vector_factory = VectorStoreFactory()
        self.vector_store = self.vector_factory.create_vector_store(self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        self.memory_factory = MemoryFactory()
        
        guardrails_config = self.config.get("guardrails", {})
        self.guardrails = GuardrailsManager(guardrails_config, llm=self.llm)
        

    def _log(self, msg):
        """Logs message only if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    def check_health(self):
        """
        Verifies connections to LLM and Vector DB.
        """
        print("\n---- Running Health Checks ----")
        
        try:
            print("1. Checking LLM connection...", end=" ")
            self.llm.invoke("Hello")
            print("✅ OK")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            print("   Hint: Check your API keys, internet connection, or if local model is running.")
            
        try:
            print("2. Checking Vector DB connection...", end=" ")
            self.vector_store.similarity_search("test", k=1)
            print("✅ OK")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            print("   Hint: Check if database path exists or if cloud service is reachable.")
            
        print("-------------------------------\n")

    def run(self, query:str, session_id:str = "default"):
        """
        Checks for relevant documents.
        If found then RAG is used, else direct LLM (General Chat knkowledge)
        """
        
        try:
            validation_result = self.guardrails.validate_input(query)
            if not validation_result["valid"]:
                print(f"⚠️  Input validation failed: {validation_result['reason']}")
                return self.guardrails.get_safe_response(validation_result)
            
            sanitized_query = validation_result["sanitized_input"]
            
            history = self.memory_factory.get_chat_history(session_id)
            
            memory_type = self.config.get("memory", {}).get("type", "window")
            
            if memory_type == "summary":
                try:
                    from langchain.memory import ConversationSummaryBufferMemory
                except ImportError:
                    from langchain_classic.memory import ConversationSummaryBufferMemory
                
                memory = ConversationSummaryBufferMemory(
                    llm=self.llm,
                    chat_memory=history,
                    max_token_limit=1000,
                    return_messages=True
                )
                chat_history = memory.load_memory_variables({})["history"]
                
            elif memory_type == "vector":
                mem_config = self.config.get("memory", {}).get("vector_memory", {})
                collection_name = mem_config.get("collection_name", "chat_memory")
                k = mem_config.get("k", 5)
                
                memory_store = self._create_memory_vector_store(collection_name)
                
                memory = self.memory_factory.create_vector_memory(memory_store, k=k)
                
                memory_data = memory.load_memory_variables({"prompt": sanitized_query})
                retrieved_history = memory_data.get("history", "")
                
                from langchain_core.messages import SystemMessage
                chat_history = [SystemMessage(content=f"Relevant Past Conversations:\n{retrieved_history}")]
                
                self.active_memory = memory
                
            else:
                chat_history = history.messages
                self.active_memory = None 

            self._log(f"---- Processing query: {sanitized_query} (Session: {session_id})")
            
            rag_config = self.config.get("rag", {})
            threshold = rag_config.get("score_threshold", 0.4)
            
            results_with_scores = self.vector_store.similarity_search_with_score(sanitized_query, k=3)
            
            relevant_docs = []
            for doc, score in results_with_scores:
                self._log(f"   - Doc Score: {score:.4f} | Content: {doc.page_content[:50]}...")
                if score < threshold: 
                    relevant_docs.append(doc)
            
            if relevant_docs:
                self._log(f"Found {len(relevant_docs)} relevant document chunks (Score < {threshold}). Switching to RAG mode \n")
                response = self._run_rag_mode(sanitized_query, relevant_docs, chat_history)
            else:
                self._log(f"No documents met the threshold (Score < {threshold}). Switching to General Chat Knowledge")
                response = self._run_general_mode(sanitized_query, chat_history)
            
            output_validation = self.guardrails.validate_output(response)
            if not output_validation["valid"]:
                print(f"⚠️  Output validation warning: {output_validation['reason']}")
                response = output_validation["sanitized_output"]
            
            if hasattr(self, 'active_memory') and self.active_memory:
                self.active_memory.save_context({"input": query}, {"output": response})
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=response))
            else:
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=response))
            
            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"I encountered an error while processing your request: {str(e)}. Please check the logs or configuration."

    def _create_memory_vector_store(self, collection_name: str):
        """
        Creates a vector store specifically for memory.
        """
        return self.vector_factory.create_vector_store(self.embeddings, collection_name=collection_name)
        
    def _run_rag_mode(self, query, docs, chat_history):
        """
        Answers based on provided context
        """

        context_text = "\n\n".join([doc.page_content for doc in docs])

        template = """
        You are a specialized assistant. Answer the question based on the following context.
        
        CRITICAL INSTRUCTIONS:
        1. Answer CONCISELY. Do not ramble.
        2. If the context is NOT relevant to the question, IGNORE the context and answer from your own knowledge, but mention that you are doing so.
        3. If you don't know the answer, just say "I don't know".
        
        Context:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "question": query
        })
    
    def _run_general_mode(self, query, chat_history):
        """
        Answer question based on LLM's own training data
        """

        template = """
        You are a helpful AI assistant. Answer the following question to the best of your ability.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "chat_history": chat_history,
            "question": query
        })