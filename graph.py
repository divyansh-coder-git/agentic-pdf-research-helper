from langchain_core.documents import Document
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    question: str
    question_type: str
    answer: str
    is_relevant: bool
    docs: List[Document]
    search_query: str
    chat_history: str

def format_chat_history(messages, max_turns=3):
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

def build_agent(vectorstore, chunks):
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature = 0)

    prompt = ChatPromptTemplate.from_template("""
        You are a research assistant. Answer the question using ONLY the context below.
        If the answer isn't in the context, say you don't know.
        Use the recent conversation only to understand what the question is referring to
        (e.g. pronouns like "it" or "that") — the actual answer must still come from the context.

        Recent conversation:
        {chat_history}

        Context:
        {context}

        Question: {question}
        """)

    relevance_prompt = ChatPromptTemplate.from_template("""
        Look at the context and the question below.
        Does the context contain information that could answer the question?
        Reply with ONLY one word: "yes" or "no".

        Context:
        {context}

        Question: {question}
        """)
    
    classify_prompt = ChatPromptTemplate.from_template("""
        Given the recent conversation, classify the CURRENT question below into exactly one category:

        - "broad": the question asks about the ENTIRE document as a whole — its overall
        topic, purpose, main contributions, or a general summary. It cannot be answered
        by looking at just one small section.
        Examples: "What is this paper about?", "Summarize this document",
        "What are the main contributions?"

        - "specific": the question asks about ONE particular concept, term, fact, method,
        number, or section — even if phrased vaguely or with pronouns like "it" or "that"
        that refer back to something specific mentioned earlier in the conversation.
        Examples: "What is self-attention?", "Why is it useful?" (when "it" refers to a
        specific concept discussed earlier), "What optimizer was used?"

        Recent conversation:
        {chat_history}

        Current question: {question}

        Reply with ONLY one word: "broad" or "specific".
        """)

    summarize_prompt = ChatPromptTemplate.from_template("""
        You are a research assistant. Use the full document content below to answer
        the question, which is asking for an overview or summary.

        Document:
        {full_text}

        Question: {question}
        """)

    rewrite_prompt = ChatPromptTemplate.from_template("""
        Given the recent conversation and a follow-up question, rewrite the follow-up
        into a standalone question that makes sense without needing the conversation.
        If the question is already standalone, just repeat it unchanged.

        Recent conversation:
        {chat_history}

        Follow-up question: {question}

        Standalone question:
        """)

    def classify_question_node(state: AgentState) -> AgentState:
        chain = classify_prompt | llm
        response = chain.invoke({"question": state["question"], "chat_history": state.get("chat_history", "")})
        verdict = str(response.content).strip().lower()
        state["question_type"] = "broad" if "broad" in verdict else "specific"
        print(f"[CLASSIFY] '{state['question']}' -> {state['question_type']}")
        return state

    def route_after_classify(state: AgentState) -> str:
        return "summarize" if state["question_type"] == "broad" else "retrieve"

    def summarize_node(state: AgentState) -> AgentState:
        max_chunks = 40
        limited_chunks = chunks[:max_chunks]
        full_text = "\n\n".join(chunk.page_content for chunk in limited_chunks)
        # full_text = "\n\n".join(chunk.page_content for chunk in chunks)
        chain = summarize_prompt | llm
        response = chain.invoke({"full_text": full_text, "question": state["question"]})
        state["answer"] = str(response.content)
        return state
    
    def retrieve_node(state: AgentState) -> AgentState:
        chat_history = state.get("chat_history", "")
        if chat_history.strip():
            chain = rewrite_prompt | llm
            response = chain.invoke({
                "chat_history": chat_history,
                "question": state["question"],
            })
            search_query = str(response.content).strip()
            print(f"[REWRITE] original: '{state['question']}' -> rewritten: '{search_query}'")
        else:
            search_query = state["question"]
        state["search_query"] = search_query
        docs = vectorstore.similarity_search(search_query, k=5)
        state["docs"] = docs
        return state

    def check_relevance_node(state: AgentState) -> AgentState:
        context = "\n\n".join(doc.page_content for doc in state["docs"])
        chain = relevance_prompt | llm
        response = chain.invoke({"context": context, "question": state.get("search_query", state["question"])})
        verdict = str(response.content).strip().lower()
        state["is_relevant"] = "yes" in verdict
        return state

    def route_after_relevance_check(state: AgentState) -> str:
        if state["is_relevant"]:
            return "generate"
        return "no_context_response"

    def no_context_response_node(state: AgentState) -> AgentState:
        state["answer"] = "I couldn't find relevant information in the document to answer that question."
        state["docs"] = []
        return state

    def generate_node(state: AgentState) -> AgentState:
        context = "\n\n".join(doc.page_content for doc in state["docs"])
        chain = prompt | llm
        response = chain.invoke({"context":context, "question":state["question"], "chat_history": state.get("chat_history", "")})
        state["answer"] = str(response.content)
        return state

    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_question_node)
    graph.add_node("summarize", summarize_node)    
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_relevance", check_relevance_node)
    graph.add_node("no_context_response", no_context_response_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"summarize": "summarize", "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "check_relevance")
    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance_check,
        {
            "generate": "generate",
            "no_context_response": "no_context_response"
        }
    )

    graph.add_edge("generate", END)
    graph.add_edge("no_context_response", END)
    graph.add_edge("summarize", END)

    return graph.compile()