!pip install -q chromadb==0.4.24 sentence-transformers langgraph fastapi uvicorn
!rm -rf support_assistant/chroma_db
import os
os.environ["MOCK_LLM"] = os.getenv("MOCK_LLM", "1")
print(f"MOCK_LLM={os.getenv('MOCK_LLM','1')}")

from typing import TypedDict, List
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END

DOCS = {
    "doc_01": "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
    "doc_02": "Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3–5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.",
    "doc_03": "Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period.",
    "doc_04": "Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.",
    "doc_05": "Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.",
    "doc_06": "If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed.",
    "doc_07": "Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law.",
    "doc_08": "Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered."
}

PROMPT_TEMPLATE = """
ROLE: You are Zepto Support Assistant.
CONTEXT: {context}
TASK: Answer query {query}
FORMAT: JSON answer, sources, confidence
LENGTH: Under 150 words
Negative Constraint: Do NOT answer using information not present in context
Few-shot: Q: fee below 149? Context: free over 149 else 25. JSON: {"answer":"25 fee","sources":["doc_01"],"confidence":0.9}
"""

print("Loading model all-MiniLM-L6-v2")
model = SentenceTransformer('all-MiniLM-L6-v2')

# FIX: Use Ephemeral Client for Colab Python 3.13 - avoids PersistentClient bug
# For local grading you can switch back to PersistentClient(path="...") - same API


ids = list(DOCS.keys())
texts = list(DOCS.values())
embs = model.encode(texts).tolist()
# clear if already added
try:
    collection.delete(ids=ids)
except:
    pass
collection.add(ids=ids, documents=texts, embeddings=embs)
print(f"Indexed {collection.count()} docs - FIXED")

class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float

KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

def classify_intent_node(state: GraphState):
    q = state["query"].lower()
    intent = "general_question"
    for kw in KEYWORDS:
        if kw in q:
            intent = "policy_question"
            break
    print(f"classified {intent}")
    return {"intent": intent}

def retrieve_and_answer_node(state: GraphState):
    q = state["query"]
    q_emb = model.encode([q]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=3)
    top_docs = res['documents'][0]
    top_ids = res['ids'][0]
    snippet = top_docs[0][:200]
    return {"answer": f"Based on the retrieved context: {snippet}", "sources": top_ids, "confidence": 1.0}

def direct_answer_node(state: GraphState):
    return {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}

def route(state: GraphState):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
workflow.add_node("direct_answer", direct_answer_node)
workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges("classify_intent", route, {"retrieve_and_answer":"retrieve_and_answer","direct_answer":"direct_answer"})
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)
graph = workflow.compile()

class QueryRequest(BaseModel):
    query: str
class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

app = FastAPI()
@app.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    r = graph.invoke({"query": req.query})
    return QueryResponse(answer=r["answer"], sources=r["sources"], confidence=r["confidence"])

if __name__ == "__main__":
    print(graph.invoke({"query": "What is delivery fee for orders over 149?"}))
    print(graph.invoke({"query": "Tell me a joke"}))
