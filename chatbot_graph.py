from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import Annotated, Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os
import json
import logging
from datetime import datetime
from kb_handler import KB_handler
from PIL import Image
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ENHANCED UTILS ---
def load_prompt(path):
    """Load prompt from file with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {path}")
        return ""
    except Exception as e:
        logger.error(f"Error loading prompt {path}: {e}")
        return ""

def prepare_image(image, max_size=(512, 512), quality=85):
    """Enhanced image preparation with better quality and error handling"""
    try:
        if isinstance(image, dict) and image.get("data"):
            return image

        with Image.open(image) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Maintain aspect ratio while resizing
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            
        return {
            "data": encoded, 
            "mime_type": "image/jpeg",
            "size": img.size
        }
    except Exception as e:
        logger.error(f"Error processing image {image}: {e}")
        return None

def format_search_results(results: Dict) -> str:
    """Format Tavily search results for better readability"""
    if not results:
        return "No search results found."
    
    formatted = "🔍 Search Results:\n\n"
    
    # Add answer if available
    if results.get("answer"):
        formatted += f"**Quick Answer:** {results['answer']}\n\n"
    
    # Add top results
    for i, result in enumerate(results.get("results", [])[:3], 1):
        title = result.get("title", "No title")
        snippet = result.get("content", "No content")[:200] + "..."
        url = result.get("url", "")
        
        formatted += f"**{i}. {title}**\n"
        formatted += f"{snippet}\n"
        formatted += f"🔗 Source: {url}\n\n"
    
    return formatted

# --- ENV & INIT ---
load_dotenv()
gemini_key = os.getenv("GEMINI_KEY") or os.getenv("gemini_key")
tavily_key = os.getenv("TAVILY_KEY") or os.getenv("TAVILY")

if not gemini_key:
    raise ValueError("GEMINI_KEY not found in environment variables")
if not tavily_key:
    raise ValueError("TAVILY_KEY not found in environment variables")

# Load prompts with fallbacks
CLASSIFICATION_PROMPT = load_prompt("prompts/classification_prompt.yaml") or """
Classify the user's message into one of these categories:
- "logical": Questions that can be answered with existing knowledge, reasoning, or local knowledge base
- "online_search": Questions requiring current information, recent events, or real-time data
- "image_analysis": Messages containing or referencing images
- "follow_up": Follow-up questions to previous responses

Consider:
- Time-sensitive queries need online search
- General knowledge can be handled logically
- Current events, news, weather need online search
- Personal opinions or creative tasks are logical

Respond only with the category name.
"""

CHATBOT_PROMPT = load_prompt("prompts/chatbot_prompt.yaml") or """
You are an intelligent AI assistant. Provide helpful, accurate, and engaging responses.

Guidelines:
- Be conversational and friendly
- Use the knowledge base context when available
- For images, describe what you see in detail
- Admit when you're uncertain
- Provide step-by-step explanations for complex topics
- Use examples to clarify concepts

Context will be provided from:
1. Knowledge Base (if relevant)
2. Images (if provided)
3. Previous conversation
"""

SEARCH_PROMPT = load_prompt("prompts/search_prompt.yaml") or """
You are researching information online. Based on the search results provided:

Guidelines:
- Synthesize information from multiple sources
- Highlight key findings
- Note any conflicting information
- Provide source attribution
- Focus on the most recent and credible information
- If information is insufficient, suggest more specific queries

Format your response clearly with main points and supporting details.
"""

PARSER_PROMPT = load_prompt("prompts/parser_prompt.yaml") or """
Parse the assistant's response into a structured format that includes:
- A clear summary of the main answer
- Relevant sources (if any)
- Key insights or takeaways
- Any limitations or uncertainties mentioned

Maintain the helpful tone while organizing the information clearly.
"""

# Initialize components
try:
    kb = KB_handler()
    knowledge_base = kb._load_kb()
    logger.info("Knowledge base loaded successfully")
except Exception as e:
    logger.warning(f"Knowledge base initialization failed: {e}")
    kb = None
    knowledge_base = None

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    api_key=gemini_key,
    temperature=0.7,
    max_tokens=2048
)

tavily_client = TavilyClient(api_key=tavily_key)

# Configuration
MAX_HISTORY = 10
KB_THRESHOLD = 65
MAX_SEARCH_RESULTS = 5

# --- ENHANCED STATE ---
class EnhancedState(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str
    images: Optional[List[Dict[str, Any]]]
    kb_context: Optional[str]
    search_context: Optional[str]
    conversation_context: Optional[str]
    metadata: Optional[Dict[str, Any]]

# --- ENHANCED MODELS ---
class MessageClassifier(BaseModel):
    message_type: Literal["logical", "online_search", "image_analysis", "follow_up"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification")
    reasoning: str = Field(description="Brief explanation of classification")

class ChatResponse(BaseModel):
    content: str = Field(description="Main response content")
    sources_used: List[str] = Field(default=[], description="Sources referenced")
    confidence: Optional[float] = Field(default=None, description="Response confidence")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")

class SearchQuery(BaseModel):
    query: str = Field(description="Optimized search query")
    domains: List[str] = Field(default=[], description="Preferred domains")
    search_type: Literal["general", "academic", "news", "technical"] = "general"

class ParsedAnswer(BaseModel):
    summary: str = Field(description="Concise summary of the response")
    key_points: List[str] = Field(default=[], description="Main points covered")
    sources: List[str] = Field(default=[], description="Referenced sources")
    confidence_level: str = Field(default="medium", description="High/Medium/Low confidence")
    follow_up_suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    limitations: Optional[str] = Field(default=None, description="Any limitations or uncertainties")

# --- ENHANCED NODES ---
def enhanced_classifier(state: EnhancedState) -> Dict[str, Any]:
    """Enhanced message classification with confidence scoring"""
    try:
        # Handle both dict and AIMessage objects
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):
            last_message = last_msg.content
        else:
            last_message = last_msg.get("content", "")
            
        images = state.get("images", [])
        
        # Check for images first
        if images:
            return {
                "message_type": "image_analysis",
                "metadata": {"has_images": True, "image_count": len(images)}
            }
        
        # Enhanced classification prompt
        classification_input = f"""
        {CLASSIFICATION_PROMPT}
        
        Message to classify: "{last_message}"
        
        Previous context: {len(state.get('messages', [])) > 1}
        """
        
        # Use simple LLM call to avoid structured output issues
        classification_response = llm.invoke([
            {"role": "user", "content": classification_input}
        ])
        
        # Simple text parsing for message type
        response_text = classification_response.content.lower()
        if "online_search" in response_text or "search" in response_text:
            message_type = "online_search"
            confidence = 0.8
        elif "image" in response_text:
            message_type = "image_analysis"
            confidence = 0.9
        elif "follow" in response_text:
            message_type = "follow_up"
            confidence = 0.7
        else:
            message_type = "logical"
            confidence = 0.8
        
        logger.info(f"Message classified as: {message_type} (confidence: {confidence})")
        
        return {
            "message_type": message_type,
            "messages": state["messages"][-MAX_HISTORY:],
            "metadata": {
                "classification_confidence": confidence,
                "classification_reasoning": classification_response.content,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {
            "message_type": "logical",  # Safe fallback
            "metadata": {"classification_error": str(e)}
        }

def knowledge_retrieval_node(state: EnhancedState) -> Dict[str, Any]:
    """Dedicated node for knowledge base retrieval"""
    try:
        query = state["messages"][-1].content
        kb_context = ""
        
        if kb and knowledge_base:
            kb_results = kb._search_kb(query, max_results=5, threshold=KB_THRESHOLD)
            if kb_results:
                kb_context = f"📚 Knowledge Base Context:\n{kb_results}\n\n"
                logger.info(f"Retrieved {len(kb_results.split('\\n'))} KB entries")
        
        return {"kb_context": kb_context}
    
    except Exception as e:
        logger.error(f"Knowledge retrieval error: {e}")
        return {"kb_context": ""}

def enhanced_chatbot(state: EnhancedState) -> Dict[str, Any]:
    """Enhanced chatbot with better context handling and multimodal support"""
    try:
        # Handle both dict and AIMessage objects
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):
            query = last_msg.content
        else:
            query = last_msg.get("content", "")
        
        images = state.get("images", [])
        kb_context = state.get("kb_context", "")
        
        # Build conversation context
        conversation_history = ""
        if len(state["messages"]) > 1:
            recent_messages = state["messages"][-3:-1]  # Last 2 exchanges
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                    role = "User" if hasattr(msg, 'type') and msg.type == "human" else "Assistant"
                else:
                    content = msg.get('content', '')
                    role = "User" if msg.get("role") == "user" else "Assistant"
                conversation_history += f"{role}: {content[:100]}...\n"
        
        # Build enhanced prompt
        system_prompt = f"""
        {CHATBOT_PROMPT}
        
        Current timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        {kb_context}
        
        Recent conversation:
        {conversation_history}
        
        Instructions:
        - Provide a comprehensive but concise response
        - Use the knowledge base information when relevant
        - For images, provide detailed analysis
        - Suggest related topics the user might find interesting
        - Be engaging and conversational
        """
        
        # Build multimodal message
        multimodal_content = [{"type": "text", "text": system_prompt}]
        
        # Add images if present
        processed_images = 0
        for img in images or []:
            if isinstance(img, str):
                img = prepare_image(img)
            
            if isinstance(img, dict) and "data" in img:
                multimodal_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['mime_type']};base64,{img['data']}"
                    }
                })
                processed_images += 1
        
        # Add user query
        multimodal_content.append({
            "type": "text", 
            "text": f"\nUser Query: {query}"
        })
        
        # Get LLM response (without structured output for now)
        response = llm.invoke([{"role": "user", "content": multimodal_content}])
        
        logger.info(f"Generated response, {processed_images} images processed")
        
        # Return as simple message to avoid AIMessage issues
        return {
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": response.content,
                "metadata": {
                    "images_processed": processed_images,
                    "has_kb_context": bool(kb_context),
                    "timestamp": datetime.now().isoformat()
                }
            }]
        }
    
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return {
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": f"I encountered an error processing your request. Please try rephrasing your question.",
                "metadata": {"error": str(e)}
            }]
        }

def query_optimizer_node(state: EnhancedState) -> Dict[str, Any]:
    """Optimize search queries for better results"""
    try:
        original_query = state["messages"][-1].content
        
        # Generate optimized search query
        optimization_prompt = f"""
        Optimize this search query for better web search results:
        Original: "{original_query}"
        
        Consider:
        - Key terms and synonyms
        - Temporal context (if time-sensitive)
        - Domain specificity
        - Search intent
        
        Generate an optimized search query and suggest relevant domains.
        """
        
        optimized = llm.with_structured_output(SearchQuery).invoke([
            {"role": "user", "content": optimization_prompt}
        ])
        
        return {"metadata": {"optimized_query": optimized.dict()}}
    
    except Exception as e:
        logger.error(f"Query optimization error: {e}")
        return {"metadata": {"optimization_error": str(e)}}

def enhanced_search_online(state: EnhancedState) -> Dict[str, Any]:
    """Enhanced online search with better result processing"""
    try:
        # Handle both dict and AIMessage objects
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):
            original_query = last_msg.content
        else:
            original_query = last_msg.get("content", "")
            
        kb_context = state.get("kb_context", "")
        
        # Use optimized query if available
        optimized_query_data = state.get("metadata", {}).get("optimized_query")
        search_query = optimized_query_data.get("query", original_query) if optimized_query_data else original_query
        
        logger.info(f"Searching for: {search_query}")
        
        # Perform search with enhanced parameters
        search_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            include_domains=[".org", ".edu", ".gov", ".ac.uk", ".reuters.com", ".bbc.com"],
            exclude_domains=["reddit.com", "quora.com"],
            include_answer=True,
            include_raw_content=False,
            max_results=MAX_SEARCH_RESULTS
        )
        
        # Format results
        formatted_results = format_search_results(search_results)
        
        # Enhanced search prompt
        search_prompt = f"""
        {SEARCH_PROMPT}
        
        {kb_context}
        
        Original query: "{original_query}"
        Search query used: "{search_query}"
        
        Search Results:
        {formatted_results}
        
        Provide a comprehensive answer synthesizing the search results with any relevant knowledge base information.
        """
        
        # Get LLM response (without structured output to avoid AIMessage issues)
        response = llm.invoke([{"role": "user", "content": search_prompt}])
        
        logger.info(f"Search completed with {len(search_results.get('results', []))} results")
        
        return {
            "search_context": formatted_results,
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": response.content,
                "metadata": {
                    "search_query": search_query,
                    "results_count": len(search_results.get('results', [])),
                    "has_search_context": True,
                    "timestamp": datetime.now().isoformat()
                }
            }]
        }
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": "I encountered an error while searching online. Let me try to answer based on my existing knowledge.",
                "metadata": {"search_error": str(e)}
            }]
        }

def context_aggregator_node(state: EnhancedState) -> Dict[str, Any]:
    """Aggregate all available context for better responses"""
    try:
        contexts = []
        
        # Add KB context
        if state.get("kb_context"):
            contexts.append(f"Knowledge Base:\n{state['kb_context']}")
        
        # Add search context
        if state.get("search_context"):
            contexts.append(f"Search Results:\n{state['search_context']}")
        
        # Add conversation context
        if len(state["messages"]) > 1:
            recent_msgs = state["messages"][-3:-1]
            conv_context = "\n".join([
                f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:150]}..."
                for msg in recent_msgs
            ])
            contexts.append(f"Recent Conversation:\n{conv_context}")
        
        aggregated_context = "\n\n".join(contexts) if contexts else ""
        
        return {"conversation_context": aggregated_context}
    
    except Exception as e:
        logger.error(f"Context aggregation error: {e}")
        return {"conversation_context": ""}

def enhanced_parser_node(state: EnhancedState) -> Dict[str, Any]:
    """Enhanced parser with better structured output"""
    try:
        last_message = state["messages"][-1]
        
        # Handle both dict and AIMessage objects
        if hasattr(last_message, 'content'):
            content = last_message.content
            metadata = getattr(last_message, 'additional_kwargs', {}).get('metadata', {})
        else:
            content = last_message.get("content", "")
            metadata = last_message.get("metadata", {})
        
        parser_input = f"""
        {PARSER_PROMPT}
        
        Response to parse: "{content}"
        
        Metadata available: {json.dumps(metadata, indent=2)}
        
        Generate a structured summary that improves upon the original response.
        """
        
        # Use simple LLM call instead of structured output to avoid issues
        parsed_response = llm.invoke([
            {"role": "user", "content": parser_input}
        ])
        
        # Create enhanced final response
        final_response = f"""
{parsed_response.content}

**Processing completed at:** {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        return {
            "messages": state["messages"][:-1] + [{
                "role": "assistant",
                "content": final_response,
                "metadata": {
                    **metadata,
                    "processing_complete": True,
                    "parsed_at": datetime.now().isoformat()
                }
            }]
        }
    
    except Exception as e:
        logger.error(f"Parser error: {e}")
        # Return original message if parsing fails
        return {"messages": state["messages"]}

def quality_check_node(state: EnhancedState) -> Dict[str, Any]:
    """Quality check and response validation"""
    try:
        last_response = state["messages"][-1]
        
        # Handle both dict and AIMessage objects
        if hasattr(last_response, 'content'):
            content = last_response.content
        else:
            content = last_response.get("content", "")
        
        # Basic quality checks
        quality_score = 1.0
        issues = []
        
        if len(content.strip()) < 50:
            quality_score -= 0.3
            issues.append("Response too short")
        
        if "error" in content.lower() or "sorry" in content.lower():
            quality_score -= 0.2
            issues.append("Contains error indicators")
        
        # Check if response addresses the original query
        if len(state["messages"]) > 1:
            first_msg = state["messages"][-2]
            if hasattr(first_msg, 'content'):
                original_query = first_msg.content
            else:
                original_query = first_msg.get("content", "")
                
            if original_query and not any(word in content.lower() for word in original_query.lower().split()[:3]):
                quality_score -= 0.3
                issues.append("May not address original query")
        
        logger.info(f"Quality score: {quality_score:.2f}, Issues: {issues}")
        
        # Update metadata in the last message
        updated_messages = state["messages"][:-1]
        last_msg = state["messages"][-1]
        
        if hasattr(last_msg, 'content'):
            # Create new dict message
            updated_msg = {
                "role": "assistant",
                "content": last_msg.content,
                "metadata": {
                    "quality_score": quality_score,
                    "quality_issues": issues,
                    "quality_check_timestamp": datetime.now().isoformat()
                }
            }
        else:
            # Update existing dict message
            updated_msg = {
                **last_msg,
                "metadata": {
                    **last_msg.get("metadata", {}),
                    "quality_score": quality_score,
                    "quality_issues": issues,
                    "quality_check_timestamp": datetime.now().isoformat()
                }
            }
        
        updated_messages.append(updated_msg)
        
        return {"messages": updated_messages}
    
    except Exception as e:
        logger.error(f"Quality check error: {e}")
        return {}

# --- ENHANCED ROUTER ---
def enhanced_router(state: EnhancedState) -> str:
    """Enhanced routing logic with better decision making"""
    try:
        message_type = state.get("message_type", "logical")
        images = state.get("images", [])
        metadata = state.get("metadata", {})
        
        # Image analysis takes priority
        if images:
            logger.info("Routing to image analysis")
            return "knowledge_retrieval"
        
        # Route based on classification confidence
        confidence = metadata.get("classification_confidence", 0.5)
        
        if message_type == "online_search" and confidence > 0.7:
            logger.info("Routing to online search (high confidence)")
            return "query_optimizer"
        elif message_type == "logical" or confidence <= 0.7:
            logger.info("Routing to knowledge-based response")
            return "knowledge_retrieval"
        else:
            logger.info("Routing to search with optimization")
            return "query_optimizer"
    
    except Exception as e:
        logger.error(f"Router error: {e}")
        return "knowledge_retrieval"  # Safe fallback

def search_router(state: EnhancedState) -> str:
    """Route after query optimization"""
    return "enhanced_search"

def processing_router(state: EnhancedState) -> str:
    """Route to context aggregation"""
    return "context_aggregator"

def final_router(state: EnhancedState) -> str:
    """Final routing decision"""
    message_type = state.get("message_type", "logical")
    
    if message_type == "image_analysis" or state.get("images"):
        return "enhanced_chatbot"
    elif state.get("search_context"):
        return "enhanced_chatbot"  # Process search results
    else:
        return "enhanced_chatbot"

# --- BUILD ENHANCED GRAPH ---
def build_enhanced_graph():
    """Build the enhanced processing graph"""
    graph_builder = StateGraph(EnhancedState)
    
    # Add all nodes
    graph_builder.add_node("classifier", enhanced_classifier)
    graph_builder.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph_builder.add_node("query_optimizer", query_optimizer_node)
    graph_builder.add_node("enhanced_search", enhanced_search_online)
    graph_builder.add_node("context_aggregator", context_aggregator_node)
    graph_builder.add_node("enhanced_chatbot", enhanced_chatbot)
    graph_builder.add_node("parser", enhanced_parser_node)
    graph_builder.add_node("quality_check", quality_check_node)
    
    # Define the flow
    graph_builder.add_edge(START, "classifier")
    
    # Classification routing
    graph_builder.add_conditional_edges(
        "classifier",
        enhanced_router,
        {
            "knowledge_retrieval": "knowledge_retrieval",
            "query_optimizer": "query_optimizer"
        }
    )
    
    # Search path
    graph_builder.add_conditional_edges(
        "query_optimizer",
        search_router,
        {"enhanced_search": "enhanced_search"}
    )
    
    # Context aggregation
    graph_builder.add_conditional_edges(
        "knowledge_retrieval",
        processing_router,
        {"context_aggregator": "context_aggregator"}
    )
    
    graph_builder.add_conditional_edges(
        "enhanced_search",
        processing_router,
        {"context_aggregator": "context_aggregator"}
    )
    
    # Final processing
    graph_builder.add_conditional_edges(
        "context_aggregator",
        final_router,
        {"enhanced_chatbot": "enhanced_chatbot"}
    )
    
    graph_builder.add_edge("enhanced_chatbot", "parser")
    graph_builder.add_edge("parser", "quality_check")
    graph_builder.add_edge("quality_check", END)
    
    return graph_builder.compile()

# --- ENHANCED INTERFACE ---
class EnhancedChatInterface:
    """Enhanced interface for the chat system"""
    
    def __init__(self):
        self.graph = build_enhanced_graph()
        self.session_history = []
    
    def process_message(self, user_input: str, images: List = None) -> Dict[str, Any]:
        """Process a user message with enhanced error handling"""
        try:
            # Prepare images
            prepared_images = []
            if images:
                for img in images:
                    prepared = prepare_image(img) if isinstance(img, str) else img
                    if prepared:
                        prepared_images.append(prepared)
            
            # Build initial state
            initial_state = {
                "messages": [{"role": "user", "content": user_input}],
                "images": prepared_images if prepared_images else None,
                "metadata": {
                    "session_id": id(self),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Process through graph
            result = self.graph.invoke(initial_state)
            
            # Store in session history
            self.session_history.append({
                "input": user_input,
                "output": result["messages"][-1],
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "I encountered an unexpected error. Please try again or rephrase your question.",
                    "metadata": {"error": str(e)}
                }]
            }
    
    def get_response_content(self, result: Dict[str, Any]) -> str:
        """Extract clean response content"""
        try:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, 'content'):
                return last_msg.content
            else:
                return last_msg.get("content", "No response generated.")
        except (KeyError, IndexError):
            return "No response generated."
    
    def get_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract response metadata"""
        try:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, 'additional_kwargs'):
                return last_msg.additional_kwargs.get("metadata", {})
            else:
                return last_msg.get("metadata", {})
        except (KeyError, IndexError):
            return {}

# --- DEMO AND TESTING ---
def run_enhanced_demo():
    """Run demonstration of enhanced capabilities"""
    chat = EnhancedChatInterface()
    
    # Test cases
    test_cases = [
        {
            "input": "What are the latest developments in quantum computing?",
            "description": "Current events query - should trigger search"
        },
        {
            "input": "Explain how photosynthesis works",
            "description": "Knowledge-based query - should use KB/logical reasoning"
        },
        {
            "input": "What did we discuss about quantum computing earlier?",
            "description": "Follow-up query - should reference conversation history"
        }
    ]
    
    print("🚀 Enhanced LangGraph Chat System Demo")
    print("=" * 50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['description']}")
        print(f"Input: {test['input']}")
        print("-" * 30)
        
        result = chat.process_message(test['input'])
        response = chat.get_response_content(result)
        metadata = chat.get_metadata(result)
        
        print(f"Response: {response}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        print("-" * 50)

graph = build_enhanced_graph()
if __name__ == "__main__":
    # Initialize the enhanced system
    try:
        graph = build_enhanced_graph()
        print("✅ Enhanced graph built successfully!")
        
        # Example usage
        chat_interface = EnhancedChatInterface()
        
        # Example 1: Knowledge-based query
        print("\n🧠 Testing knowledge-based query...")
        result1 = chat_interface.process_message("Explain the theory of relativity in simple terms")
        print("Response:", chat_interface.get_response_content(result1))
        
        # Example 2: Search query
        print("\n🔍 Testing search query...")
        result2 = chat_interface.process_message("What are the latest Olympic records set in 2024?")
        print("Response:", chat_interface.get_response_content(result2))
        
        # Example 3: Image query (if image path exists)
        print("\n🖼️ Testing image query...")
        try:
            # Replace with your actual image path
            image_path = '/Users/maryamsaad/Downloads/IMG_8082 Medium.jpeg'
            if os.path.exists(image_path):
                result3 = chat_interface.process_message("Describe this image in detail", [image_path])
                print("Response:", chat_interface.get_response_content(result3))
            else:
                print("Image file not found - skipping image test")
        except Exception as e:
            print(f"Image test failed: {e}")
        
        # Run full demo
        # run_enhanced_demo()
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        print(f"❌ Error: {e}")