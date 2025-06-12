import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

class PortfolioChatbot:
    def __init__(self):
        # Initialize Gemini LLM
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(
            model_name="gemini-2.0-flash", 
            generation_config={"temperature": 0.9}
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("portfolio") 
        
        # Initialize embedding model (same as used for upload)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # System prompt for the chatbot
        self.system_prompt = """
        You are a helpful AI assistant representing a job candidate's portfolio. 
        You have access to their resume, projects, skills, and experience details.
        
        Your role is to:
        - Answer questions about the candidate's background, skills, and experience
        - Provide specific details from their portfolio when asked
        - Be professional, friendly, and informative
        - Help recruiters understand why this candidate would be a good fit
        - If you don't have specific information, politely say so and suggest contacting the candidate directly
        
        Always respond in a conversational, professional manner as if you're representing the candidate.
        """

    def search_portfolio(self, query, top_k=3):
        """Search the portfolio database for relevant information"""
        try:
            # Convert query to embedding
            query_vector = self.embedding_model.embed_query(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k
            )
            matches = results.get("matches", [])
            texts = [m["metadata"]["text"] for m in matches if "metadata" in m and "text" in m["metadata"]]
            print("Text",texts)
            return texts
            
    
            
        except Exception as e:
            print(f"Error searching portfolio: {e}")
            return []

    def generate_response(self, user_query, context_texts):
        """Generate response using Gemini LLM with context"""
        
        # Prepare context from search results
        context = "\n\n".join(context_texts) if context_texts else "No specific information found in portfolio."
        
        # Create the full prompt
        full_prompt = f"""
        {self.system_prompt}
        
        CONTEXT FROM PORTFOLIO:
        {context}
        
        RECRUITER'S QUESTION: {user_query}
        
        Please provide a helpful, professional response based on the portfolio information above.
        If the context doesn't contain relevant information, politely acknowledge this and provide general guidance.
        """
        
        try:
            response = self.llm.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your question right now. Please try again or contact the candidate directly. Error: {e}"

    def chat(self, user_input):
        """Main chat function"""
        # Search for relevant portfolio information
        relevant_info = self.search_portfolio(user_input)
        
        # Generate response using LLM
        response = self.generate_response(user_input, relevant_info)
        
        return response

def main():
    """Interactive chat loop"""
    chatbot = PortfolioChatbot()
    
    print("🤖 Portfolio Assistant: Hello! I'm here to help you learn about this candidate.")
    print("Ask me anything about their skills, experience, projects, or background.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        user_input = input("Recruiter: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("🤖 Portfolio Assistant: Thank you for your interest! Feel free to reach out to the candidate directly for further discussions.")
            break
        
        if not user_input:
            continue
            
        print("🤖 Portfolio Assistant: Let me check the portfolio for you...")
        
        response = chatbot.chat(user_input)
        print(f"🤖 Portfolio Assistant: {response}\n")

# Example usage functions
def quick_query(question):
    """Function to ask a single question without interactive loop"""
    chatbot = PortfolioChatbot()
    return chatbot.chat(question)

if __name__ == "__main__":
    # You can either run the interactive chat or use quick_query
    
    # Option 1: Interactive chat
    main()
    
    # Option 2: Single query example (uncomment to use)
    # response = quick_query("What are the candidate's main technical skills?")
    # print(response)