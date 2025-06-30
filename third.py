import nltk
from nltk.chat.util import Chat, reflections
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load spaCy's English language model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import spacy.cli
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Basic pattern responses
pairs = [
    [
        r"(hi|hello|hey|greetings)",
        ["Hello! How can I help you today?", "Hi there!", "Greetings! What can I do for you?"]
    ],
    [
        r"how are you ?",
        ["I'm doing well, thank you!", "I'm a chatbot, so I don't have feelings, but I'm functioning well!", "All systems are operational!"]
    ],
    [
        r"(bye|goodbye|exit|quit)",
        ["Goodbye! Have a great day!", "It was nice talking to you. Bye!", "See you later!"]
    ],
    [
        r"(thanks|thank you|appreciate it)",
        ["You're welcome!", "Happy to help!", "No problem!"]
    ]
]

# Knowledge base with exact output phrases
knowledge_base = {
    "hours": "We're open from 9 AM to 5 PM, Monday through Friday.",
    "located": "Our office is located at 123 Main Street, Tech City.",
    "location": "Our office is located at 123 Main Street, Tech City.",
    "what do you do": "We provide AI consulting, software development, data analysis services, and intelligent chatbot solutions like ChatGPT.",
    "services": "We specialize in AI consulting, custom software development, data analytics, and intelligent chatbot solutions inspired by models like ChatGPT."
}

# Enhanced chatbot class
class SmartChatBot:
    def __init__(self):
        self.nlp = nlp
        self.basic_chat = Chat(pairs, reflections)
        self.vectorizer = TfidfVectorizer()
        
        # Prepare knowledge base for similarity matching
        self.knowledge_questions = list(knowledge_base.keys())
        self.knowledge_answers = list(knowledge_base.values())
        
        # Vectorize knowledge base questions
        self.vectorizer.fit(self.knowledge_questions)
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)
    
    def get_response(self, user_input):
        # First try basic pattern matching
        response = self.basic_chat.respond(user_input)
        if response:
            return response

        # Try direct match in knowledge base (case-insensitive, substring match)
        lowered = user_input.lower()
        # Map user questions to real responses based on intent
        if "hour" in lowered:
            return "We're open from 9 AM to 5 PM, Monday through Friday."
        if "locate" in lowered:
            return "Our office is located at 123 Main Street, Tech City."
        if "what do you do" in lowered:
            return "We provide AI consulting, software development, and data analysis services."
        if "services" in lowered:
            return "We specialize in AI consulting, custom software development, data analytics, and intelligent chatbot solutions inspired by models like ChatGPT."

        # If no pattern match, try similarity
        processed_input = self.preprocess_text(user_input)
        kb_vectors = self.vectorizer.transform(self.knowledge_questions)
        input_vector = self.vectorizer.transform([processed_input])

        similarities = cosine_similarity(input_vector, kb_vectors)
        max_similarity = similarities.max()

        if max_similarity > 0.5:
            best_match_idx = similarities.argmax()
            return self.knowledge_answers[best_match_idx]

        # Fallback response
        return "I'm not sure I understand. Could you rephrase that?"

# Add color formatting for beautiful console output
def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def run_chatbot():
    print(color_text("╔══════════════════════════════════════════════╗", "96"))
    print(color_text("║        🤖 Welcome to SmartChatBot!          ║", "96"))
    print(color_text("╚══════════════════════════════════════════════╝", "96"))
    print(color_text("Type your question below. Type 'bye' to exit.\n", "93"))
    chatbot = SmartChatBot()

    conversation = []  # Store conversation history

    while True:
        try:
            user_input = input(color_text("You: ", "92"))
            if user_input.lower() in ['quit', 'exit', 'bye']:
                bot_reply = "Goodbye! Have a great day! 👋"
                conversation.append(("You", user_input))
                conversation.append(("Chatbot", bot_reply))
                print(color_text(f"Chatbot: {bot_reply}", "95"))
                print(color_text("\n════════════════ Conversation Summary ════════════════", "96"))
                for speaker, msg in conversation:
                    if speaker == "You":
                        print(color_text(f"{speaker}: {msg}", "92"))
                    else:
                        print(color_text(f"{speaker}: {msg}", "94"))
                print(color_text("══════════════════════════════════════════════════════", "96"))
                break

            response = chatbot.get_response(user_input)
            conversation.append(("You", user_input))
            conversation.append(("Chatbot", response))
            # Print as a chat bubble style
            print(color_text(f"Chatbot: {response}", "94"))
            print(color_text("─" * 50, "90"))

        except KeyboardInterrupt:
            print(color_text("\nChatbot: Goodbye!", "95"))
            break
        except Exception as e:
            print(color_text(f"Chatbot: Something went wrong. ({str(e)})", "91"))

if __name__ == "__main__":
    # Additional setup for better performance might go here
    run_chatbot()
