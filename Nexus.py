import pandas as pd
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import textwrap
import urllib.parse

print("Initializing AI Movie Recommender Environment...")

# We are using a robust, highly detailed TMDB 5000 Movies Dataset hosted on a reliable academic repo
DATASET_URL = 'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv'

print("Downloading movie database (~4,700 movies)...")
df = pd.read_csv(DATASET_URL)

# Fill empty values ONLY in the text columns to prevent pandas strict dtype errors
text_columns = ['Movie_Title', 'Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Overview', 'Movie_Cast']
for col in text_columns:
    df[col] = df[col].fillna('')

# We create a "Rich Text Feature" for every movie by combining its best metadata.
# This gives the AI maximum context about what the movie is actually about.
df['ai_context'] = (
    df['Movie_Title'] + " " + 
    df['Movie_Genre'] + " " + 
    df['Movie_Keywords'] + " " + 
    df['Movie_Tagline'] + " " + 
    df['Movie_Overview'] + " " +
    df['Movie_Cast']
)

print("Loading Deep Learning NLP Model (this will take about 30 seconds)...")
# 'all-MiniLM-L6-v2' is a blazingly fast, highly accurate semantic sentence transformer.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate mathematical embeddings (vectors) for all 4,700+ movies based on their 'ai_context'
print("Encoding movie database into neural vectors... (Happens only once)")
corpus_embeddings = model.encode(df['ai_context'].tolist(), show_progress_bar=True, convert_to_tensor=True)

print("Loading Generative LLM for Agentic Reasoning...")
from transformers import pipeline
import torch

# We use a highly efficient, quantized LLM (like TinyLlama or a small T5) 
# that can run smoothly on Colab's free T4 GPU to generate the reasoning.
try:
    generator = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Warning: GPU not found or model failed to load. Falling back to CPU/Smaller model. {e}")
    generator = pipeline('text2text-generation', model='google/flan-t5-small')

def get_agentic_recommendations(user_query, num_results=3):
    """
    Advanced Agentic RAG Pipeline:
    1. Semantic Search (Retrieval)
    2. LLM Prompt Construction (Augmentation)
    3. Custom AI Reasoning (Generation)
    """
    if not user_query.strip():
        return "Please enter a prompt to get started.", pd.DataFrame()
        
    # Phase 1: Retrieval (Vector Search)
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=num_results)[0]
    
    results_for_df = []
    context_for_llm = ""
    
    for i, hit in enumerate(hits):
        idx = hit['corpus_id']
        row = df.iloc[idx]
        
        # Generate dynamic Google Search link for Streaming and Reviews
        search_query = urllib.parse.quote(f"{row['Movie_Title']} movie where to watch reviews")
        google_link = f"[🍿 Stream & Reviews](https://www.google.com/search?q={search_query})"
        
        # Build Dataframe for UI
        results_for_df.append({
            "Movie Title": row['Movie_Title'],
            "Genres": row['Movie_Genre'].replace(",", ", "),
            "Rating": f"⭐ {row['Movie_Vote']}/10",
            "Action": google_link
        })
        
        # Build Context for the LLM
        context_for_llm += f"\nOption {i+1}: {row['Movie_Title']}. Plot: {row['Movie_Overview']}"

    # Phase 2 & 3: Augmentation and Generation
    prompt = f"""<|system|>
You are an expert movie critic and recommendation agent. A user has asked for a movie recommendation based on this prompt: "{user_query}".
Based on your database search, you have found the following options: {context_for_llm}
Write a short, engaging paragraph recommending the best option from this list and explaining exactly why it fits their prompt. Focus on the themes and plot.
<|user|>
Give me your recommendation.
<|assistant|>
"""
    
    try:
        # Generate the agentic response
        llm_response = generator(prompt, max_new_tokens=150, temperature=0.7, truncation=True)[0]['generated_text']
        # Clean up the output to only show the assistant's response
        final_insight = llm_response.split("<|assistant|>")[-1].strip()
    except Exception as e:
        final_insight = f"Agentic reasoning temporarily unavailable. Displaying raw semantic matches below."

    return final_insight, pd.DataFrame(results_for_df)

print("Launching Gradio Agentic Interface...")

custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="Nexus Agentic Recommender") as demo:
    
    with gr.Row():
        gr.Markdown(
            """
            # 🧠 Agentic RAG Movie Recommender
            ### Powered by SentenceTransformers + Local LLM Reasoning
            This system doesn't just find movies; an AI Agent reads the plots of the top matches and writes a custom, personalized explanation of why you should watch them based on your exact prompt.
            """
        )
        
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Describe your perfect movie vibe...", 
                placeholder="e.g. 'A dark, psychological thriller where the protagonist questions their own memory.'",
                lines=3
            )
            submit_btn = gr.Button("✨ Ask the AI Agent", variant="primary")
            
        with gr.Column(scale=3):
            agent_output = gr.Textbox(
                label="AI Agent's Custom Reasoning", 
                lines=5,
                interactive=False
            )
            
    with gr.Row():
        gr.Markdown("### Raw Semantic Matches from Database")
    with gr.Row():
        output_df = gr.Dataframe(
            headers=["Movie Title", "Genres", "Rating", "Action"],
            datatype=["str", "str", "str", "markdown"],
            interactive=False,
            wrap=True
        )
        
    # Connect UI to Python Logic
    submit_btn.click(
        fn=get_agentic_recommendations, 
        inputs=[query_input], 
        outputs=[agent_output, output_df]
    )

# debug=True ensures errors print to the Colab console, inline=True embeds it in the notebook
demo.launch(debug=True, inline=True, share=True)
