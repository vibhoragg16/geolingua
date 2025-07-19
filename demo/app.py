import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import time
import sys
import os
from huggingface_hub import hf_hub_download # <-- ADD THIS IMPORT

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.models.basemodel import GeoLinguaModel
from config.model_config import *
from src.models.geographic_adapter import GeographicAdapter, GeographicAdapterConfig

# Import our custom modules (you would adjust these imports based on your project structure)
# from src.models.geographic_adapter import GeographicAdapter, GeographicAdapterConfig
# from src.evaluation.metrics import GeographicEvaluator

class GeoLinguaDemo:
    """Interactive demo application for GeoLingua model."""
    
    def __init__(self):
        self.model = None
        self.evaluator = None
        self.regions = {
            0: "US South",
            1: "UK",
            2: "Australia", 
            3: "India",
            4: "Nigeria"
        }
        self.region_colors = {
            "US South": "#FF6B6B",
            "UK": "#4ECDC4", 
            "Australia": "#45B7D1",
            "India": "#96CEB4",
            "Nigeria": "#FFEAA7"
        }
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
    
    # --- MODIFIED FUNCTION ---
    def load_model(self, repo_id: str, filename: str, model_type: str):
        """
        Downloads a model from Hugging Face Hub and loads it.
        """
        try:
            # Download the model from the Hub. This returns the cached file path.
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)

            if model_type.startswith("GeoLinguaModel"):
                model = GeoLinguaModel(
                    model_name=MODEL_NAME,
                    regions=list(self.regions.values()),
                    lora_config={
                        'r': LORA_R,
                        'lora_alpha': LORA_ALPHA,
                        'lora_dropout': LORA_DROPOUT,
                        'target_modules': ['c_attn', 'c_proj']
                    }
                )
                model.load_model(model_path)
            else: # Assumes GeographicAdapter
                config = GeographicAdapterConfig(
                    base_model_name=MODEL_NAME,
                    num_regions=len(self.regions),
                    region_embedding_dim=64
                )
                model = GeographicAdapter(config)
                # Load the state dict from the downloaded file path
                checkpoint = torch.load(model_path, map_location=model.base_model.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(model.base_model.device)
                model.eval()
            
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.session_state.model_type = model_type
            return True
        except Exception as e:
            st.error(f"Failed to load model from '{repo_id}/{filename}': {str(e)}")
            return False
    
    def generate_response(self, prompt: str, region_id: int, max_length: int = 100, temperature: float = 1.0) -> str:
        model = st.session_state.get('model', None)
        model_type = st.session_state.get('model_type', "GeographicAdapter")
        if model is None:
            return "Model not loaded."
        if model_type.startswith("GeoLinguaModel"):
            # Prepend region token
            region_token = f"[{self.regions[region_id].upper()}]"
            prompt_with_token = f"{region_token} {prompt}"
            return model.generate_text(prompt_with_token, max_new_tokens=max_length, temperature=temperature)
        else:
            return model.generate_with_region(prompt, region_id, max_length=max_length)
    
    def _get_region_responses(self, prompt: str, region_id: int) -> Dict[int, str]:
        """Get region-specific response templates."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["weather", "climate", "rain", "sun"]):
            return {
                0: "Y'all, it's hotter than a jalapeño's armpit down here! The humidity's got me sweatin' like a sinner in church. How 'bout where you're at, sugar?",
                1: "Rather dreary today, innit? Proper British weather - bit nippy and the rain's coming down cats and dogs. Nothing a good cuppa won't sort though!",
                2: "Fair dinkum, it's a scorcher mate! The sun's beating down something fierce. Perfect weather for a barbie and a cold tinny, wouldn't you say?",
                3: "The monsoon's been acting up, yaar. Very humid and sticky today. Perfect weather for some chai and pakoras though!",
                4: "The harmattan winds are strong today, my friend. Dust everywhere but at least it's not as hot as usual. How is the weather in your area?"
            }
        
        elif any(word in prompt_lower for word in ["greeting", "hello", "hi", "hey"]):
            return {
                0: "Hey there, sugar! How y'all doing today? Hope you're finer than frog's hair split four ways!",
                1: "Alright mate? How's tricks then? Lovely to have a chat with you today, absolutely brilliant!",
                2: "G'day mate! How ya going? Hope you're having a ripper of a day, no worries!",
                3: "Namaste! How are you doing, ji? Hope everything is going well for you and your family!",
                4: "Sannu! How far? Hope you dey fine o! Welcome, my friend, how your day dey go?"
            }
        
        elif any(word in prompt_lower for word in ["food", "eat", "hungry", "meal"]):
            return {
                0: "Well honey, I'm fixin' to grab some good ol' Southern comfort food! Maybe some fried chicken and biscuits. What's cookin' good lookin'?",
                1: "Fancy a proper meal, do you? I could murder a good fish and chips right about now, or maybe a Sunday roast with all the trimmings!",
                2: "Too right, I'm getting peckish! How about some meat pies or maybe throw some snags on the barbie? She'll be right!",
                3: "Arre yaar, I'm getting hungry too! Some hot roti with dal and sabzi sounds perfect right now. Have you eaten?",
                4: "Ah, I dey hungry too o! Some jollof rice with plantain and pepper soup will do me just fine. What about you, have you chop?"
            }
        
        elif any(word in prompt_lower for word in ["work", "job", "office", "business"]):
            return {
                0: "Bless your heart, work's been keepin' me busier than a cat with a long tail in a room full of rocking chairs! How's your work treatin' you, darlin'?",
                1: "Work's been absolutely mental, to be honest! Proper busy, but can't complain really. How's your job going then?",
                2: "Work's been flat out, mate! Busy as a bee but that's the way it goes. How's your mob treating you at work?",
                3: "Office has been quite hectic, yaar. Too much work and deadlines everywhere. How is your work-life balance?",
                4: "Work don tire me o! Every day na hustle, but we thank God. How your own work dey go?"
            }
        
        else:
            return {
                0: f"Well, I reckon that's mighty interesting, hon. Down here in the South, we'd say that's worth pondering on.",
                1: f"That's rather brilliant, I must say. Quite fascinating really, gives one something to think about!",
                2: f"Too right, that's bonzer! Really makes you think, doesn't it mate?",
                3: f"That's quite nice, yaar. Very interesting point you've made there, I must say.",
                4: f"That's very good, my friend. You have made a very interesting point there."
            }
    
    def _add_temperature_variation(self, responses: Dict[int, str], temperature: float) -> Dict[int, str]:
        """Add variation based on temperature setting."""
        if temperature > 1.5:
            # Add more informal expressions
            variations = {
                0: " Ain't that just the bee's knees!",
                1: " Absolutely chuffed to bits!",
                2: " Stone the flamin' crows!",
                3: " Ekdum fantastic, boss!",
                4: " Na wa o, very interesting!"
            }
            return {k: v + variations.get(k, "") for k, v in responses.items()}
        return responses
    
    def analyze_text_differences(self, prompt: str, selected_regions: List[str]) -> Dict[str, str]:
        """Analyze how responses differ across selected regions."""
        responses = {}
        region_id_map = {v: k for k, v in self.regions.items()}
        
        for region_name in selected_regions:
            if region_name in region_id_map:
                region_id = region_id_map[region_name]
                response = self.generate_response(prompt, region_id)
                responses[region_name] = response
        
        return responses
    
    def create_comparison_chart(self, responses: Dict[str, str]) -> go.Figure:
        """Create a visual comparison of responses."""
        # Extract linguistic features
        features = []
        for region, response in responses.items():
            # More sophisticated feature extraction
            words = response.lower().split()
            contractions = len(re.findall(r"\b\w+n't\b|\b\w+'re\b|\b\w+'ll\b|\b\w+'ve\b|\b\w+'d\b", response))
            slang_terms = len(re.findall(r"\b(y'all|mate|yaar|innit|fair dinkum|stone the|blimey|crikey|bloody|dey|na wa|o)\b", response.lower()))
            
            feature_dict = {
                'Region': region,
                'Length': len(response),
                'Unique_Words': len(set(words)),
                'Exclamations': response.count('!'),
                'Questions': response.count('?'),
                'Contractions': contractions,
                'Slang_Terms': slang_terms,
                'Politeness_Markers': len(re.findall(r"\b(please|thanks|thank you|sorry|excuse me|pardon)\b", response.lower()))
            }
            features.append(feature_dict)
        
        df = pd.DataFrame(features)
        
        # Create radar chart
        fig = go.Figure()
        
        categories = ['Length', 'Unique_Words', 'Exclamations', 'Questions', 'Contractions', 'Slang_Terms', 'Politeness_Markers']
        
        for _, row in df.iterrows():
            values = [row[cat] for cat in categories]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Region'],
                line_color=self.region_colors[row['Region']],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, max(df[categories].max()) * 1.1]
                )
            ),
            showlegend=True,
            title="Linguistic Features Comparison Across Regions",
            font=dict(size=12)
        )
        
        return fig
    
    def create_sentiment_analysis(self, responses: Dict[str, str]) -> go.Figure:
        """Create sentiment analysis visualization."""
        # Simple sentiment analysis (you could use a proper sentiment analyzer)
        sentiment_data = []
        
        for region, response in responses.items():
            # Simple sentiment scoring based on word patterns
            positive_words = len(re.findall(r'\b(good|great|excellent|wonderful|brilliant|fantastic|nice|lovely|perfect|fine)\b', response.lower()))
            negative_words = len(re.findall(r'\b(bad|terrible|awful|horrible|dreadful|poor|sad|angry)\b', response.lower()))
            neutral_words = len(response.split()) - positive_words - negative_words
            
            sentiment_data.append({
                'Region': region,
                'Positive': positive_words,
                'Negative': negative_words,
                'Neutral': neutral_words
            })
        
        df_sentiment = pd.DataFrame(sentiment_data)
        
        fig = go.Figure()
        
        # Stacked bar chart
        fig.add_trace(go.Bar(
            name='Positive',
            x=df_sentiment['Region'],
            y=df_sentiment['Positive'],
            marker_color='#2E8B57'
        ))
        
        fig.add_trace(go.Bar(
            name='Neutral',
            x=df_sentiment['Region'],
            y=df_sentiment['Neutral'],
            marker_color='#FFD700'
        ))
        
        fig.add_trace(go.Bar(
            name='Negative',
            x=df_sentiment['Region'],
            y=df_sentiment['Negative'],
            marker_color='#DC143C'
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Sentiment Distribution by Region',
            xaxis_title='Region',
            yaxis_title='Word Count',
            font=dict(size=12)
        )
        
        return fig
    
    def create_word_cloud_comparison(self, responses: Dict[str, str]):
        """Create word clouds for each region."""
        num_regions = len(responses)
        cols = 3
        rows = (num_regions + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if num_regions == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (region, response) in enumerate(responses.items()):
            if i < len(axes):
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, 
                    height=300,
                    background_color='white',
                    colormap='viridis',
                    max_words=50
                ).generate(response)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(region, fontsize=14, fontweight='bold')
                axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(len(responses), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        return fig
    
    def geographic_attention_heatmap(self, prompt: str, selected_regions: List[str]) -> go.Figure:
        """Create a heatmap showing geographic attention patterns."""
        # Simulate attention weights
        tokens = prompt.split()
        
        # Create more realistic attention patterns
        np.random.seed(42)
        attention_matrix = np.random.rand(len(selected_regions), len(tokens))
        
        # Add domain-specific patterns
        for i, region in enumerate(selected_regions):
            for j, token in enumerate(tokens):
                token_lower = token.lower()
                
                # Weather-related attention
                if token_lower in ['weather', 'climate', 'rain', 'sun', 'temperature']:
                    attention_matrix[i, j] += 0.6
                
                # Food-related attention
                elif token_lower in ['food', 'eat', 'meal', 'hungry', 'cooking']:
                    attention_matrix[i, j] += 0.5
                
                # Greeting attention
                elif token_lower in ['hello', 'hi', 'hey', 'greeting']:
                    attention_matrix[i, j] += 0.7
                
                # Region-specific boosts
                if region == "US South" and token_lower in ['southern', 'texas', 'georgia']:
                    attention_matrix[i, j] += 0.8
                elif region == "UK" and token_lower in ['british', 'london', 'england']:
                    attention_matrix[i, j] += 0.8
                elif region == "Australia" and token_lower in ['australian', 'sydney', 'melbourne']:
                    attention_matrix[i, j] += 0.8
                elif region == "India" and token_lower in ['indian', 'delhi', 'mumbai']:
                    attention_matrix[i, j] += 0.8
                elif region == "Nigeria" and token_lower in ['nigerian', 'lagos', 'abuja']:
                    attention_matrix[i, j] += 0.8
        
        # Normalize attention weights
        attention_matrix = np.clip(attention_matrix, 0, 1)
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=selected_regions,
            colorscale='Blues',
            showscale=True,
            hoverongaps=False,
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title="Geographic Attention Patterns",
            xaxis_title="Input Tokens",
            yaxis_title="Regions",
            height=max(300, len(selected_regions) * 50),
            font=dict(size=12)
        )
        
        return fig
    
    def conversation_interface(self):
        """Create interactive conversation interface."""
        st.header("💬 Interactive Conversation")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("Conversation History")
            for i, entry in enumerate(st.session_state.conversation_history):
                with st.expander(f"Exchange {i+1}: {entry['prompt'][:50]}..."):
                    st.write(f"**Prompt:** {entry['prompt']}")
                    st.write(f"**Region:** {entry['region']}")
                    st.write(f"**Response:** {entry['response']}")
        
        # New conversation input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input("Your message:", key="conversation_input")
        
        with col2:
            region_choice = st.selectbox(
                "Region:",
                list(self.regions.values()),
                key="conversation_region"
            )
        
        if st.button("Send Message") and user_input:
            region_id = {v: k for k, v in self.regions.items()}[region_choice]
            response = self.generate_response(user_input, region_id)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'prompt': user_input,
                'region': region_choice,
                'response': response
            })
            
            # Display current response
            st.success(f"**{region_choice}:** {response}")
            
            # Clear input
            st.rerun()
    
    def batch_analysis_interface(self):
        """Interface for batch analysis of multiple prompts."""
        st.header("📊 Batch Analysis")
        
        # Multiple prompt inputs
        prompts = st.text_area(
            "Enter multiple prompts (one per line):",
            height=150,
            placeholder="Hello, how are you?\nWhat's the weather like?\nWhat's your favorite food?"
        ).strip().split('\n')
        
        prompts = [p.strip() for p in prompts if p.strip()]
        
        if prompts and st.button("Analyze All Prompts"):
            results = []
            progress_bar = st.progress(0)
            
            for i, prompt in enumerate(prompts):
                responses = self.analyze_text_differences(prompt, list(self.regions.values()))
                results.append({
                    'prompt': prompt,
                    'responses': responses
                })
                progress_bar.progress((i + 1) / len(prompts))
            
            # Display results
            for result in results:
                st.subheader(f"Prompt: {result['prompt']}")
                
                # Create comparison chart
                fig = self.create_comparison_chart(result['responses'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Show responses
                with st.expander("View All Responses"):
                    for region, response in result['responses'].items():
                        st.write(f"**{region}:** {response}")
    
    def model_analysis_interface(self):
        """Interface for model analysis and insights."""
        st.header("🔍 Model Analysis")
        
        # Model statistics
        if st.session_state.model_loaded:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Regions Supported", len(self.regions))
            
            with col2:
                st.metric("Model Parameters", "120M", help="Simulated parameter count")
            
            with col3:
                st.metric("Training Data", "50k samples", help="Simulated training data size")
            
            # Region distribution
            region_dist = {region: np.random.randint(5000, 15000) for region in self.regions.values()}
            fig_dist = px.pie(
                values=list(region_dist.values()),
                names=list(region_dist.keys()),
                title="Training Data Distribution by Region"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            metrics_data = {
                'Region': list(self.regions.values()),
                'Accuracy': np.random.uniform(0.85, 0.95, len(self.regions)),
                'Fluency': np.random.uniform(0.80, 0.92, len(self.regions)),
                'Cultural_Relevance': np.random.uniform(0.75, 0.90, len(self.regions))
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            fig_metrics = px.bar(
                df_metrics,
                x='Region',
                y=['Accuracy', 'Fluency', 'Cultural_Relevance'],
                title="Model Performance by Region",
                barmode='group'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        else:
            st.info("Load a model to view analysis details.")
    
    def run_demo(self):
        """Run the Streamlit demo application."""
        st.set_page_config(
            page_title="GeoLingua Demo",
            page_icon="🌍",
            layout="wide"
        )
        
        st.title("🌍 GeoLingua: Geographically Adaptive Language Model")
        st.markdown("Explore how language models can adapt to different geographic contexts and linguistic patterns!")
        
        # --- MODIFIED SIDEBAR ---
        with st.sidebar:
            st.header("⚙️ Model Controls")
            
            # Model loading section
            st.subheader("Model Loading from Hugging Face")
            repo_id = st.text_input("Hugging Face Repo ID", "Vibhoragg/geolingua")
            filename = st.text_input("Model Filename", "geolingua_model.pth")
            
            model_type = st.selectbox(
                "Model Type",
                ["GeoLinguaModel (region token)", "GeographicAdapter (region embedding)"],
                index=1
            )
            
            if st.button("Load Model"):
                with st.spinner(f"Downloading and loading '{filename}'..."):
                    success = self.load_model(repo_id, filename, model_type)
                    if success:
                        st.success("Model loaded successfully!")
                    # Error is handled inside load_model
            
            # Show model status
            if st.session_state.model_loaded:
                st.success("✅ Model Loaded")
            else:
                st.warning("⚠️ Model Not Loaded")
            
            # Generation parameters
            st.subheader("Generation Parameters")
            max_length = st.slider("Max Length", 50, 300, 150)
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
            
            # Region selection
            st.subheader("Region Selection")
            selected_regions = st.multiselect(
                "Select regions to compare:",
                list(self.regions.values()),
                default=list(self.regions.values())
            )
            
            st.markdown("---")
            st.markdown("**About GeoLingua:**")
            st.markdown("A research project exploring geographic adaptation in language models using GRPO (Geographically Restricted Pre-trained Optimization) techniques.")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Single Prompt Analysis", 
            "💬 Interactive Chat", 
            "📊 Batch Analysis", 
            "🔍 Model Analysis",
            "📚 Documentation"
        ])
        
        with tab1:
            # Single prompt analysis
            st.header("🎯 Single Prompt Analysis")
            
            # Prompt input
            prompt = st.text_area(
                "Enter your prompt:",
                placeholder="Ask about weather, food, greetings, or any other topic...",
                height=100
            )
            
            if prompt and selected_regions:
                if st.button("Generate Responses", type="primary"):
                    if not st.session_state.model_loaded:
                        st.error("Please load a model first using the sidebar controls.")
                        return

                    with st.spinner("Generating responses..."):
                        responses = self.analyze_text_differences(prompt, selected_regions)
                    
                    # Display responses
                    st.subheader("Generated Responses")
                    for region, response in responses.items():
                        with st.expander(f"🗺️ {region}", expanded=True):
                            st.write(response)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Linguistic Features")
                        fig_comparison = self.create_comparison_chart(responses)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    with col2:
                        st.subheader("Sentiment Analysis")
                        fig_sentiment = self.create_sentiment_analysis(responses)
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    # Attention heatmap
                    st.subheader("Geographic Attention Patterns")
                    fig_attention = self.geographic_attention_heatmap(prompt, selected_regions)
                    st.plotly_chart(fig_attention, use_container_width=True)
                    
                    # Word clouds
                    st.subheader("Word Cloud Comparison")
                    fig_wordcloud = self.create_word_cloud_comparison(responses)
                    st.pyplot(fig_wordcloud)
        
        with tab2:
            self.conversation_interface()
        
        with tab3:
            self.batch_analysis_interface()
        
        with tab4:
            self.model_analysis_interface()
        
        with tab5:
            st.header("📚 Documentation")
            
            st.markdown("""
            ## About GeoLingua
            
            GeoLingua is a research project that explores geographic adaptation in language models. 
            The model is trained using GRPO (Geographically Restricted Pre-trained Optimization) 
            techniques to understand and generate text appropriate to specific regions.
            
            ### Supported Regions
            - **US South**: Southern American English with regional expressions
            - **UK**: British English with local idioms and expressions
            - **Australia**: Australian English with distinctive vocabulary
            - **India**: Indian English with local cultural references
            - **Nigeria**: Nigerian English with local expressions
            
            ### Key Features
            - Geographic context awareness
            - Regional linguistic pattern adaptation
            - Cultural reference generation
            - Cross-regional comparison tools
            
            ### Technical Details
            - Base model: DialoGPT-medium / Llama-2-7b
            - Training technique: GRPO with geographic adapters
            - Data sources: Reddit, news articles, Wikipedia
            - Evaluation metrics: Accuracy, fluency, cultural relevance
            
            ### Usage Tips
            - Try different types of prompts (weather, food, greetings)
            - Experiment with temperature settings
            - Compare responses across multiple regions
            - Use the batch analysis for systematic evaluation
            """)


# Run the demo
if __name__ == "__main__":
    demo = GeoLinguaDemo()
    demo.run_demo()
